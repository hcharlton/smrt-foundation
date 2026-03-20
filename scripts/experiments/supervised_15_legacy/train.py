import sys
import os
import subprocess
import yaml
import torch
import polars as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LabeledMemmapDataset, LegacyMethylDataset, compute_log_normalization_stats
from smrt_foundation.model import DirectClassifier
from smrt_foundation.optim import get_cosine_schedule_with_warmup

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT = {
        'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 128,
        'batch_size': 64, 'epochs': 10, 'ds_limit': 2_000_000,
        'max_lr': 1e-3, 'temperature': 0.1, 'p_mask': 0.05,
        'weight_decay': 0.02, 'pct_start': 0.4,
    }

    config_updated = DEFAULT | config.get('classifier', {})
    config['classifier'] = config_updated
    config['git_hash'] = get_git_revision_hash()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    print('init accelerator')
    accelerator = Accelerator(
        mixed_precision='no',
        log_with="tensorboard",
        project_dir="training_logs",
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        print(config)

    set_seed(42)

    exp_type = config.get('experiment_type', 'supervised')
    exp_name = config.get('experiment_name', 'supervised_experiment')
    project_namespace = f"{exp_type}/{exp_name}"
    print('init trackers')
    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker("tensorboard")
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, "hparams.yaml"), "w") as f:
            yaml.dump(config, f)
        tracker.writer.add_text("Full_Config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # train_ds = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=config_updated['ds_limit'])
    # train_dl = DataLoader(train_ds, batch_size=config_updated['batch_size'], num_workers=4, pin_memory=True, prefetch_factor=4, shuffle=True)

    # val_ds = LabeledMemmapDataset(config.get('pos_data_val'), config.get('neg_data_val'), limit=config_updated['ds_limit'])
    # val_dl = DataLoader(val_ds, batch_size=config_updated['batch_size'], num_workers=4, pin_memory=True, prefetch_factor=4, shuffle=True)
    print('insantiating stats df')
    q = (
        pl.scan_parquet(config.get('legacy_train'))
        .head(1_000_000)
    )
    df = q.collect()
    print('calculating statistics)')
    KINETICS_FEATURES = ['fi', 'fp', 'ri', 'rp']
    train_means, train_stds = compute_log_normalization_stats(df, KINETICS_FEATURES)
    print('init train ds')
    train_ds = LegacyMethylDataset(config.get('legacy_train'), train_means, train_stds, context=32, restrict_row_groups=10, single_strand=True)
    print(len(train_ds))
    print('init train dl')
    train_dl = DataLoader(train_ds,
                         # num_workers=8,
                        batch_size=256,
                        drop_last=True,
                        persistent_workers=False,
                        # prefetch_factor=5
                          )


    val_ds = LegacyMethylDataset(config.get('legacy_val'),
                                means=train_means,
                                stds=train_stds,
                                context=32,
                                restrict_row_groups=10,
                                single_strand=True)
    val_dl = DataLoader(val_ds,
                        batch_size=256,
                        drop_last=True,
                        persistent_workers=False,
                        prefetch_factor=None)

    model = DirectClassifier(
        d_model=config_updated['d_model'], 
        n_layers=config_updated['n_layers'],
        n_head=config_updated['n_head'], 
        max_len=config_updated['context']
    )
    print('init optimizer')
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config_updated['max_lr']), weight_decay=config_updated['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    print('prepare objects')
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    total_steps = len(train_dl) * config_updated['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=config_updated['pct_start']
    )
    scheduler = accelerator.prepare(scheduler)

    f1_metric, auroc_metric, auprc_metric, acc_metric = accelerator.prepare(
        BinaryF1Score(),
        BinaryAUROC(thresholds=100),
        BinaryAveragePrecision(thresholds=100),
        BinaryAccuracy()
    )

    global_step = 0
    print('begin training')
    for epoch in range(config_updated['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{config_updated['epochs']}", disable=not accelerator.is_main_process)

        for batch in progress_bar:
            x, y = batch['data'], batch['label']
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1).to(torch.float32))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            loss_reduced = accelerator.reduce(loss, reduction="mean").item()
            epoch_loss += loss_reduced

            accelerator.log({
                "train_loss": loss_reduced,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            }, step=global_step)

            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{loss_reduced:.4f}")

        avg_epoch_loss = epoch_loss / len(train_dl)
        accelerator.log({"epoch_avg_loss": avg_epoch_loss}, step=global_step)
        print('begin eval')
        model.eval()
        total_count = 0
        total_correct = 0
        eval_progress = tqdm(val_dl, desc=f"Eval set run {epoch+1}/{config_updated['epochs']}", disable=not accelerator.is_main_process)
        
        for batch in eval_progress:
            x, y = batch['data'], batch['label']
            with torch.no_grad():
                logits = model(x)
                
            y_hat, y, logits = accelerator.gather_for_metrics((logits > 0, y, logits))
            
            f1_metric.update(y_hat.squeeze(-1), y.long())
            auroc_metric.update(logits.squeeze(-1), y.long())
            auprc_metric.update(logits.squeeze(-1), y.long())
            acc_metric.update(y_hat.squeeze(-1), y.long())

        epoch_f1 = f1_metric.compute().item()
        epoch_auroc = auroc_metric.compute().item()
        epoch_auprc = auprc_metric.compute().item()
        epoch_acc = acc_metric.compute().item()

        f1_metric.reset()
        auroc_metric.reset()
        auprc_metric.reset()
        acc_metric.reset()

        accelerator.log({
            "eval_f1": epoch_f1,
            "eval_auroc": epoch_auroc,
            "eval_auprc": epoch_auprc,
            "eval_top1": epoch_acc
        }, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    main()