"""
Clean rewrite of exp 31 (supervised baseline) with fixed LR scheduler.

The only functional change: the scheduler is no longer wrapped by
accelerator.prepare(), which caused it to not advance at the intended
rate. All architecture, data, optimizer, and metric logic is identical.
"""

import sys
import os
import subprocess
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy,
)

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


REQUIRED_DATA_KEYS = ['pos_data_train', 'neg_data_train', 'pos_data_val', 'neg_data_val']

DEFAULT = {
    'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 32,
    'batch_size': 512, 'epochs': 20, 'ds_limit': 0,
    'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
}


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in REQUIRED_DATA_KEYS:
        assert key in config, f"Missing required config key: {key}"
    c = DEFAULT | config.get('classifier', {})
    config['classifier'] = c
    config['git_hash'] = get_git_revision_hash()
    return config, c


def build_data(config, c):
    """Build normalization, datasets, and DataLoaders."""
    norm_limit = min(c['ds_limit'], 2_000_000) if c['ds_limit'] > 0 else 2_000_000
    tmp_ds = LabeledMemmapDataset(config['pos_data_train'], config['neg_data_train'], limit=norm_limit)
    norm_fn = KineticsNorm(tmp_ds, log_transform=True)
    del tmp_ds

    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=c['ds_limit'], norm_fn=norm_fn, balance=True,
    )
    val_ds = LabeledMemmapDataset(
        config['pos_data_val'], config['neg_data_val'],
        limit=c['ds_limit'], norm_fn=norm_fn,
    )

    train_dl = DataLoader(
        train_ds, batch_size=c['batch_size'], num_workers=2,
        pin_memory=True, prefetch_factor=4, shuffle=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=c['batch_size'], num_workers=2,
        pin_memory=True, prefetch_factor=4, shuffle=False,
    )
    return train_dl, val_dl, norm_fn


def build_model(c):
    return DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context'],
    )


def evaluate(model, val_dl, metrics, accelerator):
    """Run eval, return metrics dict. Resets all metric accumulators."""
    model.eval()
    for x, y in val_dl:
        with torch.no_grad():
            logits = model(x)
        y_hat, y, logits = accelerator.gather_for_metrics((logits > 0, y, logits))
        y_long = y.long()
        metrics['f1'].update(y_hat.squeeze(-1), y_long)
        metrics['auroc'].update(logits.squeeze(-1), y_long)
        metrics['auprc'].update(logits.squeeze(-1), y_long)
        metrics['accuracy'].update(y_hat.squeeze(-1), y_long)

    results = {name: m.compute().item() for name, m in metrics.items()}
    for m in metrics.values():
        m.reset()
    return results


def save_checkpoint(accelerator, model, config, epoch, metrics, norm_fn, checkpoint_dir):
    """Save model + norm stats after a completed epoch."""
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    save_path = os.path.join(checkpoint_dir, f'epoch_{epoch:02d}.pt')
    unwrapped = accelerator.unwrap_model(model)
    try:
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'config': config,
            'epoch': epoch,
            'metrics': metrics,
            **norm_fn.save_stats(),
        }, save_path)
        print(f"Saved checkpoint to {save_path}")
    except Exception as e:
        print(f"ERROR: failed to save checkpoint to {save_path}: {type(e).__name__}: {e}")
        raise


def main():
    config, c = load_config(sys.argv[1])

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with="tensorboard",
        project_dir="training_logs",
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(42)

    # --- Logging setup ---
    project_namespace = f"{config.get('experiment_type', 'supervised')}/{config.get('experiment_name', 'supervised_experiment')}"
    if accelerator.is_main_process:
        print(config)
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker("tensorboard")
        tracker.writer.add_text("Full_Config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # --- Checkpoint directory ---
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

    # --- Data ---
    train_dl, val_dl, norm_fn = build_data(config, c)

    if accelerator.is_main_process:
        print(f"KineticsNorm stats — means: {norm_fn.means}, stds: {norm_fn.stds}")
        print(f"Training samples: {len(train_dl.dataset)}")
        print(f"Validation samples: {len(val_dl.dataset)}")

    # --- Model ---
    model = build_model(c)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"CNN receptive field: {model.encoder.cnn.r0} bases  (ctx = {c['context']})")
        tracker.writer.add_scalar("architecture/cnn_receptive_field", model.encoder.cnn.r0, 0)

    # --- Optimizer + prepare ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c['max_lr']), weight_decay=c['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    # --- Schedule (NOT prepared — stepped manually) ---
    # accelerator.prepare(scheduler) wraps it in AcceleratedScheduler which
    # modifies stepping behaviour and causes pct_start to not work as configured.
    total_steps = len(train_dl) * c['epochs']
    warmup_steps = int(total_steps * c['pct_start'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=c['pct_start'],
    )

    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(train_dl)}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps} (peak LR at step {warmup_steps})")
        print(f"Effective batch size: {c['batch_size'] * accelerator.num_processes}")

    # --- Metrics ---
    device = accelerator.device
    metrics = {
        'f1': BinaryF1Score().to(device),
        'auroc': BinaryAUROC().to(device),
        'auprc': BinaryAveragePrecision().to(device),
        'accuracy': BinaryAccuracy().to(device),
    }

    # --- Training ---
    global_step = 0

    for epoch in range(c['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dl, desc=f"Epoch {epoch+1}/{c['epochs']}",
            disable=not accelerator.is_main_process,
        )

        for x, y in progress_bar:
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
                "epoch": epoch,
            }, step=global_step)

            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{loss_reduced:.4f}")

        avg_epoch_loss = epoch_loss / len(train_dl)
        accelerator.log({"epoch_avg_loss": avg_epoch_loss}, step=global_step)

        # --- Eval ---
        eval_results = evaluate(model, val_dl, metrics, accelerator)

        accelerator.log({
            "eval_f1": eval_results['f1'],
            "eval_auroc": eval_results['auroc'],
            "eval_auprc": eval_results['auprc'],
            "eval_top1": eval_results['accuracy'],
        }, step=global_step)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: loss={avg_epoch_loss:.4f}  "
                  f"top1={eval_results['accuracy']:.4f}  f1={eval_results['f1']:.4f}  "
                  f"auroc={eval_results['auroc']:.4f}")

        # --- Per-epoch checkpoint ---
        epoch_metrics = {
            'train_loss': avg_epoch_loss,
            'eval_top1': eval_results['accuracy'],
            'eval_f1': eval_results['f1'],
            'eval_auroc': eval_results['auroc'],
            'eval_auprc': eval_results['auprc'],
        }
        save_checkpoint(accelerator, model, config, epoch + 1, epoch_metrics, norm_fn, checkpoint_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
