"""
Transformer ablation of exp 31 (supervised_31_baseline_clean).

Uses `DirectClassifierNoTransformer` — SmrtEmbedding + default CNN (11
ResBlocks, RF=107, 4x downsample) + classification head, with NO
PositionalEncoding and NO TransformerBlocks. One-variable change from
exp 31: same data, KineticsNorm, optimizer, schedule, metrics, per-epoch
checkpointing. The only thing that differs is the model class.

At ctx=32 the CNN's receptive field (107) already covers the entire input,
so every CNN latent is a function of the full 32 bases. The transformer's
job in the original model is to re-mix those already-global latents. This
ablation tests how much of exp 31's ~82% top-1 actually depends on that
re-mixing — see README.md for the interpretation thresholds.

Checkpoint layout differs from exp 31 in one way: `encoder_state_dict` is
not written. DirectClassifierNoTransformer has no `.encoder` attribute
(submodules sit directly on self), so there is no reusable backbone to
persist under that key. `model_state_dict` remains the full backup.
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
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifierNoTransformer
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


REQUIRED_DATA_KEYS = ['pos_data_train', 'neg_data_train', 'pos_data_val', 'neg_data_val']

DEFAULT = {
    'd_model': 128, 'context': 32,
    'batch_size': 512, 'epochs': 20, 'ds_limit': 0,
    'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
}


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def save_epoch_checkpoint(accelerator, model, config, epoch, metrics, norm_fn, checkpoint_dir):
    """Save a checkpoint for a completed epoch.

    Called at the end of every epoch after eval. Synchronizes all ranks
    before saving so no rank is still writing gradients while the main
    process reads the state dict. The save itself runs only on the main
    process and is wrapped in try/except so filesystem errors surface
    with context instead of crashing the process with a bare traceback.

    epoch is 1-indexed to match human convention ("epoch 1 of 20").

    The normalization stats (`norm_means`, `norm_stds`, `norm_log_transform`)
    are persisted alongside the model so inference code can reconstruct the
    exact training-time transform without re-sampling statistics from the
    training data. Load them via `KineticsNorm.load_stats(ckpt)`.

    Unlike exp 31's save helper, this one does NOT write an
    `encoder_state_dict` key — DirectClassifierNoTransformer has no
    `.encoder` attribute (submodules sit directly on the module), so
    there is no reusable backbone to save separately. `model_state_dict`
    is the sole source of model weights.
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    save_path = os.path.join(checkpoint_dir, f'epoch_{epoch:02d}.pt')
    unwrapped = accelerator.unwrap_model(model)
    try:
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
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
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for key in REQUIRED_DATA_KEYS:
        assert key in config, f"Missing required config key: {key}"

    c = DEFAULT | config.get('classifier', {})
    config['classifier'] = c
    config['git_hash'] = get_git_revision_hash()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='bf16',
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

    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker("tensorboard")
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, "hparams.yaml"), "w") as f:
            yaml.dump(config, f)
        tracker.writer.add_text("Full_Config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # --- Checkpoint directory (set up before training so a bad path fails loud) ---
    # Derived from __file__, not config_path, so caller cwd can't misplace output.
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

    # --- Normalization (capped 2M sample for statistics) ---
    # Same KineticsNorm setup as exp 31. Saved stats are persisted in every
    # checkpoint via `**train_norm_fn.save_stats()` so inference code can
    # reconstruct the exact training-time transform via KineticsNorm.load_stats.
    norm_limit = min(c['ds_limit'], 2_000_000) if c['ds_limit'] > 0 else 2_000_000
    tmp_ds = LabeledMemmapDataset(config['pos_data_train'], config['neg_data_train'], limit=norm_limit)
    train_norm_fn = KineticsNorm(tmp_ds, log_transform=True)
    del tmp_ds

    if accelerator.is_main_process:
        print(f"KineticsNorm stats — means: {train_norm_fn.means}, stds: {train_norm_fn.stds}")

    # --- Datasets and loaders ---
    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=c['ds_limit'], norm_fn=train_norm_fn, balance=True
    )
    val_ds = LabeledMemmapDataset(
        config['pos_data_val'], config['neg_data_val'],
        limit=c['ds_limit'], norm_fn=train_norm_fn
    )

    if accelerator.is_main_process:
        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=c['batch_size'], num_workers=2,
        pin_memory=True, prefetch_factor=4, shuffle=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=c['batch_size'], num_workers=2,
        pin_memory=True, prefetch_factor=4, shuffle=False
    )

    # --- Model ---
    model = DirectClassifierNoTransformer(
        d_model=c['d_model'],
        max_len=c['context'],
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"CNN receptive field: {model.cnn.r0} bases  (ctx = {c['context']})")
        tracker.writer.add_scalar("architecture/cnn_receptive_field", model.cnn.r0, 0)

    # --- Optimizer, loss, schedule ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c['max_lr']), weight_decay=c['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    total_steps = len(train_dl) * c['epochs']

    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(train_dl)}")
        print(f"Total steps: {total_steps}")
        print(f"Effective batch size: {c['batch_size'] * accelerator.num_processes}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=c['pct_start']
    )
    scheduler = accelerator.prepare(scheduler)

    f1_metric, auroc_metric, auprc_metric, acc_metric = accelerator.prepare(
        BinaryF1Score(),
        BinaryAUROC(),
        BinaryAveragePrecision(),
        BinaryAccuracy()
    )

    global_step = 0

    for epoch in range(c['epochs']):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dl, desc=f"Epoch {epoch+1}/{c['epochs']}",
            disable=not accelerator.is_main_process
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
                "epoch": epoch
            }, step=global_step)

            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=f"{loss_reduced:.4f}")

        avg_epoch_loss = epoch_loss / len(train_dl)
        accelerator.log({"epoch_avg_loss": avg_epoch_loss}, step=global_step)

        # --- Eval ---
        model.eval()
        eval_progress = tqdm(
            val_dl, desc=f"Eval {epoch+1}/{c['epochs']}",
            disable=not accelerator.is_main_process
        )

        for x, y in eval_progress:
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

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: loss={avg_epoch_loss:.4f}  top1={epoch_acc:.4f}  f1={epoch_f1:.4f}  auroc={epoch_auroc:.4f}")

        # --- Per-epoch checkpoint ---
        epoch_metrics = {
            'train_loss': avg_epoch_loss,
            'eval_top1': epoch_acc,
            'eval_f1': epoch_f1,
            'eval_auroc': epoch_auroc,
            'eval_auprc': epoch_auprc,
        }
        save_epoch_checkpoint(
            accelerator, model, config, epoch + 1, epoch_metrics, train_norm_fn, checkpoint_dir
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
