"""Shared supervised baseline training loop for the size grid.

Adaptation of supervised_40_baseline_v2/train.py to a multi-size grid.
The training loop, model construction, data pipeline, optimizer, and
LR-scheduler logic are verbatim from v40 — including the v40 fix that
keeps `scheduler` outside `accelerator.prepare(...)` so the cosine
horizon doesn't get compressed by `num_processes` (the bug that broke
exp 31's pct_start).

What the shared loop adds on top of v40:

  1. Per-size checkpoint directory keyed off the *config* path, not the
     train script path, so each `size_*/` writes its own
     `checkpoints/` and `results.csv`. Pattern lifted from
     `ssl_57_inputmask_grid_lnhead/_shared_train.py:498`.

  2. Per-epoch CSV at `<size_dir>/results.csv`. One row per completed
     epoch, written by the main process. Schema documented inline at the
     header write. Pattern matches
     `supervised_33_data_scaling/train.py:161-170, 253-260`.

  3. Walltime tracking via `time.perf_counter()` at training start; the
     elapsed seconds are logged in TB and in the CSV per epoch.

  4. `architecture/param_count` logged to TB at step 0 alongside the
     existing `cnn_receptive_field` scalar v40 emits.

Everything else (data, model, evaluator, optimizer, criterion, DDP
prepare order, mixed_precision='bf16', seed=42, per-epoch checkpoint
format including `model_state_dict`, `encoder_state_dict`, `config`,
`epoch`, `metrics`, and `KineticsNorm.save_stats()`) is identical to
v40.
"""

import sys
import os
import csv
import time
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

# head_dim=64 across the grid (matches ssl_55/57 convention). v40 used
# H=4 at d=128 (head_dim=32); the grid switches to H=2 at d=128 to keep
# head_dim constant across sizes — clean cross-size scaling read.
DEFAULT = {
    'd_model': 128, 'n_layers': 4, 'n_head': 2, 'context': 32,
    'batch_size': 512, 'epochs': 60, 'ds_limit': 0,
    'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
}

CSV_HEADER = [
    'epoch', 'global_step', 'walltime_s',
    'train_loss_avg', 'lr_at_epoch_end',
    'eval_top1', 'eval_f1', 'eval_auroc', 'eval_auprc',
    'd_model', 'n_layers', 'n_head',
    'batch_size_per_gpu', 'effective_bs', 'params_count',
]


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
    """Build normalization, datasets, and DataLoaders. Verbatim from v40."""
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
    """Run eval, return metrics dict. Resets all metric accumulators. Verbatim from v40."""
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
    """Save model + norm stats after a completed epoch. Verbatim from v40."""
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
    config_path = sys.argv[1]
    config, c = load_config(config_path)

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

    # --- Per-size paths (keyed off the config dir, not the train script) ---
    size_dir = os.path.dirname(os.path.abspath(config_path))
    checkpoint_dir = os.path.join(size_dir, 'checkpoints')
    csv_path = os.path.join(size_dir, 'results.csv')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"CSV path: {csv_path}")

    # --- Data ---
    train_dl, val_dl, norm_fn = build_data(config, c)

    if accelerator.is_main_process:
        print(f"KineticsNorm stats — means: {norm_fn.means}, stds: {norm_fn.stds}")
        print(f"Training samples: {len(train_dl.dataset)}")
        print(f"Validation samples: {len(val_dl.dataset)}")

    # --- Model ---
    model = build_model(c)

    n_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"Model parameters: {n_params:,}")
        print(f"CNN receptive field: {model.encoder.cnn.r0} bases  (ctx = {c['context']})")
        tracker.writer.add_scalar("architecture/cnn_receptive_field", model.encoder.cnn.r0, 0)
        tracker.writer.add_scalar("architecture/param_count", n_params, 0)

    # --- Optimizer + prepare ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c['max_lr']), weight_decay=c['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    # --- Schedule (NOT prepared — stepped manually). v40 fix preserved. ---
    total_steps = len(train_dl) * c['epochs']
    warmup_steps = int(total_steps * c['pct_start'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=c['pct_start'],
    )

    effective_bs = c['batch_size'] * accelerator.num_processes
    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(train_dl)}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps} (peak LR at step {warmup_steps})")
        print(f"Effective batch size: {effective_bs}")

    # --- Metrics ---
    device = accelerator.device
    metrics = {
        'f1': BinaryF1Score().to(device),
        'auroc': BinaryAUROC().to(device),
        'auprc': BinaryAveragePrecision().to(device),
        'accuracy': BinaryAccuracy().to(device),
    }

    # --- CSV (main process only) ---
    csv_file = None
    csv_writer = None
    if accelerator.is_main_process:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)
        csv_file.flush()

    # --- Training ---
    global_step = 0
    train_start = time.perf_counter()

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

        epoch_walltime = time.perf_counter() - train_start
        lr_end = scheduler.get_last_lr()[0]

        accelerator.log({
            "eval_f1": eval_results['f1'],
            "eval_auroc": eval_results['auroc'],
            "eval_auprc": eval_results['auprc'],
            "eval_top1": eval_results['accuracy'],
            "walltime_s": epoch_walltime,
        }, step=global_step)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: loss={avg_epoch_loss:.4f}  "
                  f"top1={eval_results['accuracy']:.4f}  f1={eval_results['f1']:.4f}  "
                  f"auroc={eval_results['auroc']:.4f}  walltime={epoch_walltime:.0f}s")

            csv_writer.writerow([
                epoch + 1, global_step, f"{epoch_walltime:.1f}",
                f"{avg_epoch_loss:.6f}", f"{lr_end:.6e}",
                f"{eval_results['accuracy']:.6f}", f"{eval_results['f1']:.6f}",
                f"{eval_results['auroc']:.6f}", f"{eval_results['auprc']:.6f}",
                c['d_model'], c['n_layers'], c['n_head'],
                c['batch_size'], effective_bs, n_params,
            ])
            csv_file.flush()

        # --- Per-epoch checkpoint (inference artifact) ---
        epoch_metrics = {
            'train_loss': avg_epoch_loss,
            'eval_top1': eval_results['accuracy'],
            'eval_f1': eval_results['f1'],
            'eval_auroc': eval_results['auroc'],
            'eval_auprc': eval_results['auprc'],
        }
        save_checkpoint(accelerator, model, config, epoch + 1, epoch_metrics, norm_fn, checkpoint_dir)

    if accelerator.is_main_process and csv_file is not None:
        csv_file.close()

    accelerator.end_training()


if __name__ == "__main__":
    main()
