"""
Data scaling experiment: train DirectClassifier from scratch at 10 training
set sizes (100 to 128k), 10 epochs each, fixed 1M validation set.

Based on supervised_31_baseline_clean. Same model architecture
(DirectClassifier d_model=128 n_layers=4 n_head=4 ctx=32), same optimizer
(AdamW lr=3e-3 wd=0.02), same cosine schedule with warmup (pct_start=0.1),
same bf16 mixed precision, same KineticsNorm, same evaluation metrics.

The only controlled variable is training set size. For each size:
  1. Fresh model initialisation (same seed)
  2. Fresh KineticsNorm computed from that training subset
  3. Fresh optimizer + cosine schedule (total_steps based on this size)
  4. 10 training epochs
  5. Evaluation after each epoch on a fixed 1M-sample validation set

Results are logged to:
  - TensorBoard: one run directory per training set size
  - CSV: one file with all (train_size, epoch) rows

Single GPU only (1 process). Multi-GPU is wasteful for small datasets and
causes issues at sizes < batch_size.
"""

import sys
import os
import subprocess
import yaml
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy
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
    'batch_size': 512,
    'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
}


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def save_checkpoint(model, config, train_size, epoch, metrics, norm_fn, checkpoint_dir):
    """Save checkpoint after final epoch of a training size."""
    save_path = os.path.join(checkpoint_dir, f'n{train_size}_final.pt')
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': model.encoder.state_dict(),
            'config': config,
            'train_size': train_size,
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

    s = config['scaling']
    train_sizes = s['train_sizes']
    val_limit = s['val_limit']
    epochs_per_size = s['epochs_per_size']

    accelerator = Accelerator(mixed_precision='bf16')

    # --- Experiment directories ---
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    tb_dir = os.path.join(experiment_dir, 'training_logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # --- CSV setup (flush after every row for crash safety) ---
    csv_path = os.path.join(experiment_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'train_size', 'epoch',
        'train_loss', 'val_loss',
        'val_f1', 'val_auroc', 'val_auprc', 'val_accuracy',
        'steps_per_epoch', 'total_train_samples_seen',
    ])
    csv_file.flush()

    print(f"Config: {config}")
    print(f"Train sizes: {train_sizes}")
    print(f"Validation limit: {val_limit:,}")
    print(f"Epochs per size: {epochs_per_size}")

    # --- Main loop: iterate over training set sizes ---
    for size_idx, train_size in enumerate(train_sizes):
        print(f"\n{'='*60}")
        print(f"Training size {size_idx+1}/{len(train_sizes)}: n={train_size:,}")
        print(f"{'='*60}")

        # Deterministic init: same model weights for every size.
        set_seed(42)

        # Compute KineticsNorm from this training subset.
        norm_ds = LabeledMemmapDataset(
            config['pos_data_train'], config['neg_data_train'],
            limit=train_size
        )
        norm_fn = KineticsNorm(norm_ds, log_transform=True)
        del norm_ds
        print(f"  KineticsNorm means: {norm_fn.means}, stds: {norm_fn.stds}")

        # Training dataset + dataloader.
        train_ds = LabeledMemmapDataset(
            config['pos_data_train'], config['neg_data_train'],
            limit=train_size, norm_fn=norm_fn, balance=True
        )
        train_dl = DataLoader(
            train_ds, batch_size=c['batch_size'], num_workers=2,
            pin_memory=True, prefetch_factor=4, shuffle=True,
            drop_last=False,
        )

        # Validation dataset + dataloader (fixed 1M, per-size norm).
        val_ds = LabeledMemmapDataset(
            config['pos_data_val'], config['neg_data_val'],
            limit=val_limit, norm_fn=norm_fn
        )
        val_dl = DataLoader(
            val_ds, batch_size=c['batch_size'], num_workers=2,
            pin_memory=True, prefetch_factor=4, shuffle=False,
        )

        print(f"  Training samples: {len(train_ds)}")
        print(f"  Validation samples: {len(val_ds)}")
        print(f"  Steps per epoch: {len(train_dl)}")

        # Fresh model.
        model = DirectClassifier(
            d_model=c['d_model'], n_layers=c['n_layers'],
            n_head=c['n_head'], max_len=c['context'],
        )
        if size_idx == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model parameters: {n_params:,}")
            print(f"  CNN receptive field: {model.encoder.cnn.r0} bases  (ctx = {c['context']})")

        # Fresh optimizer + schedule.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(c['max_lr']),
            weight_decay=c['weight_decay'],
        )
        criterion = nn.BCEWithLogitsLoss()

        model, optimizer, train_dl, val_dl = accelerator.prepare(
            model, optimizer, train_dl, val_dl
        )

        total_steps = len(train_dl) * epochs_per_size
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, total_steps=total_steps, pct_start=c['pct_start']
        )

        print(f"  Total steps: {total_steps}")

        # Metrics (fresh per size).
        f1_metric = BinaryF1Score().to(accelerator.device)
        auroc_metric = BinaryAUROC().to(accelerator.device)
        auprc_metric = BinaryAveragePrecision().to(accelerator.device)
        acc_metric = BinaryAccuracy().to(accelerator.device)

        # TensorBoard writer for this training size.
        writer = SummaryWriter(log_dir=os.path.join(tb_dir, f'n{train_size}'))
        writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
        writer.add_scalar("train_size", train_size, 0)

        global_step = 0
        avg_train_loss = 0.0
        avg_val_loss = 0.0
        epoch_f1 = epoch_auroc = epoch_auprc = epoch_acc = 0.0

        # Training loop.
        for epoch in range(epochs_per_size):
            model.train()
            epoch_loss = 0.0
            progress = tqdm(
                train_dl,
                desc=f"n={train_size} epoch {epoch+1}/{epochs_per_size}",
            )

            for x, y in progress:
                logits = model(x)
                loss = criterion(logits, y.unsqueeze(1).to(torch.float32))

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                global_step += 1
                loss_val = loss.item()
                epoch_loss += loss_val

                writer.add_scalar("train/loss", loss_val, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                progress.set_postfix(loss=f"{loss_val:.4f}")

            avg_train_loss = epoch_loss / max(len(train_dl), 1)
            writer.add_scalar("train/epoch_avg_loss", avg_train_loss, global_step)

            # --- Eval ---
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            for x, y in tqdm(val_dl, desc=f"n={train_size} eval {epoch+1}"):
                with torch.no_grad():
                    logits = model(x)
                    loss = criterion(logits, y.unsqueeze(1).to(torch.float32))
                val_loss_sum += loss.item()
                val_steps += 1
                y_hat = (logits > 0).squeeze(-1)
                y_int = y.long()
                f1_metric.update(y_hat, y_int)
                auroc_metric.update(logits.squeeze(-1), y_int)
                auprc_metric.update(logits.squeeze(-1), y_int)
                acc_metric.update(y_hat, y_int)

            avg_val_loss = val_loss_sum / max(val_steps, 1)
            epoch_f1 = f1_metric.compute().item()
            epoch_auroc = auroc_metric.compute().item()
            epoch_auprc = auprc_metric.compute().item()
            epoch_acc = acc_metric.compute().item()

            f1_metric.reset()
            auroc_metric.reset()
            auprc_metric.reset()
            acc_metric.reset()

            # TensorBoard
            writer.add_scalar("eval/loss", avg_val_loss, global_step)
            writer.add_scalar("eval/f1", epoch_f1, global_step)
            writer.add_scalar("eval/auroc", epoch_auroc, global_step)
            writer.add_scalar("eval/auprc", epoch_auprc, global_step)
            writer.add_scalar("eval/accuracy", epoch_acc, global_step)

            # CSV (flush for crash safety)
            csv_writer.writerow([
                train_size, epoch + 1,
                f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}',
                f'{epoch_f1:.6f}', f'{epoch_auroc:.6f}',
                f'{epoch_auprc:.6f}', f'{epoch_acc:.6f}',
                len(train_dl),
                (epoch + 1) * len(train_ds),
            ])
            csv_file.flush()

            print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f} "
                  f"val_loss={avg_val_loss:.4f} acc={epoch_acc:.4f} "
                  f"f1={epoch_f1:.4f} auroc={epoch_auroc:.4f}")

        # Checkpoint after all epochs for this size.
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(
            unwrapped, config, train_size, epochs_per_size,
            {'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
             'eval_f1': epoch_f1, 'eval_auroc': epoch_auroc,
             'eval_auprc': epoch_auprc, 'eval_accuracy': epoch_acc},
            norm_fn, checkpoint_dir,
        )

        writer.close()

        # Free memory before next size.
        del model, optimizer, scheduler, train_ds, train_dl, val_ds, val_dl
        del norm_fn, f1_metric, auroc_metric, auprc_metric, acc_metric
        torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nResults saved to {csv_path}")
    print(f"TensorBoard logs in {tb_dir}")
    print(f"Checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
