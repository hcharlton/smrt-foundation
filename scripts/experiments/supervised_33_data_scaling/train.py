"""
Data scaling experiment: train DirectClassifier from scratch at 10 training
set sizes (100 to 128k), fixed step budget per size, fixed 1M validation set.

Parallelised across GPUs: each training size runs independently on its own
GPU via torch.multiprocessing.spawn. No inter-GPU communication — each worker
trains a complete model from scratch. With 8 GPUs and 10 sizes, GPUs 0-1
each handle 2 sizes sequentially (the two smallest finish fast), GPUs 2-7
handle 1 size each.

Based on supervised_31_baseline_clean. Same model architecture
(DirectClassifier d_model=128 n_layers=4 n_head=4 ctx=32), same optimizer
(AdamW lr=3e-3 wd=0.02), same cosine schedule with warmup (pct_start=0.1),
same bf16 mixed precision, same KineticsNorm, same evaluation metrics.

The only controlled variable is training set size. Each size gets the same
number of optimizer steps (steps_per_size), so smaller datasets see many
more epochs — this is intentional to let the model overfit and reveal the
data scaling curve shape.
"""

import sys
import os
import subprocess
import yaml
import csv
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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


def cycling_iter(dataloader):
    """Infinite iterator that cycles through a DataLoader."""
    while True:
        yield from dataloader


def train_one_size(rank, train_size, config, c, steps_per_size, evals_per_size,
                   eval_every, experiment_dir, checkpoint_dir, tb_dir):
    """Train a single training size on GPU `rank`."""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    tag = f"[GPU {rank} n={train_size:,}]"
    print(f"{tag} Starting")

    set_seed(42)

    # KineticsNorm from this training subset.
    norm_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=train_size
    )
    norm_fn = KineticsNorm(norm_ds, log_transform=True)
    del norm_ds
    print(f"{tag} KineticsNorm means: {norm_fn.means}, stds: {norm_fn.stds}")

    # Datasets.
    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=train_size, norm_fn=norm_fn, balance=True
    )
    train_dl = DataLoader(
        train_ds, batch_size=c['batch_size'], num_workers=2,
        pin_memory=True, prefetch_factor=4, shuffle=True, drop_last=False,
    )
    val_ds = LabeledMemmapDataset(
        config['pos_data_val'], config['neg_data_val'],
        limit=config['scaling']['val_limit'], norm_fn=norm_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=c['batch_size'], num_workers=2,
        pin_memory=True, prefetch_factor=4, shuffle=False,
    )

    steps_per_epoch = len(train_dl)
    print(f"{tag} Train: {len(train_ds)}, Val: {len(val_ds)}, "
          f"Steps/epoch: {steps_per_epoch}, "
          f"Total epochs: {steps_per_size / max(steps_per_epoch, 1):.1f}")

    # Model.
    model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, "
          f"CNN receptive field: {model.encoder.cnn.r0}")

    # Optimizer + schedule.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(c['max_lr']), weight_decay=c['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=steps_per_size, pct_start=c['pct_start']
    )

    # Metrics.
    f1_metric = BinaryF1Score().to(device)
    auroc_metric = BinaryAUROC().to(device)
    auprc_metric = BinaryAveragePrecision().to(device)
    acc_metric = BinaryAccuracy().to(device)

    # TensorBoard.
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, f'n{train_size}'))
    writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
    writer.add_scalar("train_size", train_size, 0)

    # Per-size CSV (no locking needed — one file per worker).
    csv_path = os.path.join(experiment_dir, f'results_n{train_size}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'train_size', 'eval_point', 'step',
        'train_loss', 'val_loss',
        'val_f1', 'val_auroc', 'val_auprc', 'val_accuracy',
        'epochs_completed',
    ])
    csv_file.flush()

    # --- Step-based training with periodic eval ---
    train_iter = cycling_iter(train_dl)
    interval_loss = 0.0
    avg_train_loss = avg_val_loss = 0.0
    eval_f1 = eval_auroc = eval_auprc = eval_acc = 0.0

    for step in tqdm(range(1, steps_per_size + 1), desc=f"n={train_size}",
                     position=rank, leave=True):
        model.train()
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        interval_loss += loss.item()

        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        # --- Periodic eval ---
        if step % eval_every == 0:
            eval_point = step // eval_every
            avg_train_loss = interval_loss / eval_every
            interval_loss = 0.0

            writer.add_scalar("train/interval_avg_loss", avg_train_loss, step)

            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            for vx, vy in val_dl:
                vx, vy = vx.to(device), vy.to(device)
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    vlogits = model(vx)
                    vloss = criterion(vlogits, vy.unsqueeze(1))
                val_loss_sum += vloss.item()
                val_batches += 1
                vy_hat = (vlogits > 0).squeeze(-1)
                vy_int = vy.long()
                f1_metric.update(vy_hat, vy_int)
                auroc_metric.update(vlogits.squeeze(-1).float(), vy_int)
                auprc_metric.update(vlogits.squeeze(-1).float(), vy_int)
                acc_metric.update(vy_hat, vy_int)

            avg_val_loss = val_loss_sum / max(val_batches, 1)
            eval_f1 = f1_metric.compute().item()
            eval_auroc = auroc_metric.compute().item()
            eval_auprc = auprc_metric.compute().item()
            eval_acc = acc_metric.compute().item()

            f1_metric.reset()
            auroc_metric.reset()
            auprc_metric.reset()
            acc_metric.reset()

            # TensorBoard
            writer.add_scalar("eval/loss", avg_val_loss, step)
            writer.add_scalar("eval/f1", eval_f1, step)
            writer.add_scalar("eval/auroc", eval_auroc, step)
            writer.add_scalar("eval/auprc", eval_auprc, step)
            writer.add_scalar("eval/accuracy", eval_acc, step)

            epochs_completed = step / max(steps_per_epoch, 1)

            # CSV
            csv_writer.writerow([
                train_size, eval_point, step,
                f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}',
                f'{eval_f1:.6f}', f'{eval_auroc:.6f}',
                f'{eval_auprc:.6f}', f'{eval_acc:.6f}',
                f'{epochs_completed:.1f}',
            ])
            csv_file.flush()

            print(f"{tag} [{eval_point}/{evals_per_size}] step={step:,} "
                  f"({epochs_completed:.0f} ep) "
                  f"loss={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                  f"acc={eval_acc:.4f} f1={eval_f1:.4f} auroc={eval_auroc:.4f}")

    csv_file.close()

    # Checkpoint.
    save_path = os.path.join(checkpoint_dir, f'n{train_size}_final.pt')
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': model.encoder.state_dict(),
            'config': config,
            'train_size': train_size,
            'step': steps_per_size,
            'metrics': {'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                        'eval_f1': eval_f1, 'eval_auroc': eval_auroc,
                        'eval_auprc': eval_auprc, 'eval_accuracy': eval_acc},
            **norm_fn.save_stats(),
        }, save_path)
        print(f"{tag} Saved checkpoint to {save_path}")
    except Exception as e:
        print(f"{tag} ERROR saving checkpoint: {type(e).__name__}: {e}")
        raise

    writer.close()


def worker_fn(rank, world_size, train_sizes, config, c, steps_per_size,
              evals_per_size, eval_every, experiment_dir, checkpoint_dir, tb_dir):
    """Worker entry point: handles all training sizes assigned to this rank."""
    for train_size in train_sizes[rank::world_size]:
        train_one_size(rank, train_size, config, c, steps_per_size, evals_per_size,
                       eval_every, experiment_dir, checkpoint_dir, tb_dir)
        torch.cuda.empty_cache()


def merge_csvs(experiment_dir, train_sizes):
    """Merge per-size CSV files into a single results.csv."""
    merged_path = os.path.join(experiment_dir, 'results.csv')
    rows = []
    header = None

    for train_size in train_sizes:
        per_size_path = os.path.join(experiment_dir, f'results_n{train_size}.csv')
        if not os.path.exists(per_size_path):
            print(f"WARNING: missing {per_size_path}")
            continue
        with open(per_size_path) as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            rows.extend(list(reader))
        os.remove(per_size_path)

    rows.sort(key=lambda r: (int(r[0]), int(r[1])))

    with open(merged_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)

    print(f"Merged results into {merged_path} ({len(rows)} rows)")


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
    steps_per_size = s['steps_per_size']
    evals_per_size = s['evals_per_size']
    eval_every = steps_per_size // evals_per_size

    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    tb_dir = os.path.join(experiment_dir, 'training_logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    world_size = min(num_gpus, len(train_sizes))

    print(f"Config: {config}")
    print(f"Train sizes: {train_sizes}")
    print(f"Steps per size: {steps_per_size:,}")
    print(f"Eval every {eval_every:,} steps ({evals_per_size} evals per size)")
    print(f"GPUs available: {num_gpus}, using {world_size} workers")

    if world_size > 1:
        mp.spawn(
            worker_fn,
            args=(world_size, train_sizes, config, c, steps_per_size,
                  evals_per_size, eval_every, experiment_dir, checkpoint_dir, tb_dir),
            nprocs=world_size,
        )
    else:
        # Fallback: sequential on single GPU.
        worker_fn(0, 1, train_sizes, config, c, steps_per_size,
                  evals_per_size, eval_every, experiment_dir, checkpoint_dir, tb_dir)

    merge_csvs(experiment_dir, train_sizes)
    print(f"TensorBoard logs in {tb_dir}")
    print(f"Checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()
