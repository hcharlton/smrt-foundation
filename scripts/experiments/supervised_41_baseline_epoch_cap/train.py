"""
Data scaling with epoch-capped step budget: train DirectClassifier from scratch
at 16 training set sizes, per-size step budget, fixed 1M validation set.

Replaces exp 33/39's fixed 400k steps with:
  total_steps = min(max(max_epochs * steps_per_epoch, min_steps), max_steps)
This prevents catastrophic overtraining at small sizes (exp 33 trained n=100
for 400k epochs) while keeping large sizes unchanged. The LR schedule scales
proportionally so pct_start=0.1 means 10% warmup regardless of dataset size.
"""

import sys
import os
import math
import subprocess
import yaml
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in REQUIRED_DATA_KEYS:
        assert key in config, f"Missing required config key: {key}"
    c = DEFAULT | config.get('classifier', {})
    config['classifier'] = c
    config['git_hash'] = get_git_revision_hash()
    return config, c


def preload_to_gpu(dataset, device, batch_size=8192, num_workers=4):
    """Load entire dataset into GPU tensors for zero-overhead batching."""
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def compute_step_budget(train_size, batch_size, scaling_config):
    """Compute per-size step budget and eval schedule."""
    steps_per_epoch = math.ceil(train_size / batch_size)
    total_steps = min(
        max(scaling_config['max_epochs'] * steps_per_epoch, scaling_config['min_steps']),
        scaling_config['max_steps'],
    )
    first_eval = min(scaling_config['first_eval_step'], max(1, total_steps // scaling_config['n_evals']))
    eval_steps = sorted(set(
        np.geomspace(first_eval, total_steps, scaling_config['n_evals']).astype(int).tolist()
    ))
    return total_steps, steps_per_epoch, eval_steps


def build_metrics(device):
    return {
        'f1': BinaryF1Score().to(device),
        'auroc': BinaryAUROC().to(device),
        'auprc': BinaryAveragePrecision().to(device),
        'accuracy': BinaryAccuracy().to(device),
    }


def evaluate(model, val_x, val_y, batch_size, metrics, criterion):
    """Run eval, return metrics dict + avg val loss."""
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    for i in range(0, len(val_x), batch_size):
        vx = val_x[i:i+batch_size]
        vy = val_y[i:i+batch_size]
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vlogits = model(vx)
            vloss = criterion(vlogits, vy.unsqueeze(1))
        val_loss_sum += vloss.item()
        val_batches += 1
        vy_hat = (vlogits > 0).squeeze(-1)
        vy_int = vy.long()
        metrics['f1'].update(vy_hat, vy_int)
        metrics['auroc'].update(vlogits.squeeze(-1).float(), vy_int)
        metrics['auprc'].update(vlogits.squeeze(-1).float(), vy_int)
        metrics['accuracy'].update(vy_hat, vy_int)

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    results = {name: m.compute().item() for name, m in metrics.items()}
    for m in metrics.values():
        m.reset()
    return results, avg_val_loss


def train_one_size(rank, train_size, config, c, experiment_dir, tb_dir):
    """Train a single training size on GPU `rank`."""
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    size_dir = os.path.join(experiment_dir, f'n{train_size}')
    os.makedirs(size_dir, exist_ok=True)

    tag = f"[GPU {rank} n={train_size:,}]"
    print(f"{tag} Starting")

    set_seed(42)

    # Normalization.
    norm_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'], limit=train_size
    )
    norm_fn = KineticsNorm(norm_ds, log_transform=True)
    del norm_ds
    print(f"{tag} KineticsNorm means: {norm_fn.means}, stds: {norm_fn.stds}")

    # Datasets — created then preloaded to GPU tensors.
    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=train_size, norm_fn=norm_fn, balance=True,
    )
    val_ds = LabeledMemmapDataset(
        config['pos_data_val'], config['neg_data_val'],
        limit=config['scaling']['val_limit'], norm_fn=norm_fn,
    )

    print(f"{tag} Preloading training data ({len(train_ds)} samples) to GPU...")
    train_x, train_y = preload_to_gpu(train_ds, device)
    n_train = len(train_x)
    print(f"{tag} Preloading validation data ({len(val_ds)} samples) to GPU...")
    val_x, val_y = preload_to_gpu(val_ds, device)
    print(f"{tag} Preloaded — train: {train_x.shape}, val: {val_x.shape}")
    del train_ds, val_ds

    # Per-size step budget.
    total_steps, steps_per_epoch, eval_steps = compute_step_budget(
        n_train, c['batch_size'], config['scaling'],
    )
    warmup_steps = int(total_steps * c['pct_start'])

    print(f"{tag} Train: {n_train}, Val: {len(val_x)}, "
          f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps:,}, "
          f"Epochs: {total_steps / max(steps_per_epoch, 1):.1f}, "
          f"Warmup: {warmup_steps:,} steps")

    # Model.
    model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context'],
    ).to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, "
          f"CNN receptive field: {model.encoder.cnn.r0}")

    # Optimizer + schedule — created directly, no Accelerate wrapping.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(c['max_lr']), weight_decay=c['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=c['pct_start'],
    )

    metrics = build_metrics(device)

    # TensorBoard.
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, f'n{train_size}'))
    writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
    writer.add_scalar("train_size", train_size, 0)

    # CSV.
    csv_path = os.path.join(size_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'train_size', 'eval_point', 'step',
        'train_loss', 'val_loss',
        'val_f1', 'val_auroc', 'val_auprc', 'val_accuracy',
        'epochs_completed',
    ])
    csv_file.flush()

    # --- Training loop ---
    eval_steps_set = set(eval_steps)
    n_evals = len(eval_steps)
    interval_loss = 0.0
    steps_since_last_eval = 0
    eval_point = 0

    print(f"{tag} Eval schedule ({n_evals} points): {eval_steps}")

    for step in range(1, total_steps + 1):
        model.train()
        idx = torch.randint(0, n_train, (c['batch_size'],), device=device)
        x = train_x[idx]
        y = train_y[idx]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        interval_loss += loss.item()
        steps_since_last_eval += 1

        if step % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        if step in eval_steps_set:
            eval_point += 1
            avg_train_loss = interval_loss / steps_since_last_eval
            interval_loss = 0.0
            steps_since_last_eval = 0

            writer.add_scalar("train/interval_avg_loss", avg_train_loss, step)

            eval_results, avg_val_loss = evaluate(
                model, val_x, val_y, c['batch_size'], metrics, criterion,
            )

            writer.add_scalar("eval/loss", avg_val_loss, step)
            for name, val in eval_results.items():
                writer.add_scalar(f"eval/{name}", val, step)

            epochs_completed = step / max(steps_per_epoch, 1)

            csv_writer.writerow([
                train_size, eval_point, step,
                f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}',
                f'{eval_results["f1"]:.6f}', f'{eval_results["auroc"]:.6f}',
                f'{eval_results["auprc"]:.6f}', f'{eval_results["accuracy"]:.6f}',
                f'{epochs_completed:.1f}',
            ])
            csv_file.flush()

            print(f"{tag} [{eval_point}/{n_evals}] step={step:,} "
                  f"({epochs_completed:.0f} ep) "
                  f"loss={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                  f"acc={eval_results['accuracy']:.4f} f1={eval_results['f1']:.4f} "
                  f"auroc={eval_results['auroc']:.4f}")

            ckpt_path = os.path.join(size_dir, f'step{step}.pt')
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': model.encoder.state_dict(),
                    'config': config,
                    'train_size': train_size,
                    'step': step,
                    'eval_point': eval_point,
                    'metrics': {'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                                **{f'eval_{k}': v for k, v in eval_results.items()}},
                    **norm_fn.save_stats(),
                }, ckpt_path)
            except Exception as e:
                print(f"{tag} ERROR saving checkpoint: {type(e).__name__}: {e}")
                raise

    csv_file.close()
    print(f"{tag} Done. {eval_point} checkpoints saved.")
    writer.close()


def worker_fn(rank, world_size, train_sizes, config, c, experiment_dir, tb_dir):
    for train_size in train_sizes[rank::world_size]:
        train_one_size(rank, train_size, config, c, experiment_dir, tb_dir)
        torch.cuda.empty_cache()


def merge_csvs(experiment_dir, train_sizes):
    merged_path = os.path.join(experiment_dir, 'results.csv')
    rows = []
    header = None
    for train_size in train_sizes:
        per_size_path = os.path.join(experiment_dir, f'n{train_size}', 'results.csv')
        if not os.path.exists(per_size_path):
            print(f"WARNING: missing {per_size_path}")
            continue
        with open(per_size_path) as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            rows.extend(list(reader))
    rows.sort(key=lambda r: (int(r[0]), int(r[1])))
    with open(merged_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)
    print(f"Merged results into {merged_path} ({len(rows)} rows)")


def main():
    config, c = load_config(sys.argv[1])

    s = config['scaling']
    train_sizes = s['train_sizes']

    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    tb_dir = os.path.join(experiment_dir, 'training_logs')
    os.makedirs(tb_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    world_size = min(num_gpus, len(train_sizes))

    print(f"Config: {config}")
    print(f"Train sizes: {train_sizes}")
    print(f"Step budget: max_epochs={s['max_epochs']}, "
          f"min_steps={s['min_steps']:,}, max_steps={s['max_steps']:,}")
    print(f"GPUs available: {num_gpus}, using {world_size} workers")

    if world_size > 1:
        mp.spawn(
            worker_fn,
            args=(world_size, train_sizes, config, c, experiment_dir, tb_dir),
            nprocs=world_size,
        )
    else:
        worker_fn(0, 1, train_sizes, config, c, experiment_dir, tb_dir)

    merge_csvs(experiment_dir, train_sizes)
    print(f"TensorBoard logs in {tb_dir}")


if __name__ == "__main__":
    main()
