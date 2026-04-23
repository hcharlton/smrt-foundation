"""
Unified dataset-scaling grid trainer.

Runs one experimental condition (encoder init + training strategy + batch size
policy) across a grid of training dataset sizes, distributing work across all
available GPUs via a shared atomic counter for automatic load balancing.

Behaviour is controlled entirely by the experiment's config.yaml:

  pretrained_checkpoint   present -> load pretrained encoder
  finetune section        present -> two-stage (frozen -> gradual unfreeze)
  classifier.bs_floor/k   present -> per-size batch scaling

Usage (via thin per-experiment train.py wrapper):
  bash run.sh scripts/experiments/supervised_NN_name/
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

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


REQUIRED_DATA_KEYS = ['pos_data_train', 'neg_data_train', 'pos_data_val', 'neg_data_val']

DEFAULT_CLASSIFIER = {
    'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 32,
    'weight_decay': 0.02, 'pct_start': 0.1,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    assert 'scaling' in config, "Missing scaling config section"
    if 'finetune' not in config:
        assert 'max_lr' in config.get('classifier', {}), \
            "Single-stage experiments require classifier.max_lr"
    c = DEFAULT_CLASSIFIER | config.get('classifier', {})
    config['classifier'] = c
    config['git_hash'] = get_git_revision_hash()
    return config, c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_step_budget(train_size, batch_size, scaling_config):
    steps_per_epoch = math.ceil(train_size / batch_size)
    total_steps = min(
        max(scaling_config['max_epochs'] * steps_per_epoch, scaling_config['min_steps']),
        scaling_config['max_steps'],
    )
    first_eval = min(
        scaling_config['first_eval_step'],
        max(1, total_steps // scaling_config['n_evals']),
    )
    eval_steps = sorted(set(
        np.geomspace(first_eval, total_steps, scaling_config['n_evals']).astype(int).tolist()
    ))
    return total_steps, steps_per_epoch, eval_steps


def compute_batch_size(n_train, bs_cap, bs_floor, bs_k):
    bs = min(bs_cap, max(bs_floor, n_train // bs_k))
    return min(bs, n_train)


def make_cosine_warmup_lambda(total_steps, warmup_steps, min_lr_ratio=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def load_pretrained_encoder(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_weights = checkpoint['encoder_state_dict']
    encoder_weights = {k: v for k, v in encoder_weights.items()
                       if not (k == 'pe.pe' and v.shape != model.encoder.pe.pe.shape)}
    missing, unexpected = model.encoder.load_state_dict(encoder_weights, strict=False)
    print(f"Loaded pretrained encoder from {checkpoint_path}")
    if missing:
        print(f"  Missing keys (expected for PE): {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    return model


def preload_to_gpu(dataset, device, batch_size=8192, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def build_metrics(device):
    return {
        'f1': BinaryF1Score().to(device),
        'auroc': BinaryAUROC().to(device),
        'auprc': BinaryAveragePrecision().to(device),
        'accuracy': BinaryAccuracy().to(device),
    }


def evaluate(model, val_x, val_y, batch_size, metrics, criterion):
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


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_one_size(rank, train_size, config, experiment_dir, tb_dir):
    c = config['classifier']
    ft = config.get('finetune')
    has_pretrained = 'pretrained_checkpoint' in config
    has_finetune = ft is not None
    has_batch_scaling = 'bs_floor' in c and 'bs_k' in c

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    size_dir = os.path.join(experiment_dir, f'n{train_size}')
    os.makedirs(size_dir, exist_ok=True)

    tag = f"[GPU {rank} n={train_size:,}]"
    print(f"{tag} Starting")

    set_seed(42)

    # -- Data --
    norm_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'], limit=train_size
    )
    norm_fn = KineticsNorm(norm_ds, log_transform=True)
    del norm_ds
    print(f"{tag} KineticsNorm means: {norm_fn.means}, stds: {norm_fn.stds}")

    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=train_size, norm_fn=norm_fn, balance=True,
    )
    val_ds = LabeledMemmapDataset(
        config['pos_data_val'], config['neg_data_val'],
        limit=config['scaling']['val_limit'], norm_fn=norm_fn, balance=True,
    )

    print(f"{tag} Preloading training data ({len(train_ds)} samples) to GPU...")
    train_x, train_y = preload_to_gpu(train_ds, device)
    n_train = len(train_x)
    print(f"{tag} Preloading validation data ({len(val_ds)} samples) to GPU...")
    val_x, val_y = preload_to_gpu(val_ds, device)
    print(f"{tag} Preloaded — train: {train_x.shape}, val: {val_x.shape}")
    del train_ds, val_ds

    # -- Batch size --
    if has_batch_scaling:
        train_bs = compute_batch_size(n_train, c['batch_size'], c['bs_floor'], c['bs_k'])
    else:
        train_bs = min(c['batch_size'], n_train)

    # -- Step budget --
    total_steps, steps_per_epoch, eval_steps = compute_step_budget(
        n_train, train_bs, config['scaling'],
    )

    print(f"{tag} Train: {n_train}, Val: {len(val_x)}, "
          f"Train bs: {train_bs}, Steps/epoch: {steps_per_epoch}, "
          f"Total steps: {total_steps:,}, "
          f"Epochs: {total_steps / max(steps_per_epoch, 1):.1f}")

    # -- Model --
    model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context'],
    )
    if has_pretrained:
        load_pretrained_encoder(config['pretrained_checkpoint'], model)
    model = model.to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, "
          f"CNN receptive field: {model.encoder.cnn.r0}")

    criterion = nn.BCEWithLogitsLoss()
    metrics = build_metrics(device)

    # -- Optimizer + scheduler (depends on training strategy) --
    frozen_steps = -1
    stage2_steps = 0
    encoder_warmup = 0

    if has_finetune:
        frozen_steps = int(ft['frozen_frac'] * total_steps)
        stage2_steps = total_steps - frozen_steps
        encoder_warmup = int(ft['encoder_warmup_frac'] * stage2_steps)

        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=float(ft['frozen_lr']),
            weight_decay=c['weight_decay'],
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, total_steps=frozen_steps, pct_start=c['pct_start'],
        )
        stage = 1

        print(f"{tag} Two-stage: frozen {frozen_steps:,} steps, "
              f"then unfreeze {stage2_steps:,} steps "
              f"(encoder warmup: {encoder_warmup:,})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(c['max_lr']),
            weight_decay=c['weight_decay'],
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, total_steps=total_steps, pct_start=c['pct_start'],
        )
        stage = 1

        warmup_steps = int(total_steps * c['pct_start'])
        print(f"{tag} Single-stage: lr={c['max_lr']}, "
              f"warmup={warmup_steps:,} steps")

    # -- Logging --
    writer = SummaryWriter(log_dir=os.path.join(tb_dir, f'n{train_size}'))
    writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
    writer.add_scalar("train_size", train_size, 0)
    writer.add_scalar("train_bs", train_bs, 0)

    csv_path = os.path.join(size_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'train_size', 'train_bs', 'eval_point', 'step', 'stage',
        'train_loss', 'val_loss',
        'val_f1', 'val_auroc', 'val_auprc', 'val_accuracy',
        'epochs_completed',
    ])
    csv_file.flush()

    eval_steps_set = set(eval_steps)
    n_evals = len(eval_steps)
    interval_loss = 0.0
    steps_since_last_eval = 0
    eval_point = 0
    eval_bs = c['batch_size']

    print(f"{tag} Eval schedule ({n_evals} points): {eval_steps}")

    # -- Training loop --
    for step in range(1, total_steps + 1):

        # Stage transition (two-stage only)
        if has_finetune and step == frozen_steps + 1:
            stage = 2
            for p in model.encoder.parameters():
                p.requires_grad = True

            encoder_start_ratio = float(ft['encoder_start_lr']) / float(ft['encoder_lr'])
            head_warmup = int(stage2_steps * c['pct_start'])

            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': float(ft['encoder_lr'])},
                {'params': model.head.parameters(), 'lr': float(ft['head_lr'])},
            ], weight_decay=c['weight_decay'])

            encoder_lambda = make_cosine_warmup_lambda(
                stage2_steps, encoder_warmup, min_lr_ratio=encoder_start_ratio,
            )
            head_lambda = make_cosine_warmup_lambda(
                stage2_steps, head_warmup, min_lr_ratio=0.0,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[encoder_lambda, head_lambda],
            )

            print(f"{tag} Stage 2: unfrozen, encoder {ft['encoder_start_lr']} -> "
                  f"{ft['encoder_lr']} over {encoder_warmup:,} steps, "
                  f"head lr={ft['head_lr']}, {stage2_steps:,} steps total")

        model.train()
        idx = torch.randint(0, n_train, (train_bs,), device=device)
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
            if has_finetune and stage == 2:
                writer.add_scalar("train/encoder_lr", optimizer.param_groups[0]['lr'], step)

        if step in eval_steps_set:
            eval_point += 1
            avg_train_loss = interval_loss / steps_since_last_eval
            interval_loss = 0.0
            steps_since_last_eval = 0

            writer.add_scalar("train/interval_avg_loss", avg_train_loss, step)

            eval_results, avg_val_loss = evaluate(
                model, val_x, val_y, eval_bs, metrics, criterion,
            )

            writer.add_scalar("eval/loss", avg_val_loss, step)
            for name, val in eval_results.items():
                writer.add_scalar(f"eval/{name}", val, step)

            epochs_completed = step / max(steps_per_epoch, 1)

            csv_writer.writerow([
                train_size, train_bs, eval_point, step, stage,
                f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}',
                f'{eval_results["f1"]:.6f}', f'{eval_results["auroc"]:.6f}',
                f'{eval_results["auprc"]:.6f}', f'{eval_results["accuracy"]:.6f}',
                f'{epochs_completed:.1f}',
            ])
            csv_file.flush()

            print(f"{tag} [{eval_point}/{n_evals}] step={step:,} stage={stage} "
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
                    'train_bs': train_bs,
                    'step': step,
                    'stage': stage,
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def worker_fn(rank, sizes, counter, config, experiment_dir, tb_dir):
    while True:
        with counter.get_lock():
            idx = counter.value
            counter.value += 1
        if idx >= len(sizes):
            break
        train_one_size(rank, sizes[idx], config, experiment_dir, tb_dir)
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

    if header is None:
        print("WARNING: no per-size CSVs found, skipping merge")
        return

    ts_idx = header.index('train_size')
    ep_idx = header.index('eval_point')
    rows.sort(key=lambda r: (int(r[ts_idx]), int(r[ep_idx])))

    with open(merged_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Merged results into {merged_path} ({len(rows)} rows)")


def main(config_path):
    config, c = load_config(config_path)

    s = config['scaling']
    train_sizes = s['train_sizes']

    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    tb_dir = os.path.join(experiment_dir, 'training_logs')
    os.makedirs(tb_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    world_size = min(num_gpus, len(train_sizes))

    # Summary
    print(f"Config: {config}")
    if 'pretrained_checkpoint' in config:
        print(f"Pretrained checkpoint: {config['pretrained_checkpoint']}")
    else:
        print("Encoder: random init (no pretrained checkpoint)")
    print(f"Train sizes: {train_sizes}")
    print(f"Step budget: max_epochs={s['max_epochs']}, "
          f"min_steps={s['min_steps']:,}, max_steps={s['max_steps']:,}")

    if 'finetune' in config:
        ft = config['finetune']
        print(f"Finetune: frozen_frac={ft['frozen_frac']}, "
              f"encoder {ft['encoder_start_lr']} -> {ft['encoder_lr']}, "
              f"warmup_frac={ft['encoder_warmup_frac']}")
    else:
        print(f"Single-stage: max_lr={c['max_lr']}")

    if 'bs_floor' in c:
        print(f"Batch size: scaled (cap={c['batch_size']}, "
              f"floor={c['bs_floor']}, k={c['bs_k']})")
    else:
        print(f"Batch size: fixed {c['batch_size']}")

    print(f"GPUs available: {num_gpus}, using {world_size} workers")

    # Sort largest-first for load balancing
    sizes_sorted = sorted(train_sizes, reverse=True)
    counter = mp.Value('i', 0)

    if world_size > 1:
        mp.spawn(
            worker_fn,
            args=(sizes_sorted, counter, config, experiment_dir, tb_dir),
            nprocs=world_size,
        )
    else:
        worker_fn(0, sizes_sorted, counter, config, experiment_dir, tb_dir)

    merge_csvs(experiment_dir, train_sizes)
    print(f"TensorBoard logs in {tb_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
