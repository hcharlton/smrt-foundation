"""
Data scaling with OB007 pretrained encoder (exp 29) and gradual unfreezing,
using epoch-capped step budgets.

Two-stage setup with proportional scaling:
  Stage 1: frozen_frac of total_steps — frozen encoder, head-only at frozen_lr
  Stage 2: remainder — gradual encoder warmup (encoder_start_lr -> encoder_lr
           over encoder_warmup_frac of stage 2), head at head_lr

All fractions are relative to each size's total_steps, so the schedule structure
is preserved regardless of dataset size. Replaces exp 38's fixed 100k/300k split.
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
    'weight_decay': 0.02, 'pct_start': 0.1,
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
    assert 'pretrained_checkpoint' in config, "Missing pretrained_checkpoint path"
    assert 'finetune' in config, "Missing finetune config section"
    c = DEFAULT | config.get('classifier', {})
    config['classifier'] = c
    config['git_hash'] = get_git_revision_hash()
    return config, c


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


def make_cosine_warmup_lambda(total_steps, warmup_steps, min_lr_ratio=0.0):
    """Linear warmup from min_lr_ratio to 1.0, then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


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
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    size_dir = os.path.join(experiment_dir, f'n{train_size}')
    os.makedirs(size_dir, exist_ok=True)

    tag = f"[GPU {rank} n={train_size:,}]"
    print(f"{tag} Starting")

    set_seed(42)

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

    total_steps, steps_per_epoch, eval_steps = compute_step_budget(
        n_train, c['batch_size'], config['scaling'],
    )

    # Proportional two-stage split.
    ft = config['finetune']
    frozen_steps = int(ft['frozen_frac'] * total_steps)
    stage2_steps = total_steps - frozen_steps
    encoder_warmup = int(ft['encoder_warmup_frac'] * stage2_steps)

    print(f"{tag} Train: {n_train}, Val: {len(val_x)}, "
          f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps:,}, "
          f"Epochs: {total_steps / max(steps_per_epoch, 1):.1f}")
    print(f"{tag} Stage 1: {frozen_steps:,} steps (frozen), "
          f"Stage 2: {stage2_steps:,} steps (encoder warmup: {encoder_warmup:,})")

    # Model with pretrained encoder.
    model = DirectClassifier(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context'],
    )
    load_pretrained_encoder(config['pretrained_checkpoint'], model)
    model = model.to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, "
          f"CNN receptive field: {model.encoder.cnn.r0}")

    criterion = nn.BCEWithLogitsLoss()

    # Stage 1: freeze encoder, head-only optimizer.
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

    metrics = build_metrics(device)

    writer = SummaryWriter(log_dir=os.path.join(tb_dir, f'n{train_size}'))
    writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
    writer.add_scalar("train_size", train_size, 0)

    csv_path = os.path.join(size_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'train_size', 'eval_point', 'step', 'stage',
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

    effective_bs = min(c['batch_size'], n_train)
    print(f"{tag} Effective batch size: {effective_bs} (config: {c['batch_size']}, n_train: {n_train})")
    print(f"{tag} Eval schedule ({n_evals} points): {eval_steps}")

    for step in range(1, total_steps + 1):
        # Stage transition.
        if step == frozen_steps + 1:
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
        idx = torch.randint(0, n_train, (effective_bs,), device=device)
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
            if stage == 2:
                writer.add_scalar("train/encoder_lr", optimizer.param_groups[0]['lr'], step)

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
                train_size, eval_point, step, stage,
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
    print(f"Pretrained checkpoint: {config['pretrained_checkpoint']}")
    print(f"Train sizes: {train_sizes}")
    print(f"Step budget: max_epochs={s['max_epochs']}, "
          f"min_steps={s['min_steps']:,}, max_steps={s['max_steps']:,}")
    ft = config['finetune']
    print(f"Finetune: frozen_frac={ft['frozen_frac']}, "
          f"encoder {ft['encoder_start_lr']} -> {ft['encoder_lr']}, "
          f"warmup_frac={ft['encoder_warmup_frac']}")
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
