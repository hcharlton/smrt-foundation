"""
Matrixed dataset-scaling grid trainer (v2).

Drives a full (architecture x encoder-init x train_size) sweep from a single
config, instead of v1's one-config-per-(arch, init) pattern. All combos land
in one experiment directory with `arch_name` / `init_name` columns in the
merged CSV; the existing per-(arch, init) v1 (`scripts/ds_grid.py`) is left
untouched for prior experiments.

Behaviour is controlled entirely by the experiment's config.yaml. The new
sections are:

  architectures: { <name>: { d_model, n_layers, n_head, [classifier overrides] }, ... }
  inits:         { <name>: { checkpoint | checkpoint_template }, ... }
  skip:          [ { arch: <name>, init: <name> }, ... ]
  train_sizes:   [N1, N2, N3, ...]

`size_overrides` from v1 is intentionally not supported in v2 (per-arch
overrides on the `architectures` entries cover the same need without
introducing an arch x size override matrix).

Usage (via thin per-experiment train.py wrapper):
  bash run.sh scripts/experiments/supervised_NN_name/
"""

import sys
import os
import csv
import yaml
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier, DirectClassifierSmallRF
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm

# Reuse v1 helpers verbatim. They are stable, well-tested, and keeping the
# dependency direction (v2 imports v1) means a single bug fix propagates.
from scripts.ds_grid import (
    compute_step_budget,
    compute_batch_size,
    load_pretrained_encoder,
    preload_to_gpu,
    build_metrics,
    evaluate,
    get_git_revision_hash,
)


REQUIRED_DATA_KEYS = ['pos_data_train', 'neg_data_train', 'pos_data_val', 'neg_data_val']
REQUIRED_TOPLEVEL = ['architectures', 'inits', 'train_sizes', 'classifier', 'scaling']

DEFAULT_CLASSIFIER = {
    'context': 32,
    'weight_decay': 0.02,
    'pct_start': 0.1,
    # cnn_variant='default' (RF=107, default DirectClassifier) | 'small_rf'
    # (RF=27, DirectClassifierSmallRF for fine-tuning small-RF SSL encoders
    # like ssl_30 and ssl_58). Defaults to 'default' so existing configs
    # without the key (supervised_47/50) keep their behavior unchanged.
    'cnn_variant': 'default',
}


# ---------------------------------------------------------------------------
# Combo (one work item: arch x init x train_size)
# ---------------------------------------------------------------------------

@dataclass
class Combo:
    arch_name: str
    init_name: str
    train_size: int
    arch_cfg: dict           # merged classifier defaults + per-arch overrides
    checkpoint: Optional[str]  # absolute path or None for random init


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in REQUIRED_DATA_KEYS:
        assert key in config, f"Missing required data key: {key}"
    for key in REQUIRED_TOPLEVEL:
        assert key in config, f"Missing required top-level key: {key}"
    assert 'max_lr' in config['classifier'], (
        "v2 is single-stage only; classifier.max_lr is required"
    )
    if 'finetune' in config:
        raise ValueError(
            "v2 does not support `finetune:` (two-stage gradual unfreeze). "
            "Use the single-stage AdamW pattern from supervised_46/47/48 "
            "or the v1 trainer (scripts/ds_grid.py) for two-stage runs."
        )
    if 'size_overrides' in config:
        raise ValueError(
            "v2 does not support `size_overrides`; use per-arch overrides "
            "in the `architectures` section instead."
        )
    config['classifier'] = DEFAULT_CLASSIFIER | config['classifier']
    config['git_hash'] = get_git_revision_hash()
    return config


def _merge_arch_overrides(classifier_base, arch_override):
    """Per-arch overrides win over classifier defaults. The arch entry must
    carry (d_model, n_layers, n_head); any other keys override classifier.
    """
    arch_keys = {'d_model', 'n_layers', 'n_head'}
    missing = arch_keys - set(arch_override)
    if missing:
        raise ValueError(
            f"architecture entry missing required keys: {sorted(missing)}"
        )
    overrides = {k: v for k, v in arch_override.items() if k not in arch_keys}
    arch_only = {k: v for k, v in arch_override.items() if k in arch_keys}
    merged = {**classifier_base, **overrides, **arch_only}
    return merged


def expand_combos(config):
    """Expand the matrix into a flat list of `Combo`s.

    Validates checkpoint paths up front so a missing file fails before any
    GPU work is dispatched. Returns the list sorted by descending compute
    weight for greedy load balancing in `assign_combos`.
    """
    skip_pairs = {
        (s['arch'], s['init']) for s in config.get('skip', [])
    }

    classifier_base = config['classifier']
    train_sizes = list(config['train_sizes'])

    combos = []
    missing_checkpoints = []

    for arch_name, arch_override in config['architectures'].items():
        arch_cfg = _merge_arch_overrides(classifier_base, arch_override)
        for init_name, init_spec in config['inits'].items():
            if (arch_name, init_name) in skip_pairs:
                continue

            checkpoint = None
            if init_spec.get('checkpoint') is not None:
                checkpoint = init_spec['checkpoint']
            elif init_spec.get('checkpoint_template') is not None:
                checkpoint = init_spec['checkpoint_template'].format(arch=arch_name)

            if checkpoint is not None:
                checkpoint = os.path.abspath(os.path.expandvars(checkpoint))
                if not os.path.exists(checkpoint):
                    missing_checkpoints.append((arch_name, init_name, checkpoint))

            for train_size in train_sizes:
                combos.append(Combo(
                    arch_name=arch_name,
                    init_name=init_name,
                    train_size=int(train_size),
                    arch_cfg=arch_cfg,
                    checkpoint=checkpoint,
                ))

    if missing_checkpoints:
        lines = "\n".join(f"  {a} / {i}: {p}" for a, i, p in missing_checkpoints)
        if os.environ.get('DS_GRID_DRY_RUN'):
            print(f"[DRY RUN] WARNING: {len(missing_checkpoints)} checkpoint paths "
                  f"do not exist locally (probably on the cluster):\n{lines}")
        else:
            raise FileNotFoundError(
                f"{len(missing_checkpoints)} checkpoint paths do not exist. "
                f"Failing fast before GPU dispatch:\n{lines}"
            )

    combos.sort(key=lambda c: _combo_weight(c, config['scaling']), reverse=True)
    return combos


def _combo_weight(combo, scaling):
    """FLOPS-proportional load weight: step_budget * d_model^2 * n_layers.

    The d_model^2 * n_layers factor is the standard transformer per-step
    FLOPS proxy (attention QKV + FFN both scale with d_model^2; depth
    multiplies). This corrects for v1's step-only weighting, which would
    severely under-weight d=768 vs d=128 in cross-arch assignment.

    Batch size for the step budget uses `arch_cfg['batch_size']` directly
    or the bs_floor/bs_k scaled value if those keys are present.
    """
    c = combo.arch_cfg
    if 'bs_floor' in c and 'bs_k' in c:
        bs = compute_batch_size(combo.train_size, c['batch_size'], c['bs_floor'], c['bs_k'])
    else:
        bs = min(c['batch_size'], combo.train_size)
    steps = compute_step_budget(combo.train_size, bs, scaling)[0]
    return steps * (c['d_model'] ** 2) * c['n_layers']


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def train_one_combo(rank, combo, config, experiment_dir, tb_dir):
    """Train one (arch, init, train_size) combo on `rank`'s GPU.

    Mirrors v1's `train_one_size` for the single-stage path. Differences:
      - per-combo arch_cfg drives the classifier instantiation
      - encoder init can be random (combo.checkpoint is None) or pretrained
      - output paths are nested: experiment_dir/<init_name>/<arch_name>/n<train_size>
      - CSV header carries arch_name and init_name as the leftmost columns
    """
    c = combo.arch_cfg
    has_pretrained = combo.checkpoint is not None
    has_batch_scaling = 'bs_floor' in c and 'bs_k' in c

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    combo_dir = os.path.join(
        experiment_dir, combo.init_name, combo.arch_name, f'n{combo.train_size}',
    )
    os.makedirs(combo_dir, exist_ok=True)

    tag = f"[GPU {rank} {combo.init_name}/{combo.arch_name}/n={combo.train_size:,}]"
    print(f"{tag} Starting")
    if has_pretrained:
        print(f"{tag} Pretrained checkpoint: {combo.checkpoint}")
    else:
        print(f"{tag} Encoder: random init")

    # Same seed across inits at fixed (arch, train_size) is the fairness
    # guarantee: identical sample order and per-step `randint` indices, only
    # the encoder weights differ.
    set_seed(42)

    # -- Data --
    norm_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'], limit=combo.train_size,
    )
    norm_fn = KineticsNorm(norm_ds, log_transform=True)
    del norm_ds
    print(f"{tag} KineticsNorm means: {norm_fn.means}, stds: {norm_fn.stds}")

    train_ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=combo.train_size, norm_fn=norm_fn, balance=True,
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
    print(f"{tag} Preloaded -- train: {train_x.shape}, val: {val_x.shape}")
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
    # cnn_variant='small_rf' picks DirectClassifierSmallRF (CNN RF=27),
    # required for fine-tuning small-RF SSL encoders (ssl_30, ssl_58)
    # whose encoder.cnn.* state-dict keys are incompatible with the
    # default 11-block CNN. The two classes share head, forward, and
    # encoder.{embed,pe,blocks,layer_norm_target} structure, so the
    # rest of train_one_combo (load_pretrained_encoder, head training,
    # eval) is encoder-agnostic and needs no further changes.
    model_cls = (DirectClassifierSmallRF
                 if c.get('cnn_variant', 'default') == 'small_rf'
                 else DirectClassifier)
    model = model_cls(
        d_model=c['d_model'], n_layers=c['n_layers'],
        n_head=c['n_head'], max_len=c['context'],
    )
    if has_pretrained:
        load_pretrained_encoder(combo.checkpoint, model)
    model = model.to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, "
          f"CNN receptive field: {model.encoder.cnn.r0}")

    criterion = nn.BCEWithLogitsLoss()
    metrics = build_metrics(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(c['max_lr']),
        weight_decay=c['weight_decay'],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=c['pct_start'],
    )

    warmup_steps = int(total_steps * c['pct_start'])
    print(f"{tag} Single-stage: lr={c['max_lr']}, warmup={warmup_steps:,} steps")

    # -- Logging --
    resolved_max_lr = c['max_lr']

    tb_run_dir = os.path.join(
        tb_dir, combo.init_name, combo.arch_name, f'n{combo.train_size}',
    )
    writer = SummaryWriter(log_dir=tb_run_dir)
    writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
    writer.add_text("combo", f"arch={combo.arch_name} init={combo.init_name} n={combo.train_size}", 0)
    writer.add_scalar("train_size", combo.train_size, 0)
    writer.add_scalar("train_bs", train_bs, 0)
    writer.add_scalar("max_lr", float(resolved_max_lr), 0)

    csv_path = os.path.join(combo_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'arch_name', 'init_name', 'train_size', 'train_bs', 'max_lr',
        'eval_point', 'step', 'stage',
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
    stage = 1  # constant for v2 (single-stage); kept for CSV-schema compat with v1

    print(f"{tag} Eval schedule ({n_evals} points): {eval_steps}")

    # -- Training loop --
    for step in range(1, total_steps + 1):
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
                combo.arch_name, combo.init_name,
                combo.train_size, train_bs, resolved_max_lr,
                eval_point, step, stage,
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

            ckpt_path = os.path.join(combo_dir, f'step{step}.pt')
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': model.encoder.state_dict(),
                    'config': config,
                    'arch_name': combo.arch_name,
                    'init_name': combo.init_name,
                    'train_size': combo.train_size,
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

def assign_combos(combos, world_size, scaling):
    """Greedy load-balanced assignment of combos to GPUs by FLOPS-weighted
    work. Sort combos by weight desc, assign each to the GPU with the
    smallest accumulated weight.
    """
    assignments = [[] for _ in range(world_size)]
    workloads = [0.0] * world_size

    sorted_combos = sorted(
        combos, key=lambda c: _combo_weight(c, scaling), reverse=True,
    )
    for combo in sorted_combos:
        gpu = min(range(world_size), key=lambda i: workloads[i])
        assignments[gpu].append(combo)
        workloads[gpu] += _combo_weight(combo, scaling)

    return assignments, workloads


def worker_fn(rank, assignments, config, experiment_dir, tb_dir):
    for combo in assignments[rank]:
        train_one_combo(rank, combo, config, experiment_dir, tb_dir)
        torch.cuda.empty_cache()


def merge_csvs_v2(experiment_dir, combos):
    """Walk the (init/arch/n*) tree and concatenate per-combo CSVs into a
    single experiment_dir/results.csv. Each per-combo CSV already has
    `arch_name` and `init_name` injected by the worker, so the merge is a
    straight concatenation; sort the merged rows by
    (init_name, arch_name, train_size, eval_point) for stable ordering.
    """
    merged_path = os.path.join(experiment_dir, 'results.csv')
    rows = []
    header = None
    missing = []

    for combo in combos:
        per_path = os.path.join(
            experiment_dir, combo.init_name, combo.arch_name,
            f'n{combo.train_size}', 'results.csv',
        )
        if not os.path.exists(per_path):
            missing.append(per_path)
            continue
        with open(per_path) as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            rows.extend(list(reader))

    if missing:
        print(f"WARNING: {len(missing)} per-combo CSVs missing during merge:")
        for p in missing:
            print(f"  {p}")

    if header is None:
        print("WARNING: no per-combo CSVs found; skipping merge")
        return

    init_idx = header.index('init_name')
    arch_idx = header.index('arch_name')
    ts_idx = header.index('train_size')
    ep_idx = header.index('eval_point')
    rows.sort(key=lambda r: (r[init_idx], r[arch_idx], int(r[ts_idx]), int(r[ep_idx])))

    with open(merged_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Merged results into {merged_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path):
    config = load_config(config_path)

    combos = expand_combos(config)
    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    tb_dir = os.path.join(experiment_dir, 'training_logs')
    os.makedirs(tb_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    forced = os.environ.get('DS_GRID_FORCE_WORLD_SIZE')
    if forced and os.environ.get('DS_GRID_DRY_RUN'):
        # Lets a no-GPU dry run preview the load balance for a target world size.
        num_gpus = int(forced)
    world_size = max(1, min(num_gpus, len(combos)))

    # Summary
    init_counts = {}
    arch_counts = {}
    for combo in combos:
        init_counts[combo.init_name] = init_counts.get(combo.init_name, 0) + 1
        arch_counts[combo.arch_name] = arch_counts.get(combo.arch_name, 0) + 1

    print(f"Experiment dir: {experiment_dir}")
    print(f"Total combos: {len(combos)}")
    print(f"  by init: {init_counts}")
    print(f"  by arch: {arch_counts}")
    print(f"Train sizes: {config['train_sizes']}")
    s = config['scaling']
    print(f"Step budget: max_epochs={s['max_epochs']}, "
          f"min_steps={s['min_steps']:,}, max_steps={s['max_steps']:,}")
    print(f"Single-stage: max_lr default={config['classifier']['max_lr']}, "
          f"weight_decay={config['classifier']['weight_decay']}, "
          f"pct_start={config['classifier']['pct_start']}")

    print(f"GPUs available: {num_gpus}, using {world_size} workers")

    assignments, workloads = assign_combos(combos, world_size, config['scaling'])
    if workloads:
        max_w, min_w = max(workloads), min(workloads)
        balance = (max_w - min_w) / max_w if max_w > 0 else 0.0
        print(f"Load balance: max/min workload weights {max_w:.2e}/{min_w:.2e} "
              f"(spread {balance:.1%})")
    for i, combos_i in enumerate(assignments):
        names = [f"{c.init_name}/{c.arch_name}/n{c.train_size}" for c in combos_i]
        print(f"  GPU {i} ({len(combos_i)} combos): {names}")

    if os.environ.get('DS_GRID_DRY_RUN'):
        print("[DRY RUN] DS_GRID_DRY_RUN set; exiting before worker dispatch.")
        return

    if num_gpus == 0:
        raise RuntimeError(
            "No CUDA devices visible. ds_grid_v2 requires GPUs; submit on the cluster."
        )

    if world_size > 1:
        mp.spawn(
            worker_fn,
            args=(assignments, config, experiment_dir, tb_dir),
            nprocs=world_size,
        )
    else:
        worker_fn(0, assignments, config, experiment_dir, tb_dir)

    merge_csvs_v2(experiment_dir, combos)
    print(f"TensorBoard logs in {tb_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
