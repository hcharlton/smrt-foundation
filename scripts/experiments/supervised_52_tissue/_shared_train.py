"""Shared training loop for supervised_52_tissue.

Step-cadence supervised harness for the 8-way tissue classifier. Forks
the structure of `ssl_58_autoencoder_grid/_shared_train.py` (step-driven
loop, ProgressState, scheduler-prep skip, forward-hook diagnostics, atomic
resume, per-eval CSV + per-eval portable checkpoint, step-0 baseline)
and replaces the SSL-specific bits with supervised-specific ones:

  - Model: TissueClassifier (encoder + center-latent multiclass head)
  - Loss: nn.CrossEntropyLoss over 8 tissue classes
  - Train data: TissueMemmapDataset filtered to split=='train' from
    `<data_dir>/partition.csv`. The partition is the verifiable on-disk
    artifact produced by `scripts.make_tissue_partition`.
  - Val data: two TissueMemmapDataset instances filtered to 'val_s1'
    (held-out reads from the same cell as train) and 'val_s2' (a cell
    never seen in train). Each evaluated independently per tick.
  - Online normalisation via KineticsNorm(n_continuous=4); stats computed
    on the train split at startup and round-tripped through checkpoints
    via save_stats / load_stats.
  - Deterministic center-crop from build context (4096) to training
    context (e.g. 2048). No augmentation.
  - `dataset_on_gpu` knob: when true, materialise each split as a single
    GPU-resident TensorDataset at startup (50k * 2048 * 6 * 4B ~ 2.4 GB).
    Drops the IO path entirely; each rank holds a full copy. When false,
    fall back to TissueMemmapDataset + ChunkedRandomSampler (the standard
    memmap loader).

The scheduler is intentionally NOT prepared via `accelerator.prepare(...)`
(supervised_40 fix preserved).
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import yaml
import torch
import torch.nn as nn
import polars as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset
from accelerate import Accelerator, DistributedDataParallelKwargs, skip_first_batches
from accelerate.utils import set_seed
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


LOG_EVERY = 100  # steps between stdout status lines

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import TissueMemmapDataset, ChunkedRandomSampler
from smrt_foundation.model import TissueClassifier
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


class CroppedNormedDataset(Dataset):
    """Apply norm_fn + deterministic center-crop to each (x, y) sample.

    The yoran tissue memmap stores reads at the build context (4096). The
    classifier trains at a shorter context (typically 2048). Center crop is
    deterministic so every read maps to the same window every epoch (no
    augmentation; matches the README spec).
    """
    def __init__(self, inner, norm_fn, crop_len):
        self.inner = inner
        self.norm_fn = norm_fn
        self.crop_len = int(crop_len)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x, y = self.inner[idx]
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        T = x.shape[0]
        if T > self.crop_len:
            start = (T - self.crop_len) // 2
            x = x[start:start + self.crop_len]
        return x, y


class ProgressState:
    """Tracks training progress across interrupted runs.

    Registered with `accelerator.register_for_checkpointing(...)` so the
    counters are part of the Accelerate state directory and restore
    atomically alongside model / optimizer / scheduler / RNG state.
    """
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.step_in_epoch = 0

    def state_dict(self):
        return {
            'epoch': int(self.epoch),
            'global_step': int(self.global_step),
            'step_in_epoch': int(self.step_in_epoch),
        }

    def load_state_dict(self, sd):
        self.epoch = int(sd.get('epoch', 0))
        self.global_step = int(sd.get('global_step', 0))
        self.step_in_epoch = int(sd.get('step_in_epoch', 0))


def _check_resume_compatible(resume_dir, config):
    """Refuse to resume if the stored architecture differs from the current
    config. Emits a warning (not an error) on git hash mismatch.
    """
    sidecar = os.path.join(resume_dir, 'run_metadata.yaml')
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"Resume target {resume_dir} has no run_metadata.yaml sidecar; "
            f"refusing to resume (cannot verify architecture match)."
        )
    with open(sidecar, 'r') as f:
        stored = yaml.safe_load(f)
    cur = config.get('classifier', {})
    prev = stored.get('classifier', {})
    arch_keys = ('d_model', 'n_layers', 'n_head', 'context',
                 'n_classes', 'n_continuous')
    for k in arch_keys:
        if cur.get(k) != prev.get(k):
            raise RuntimeError(
                f"Refusing to resume: classifier.{k} differs "
                f"(stored={prev.get(k)}, current={cur.get(k)}). Architecture "
                f"must match for Accelerate state_dict to load."
            )
    if stored.get('git_hash') and config.get('git_hash') and stored['git_hash'] != config['git_hash']:
        print(
            f"[resume] WARNING: git hash differs "
            f"(stored={stored['git_hash'][:12]}, current={config['git_hash'][:12]}). "
            f"Architecture matches so resume will proceed."
        )
    return stored


def _materialize_to_gpu(wrapped_ds, device, log=print, name='split'):
    """Iterate `wrapped_ds` once on CPU, stack into one GPU tensor, return TensorDataset.

    `wrapped_ds` is expected to yield `(x, y)` per item. The resulting
    TensorDataset holds X and y entirely on `device`. Useful when the split
    fits in GPU memory and we want to drop the DataLoader IO path.
    """
    log(f"[{name}] materialising {len(wrapped_ds)} samples to {device} ...")
    t0 = time.perf_counter()
    xs, ys = [], []
    for i in range(len(wrapped_ds)):
        x, y = wrapped_ds[i]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs, dim=0).contiguous()
    Y = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in ys], dim=0).contiguous()
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)
    dt = time.perf_counter() - t0
    log(f"[{name}] X={tuple(X.shape)} {X.dtype}, Y={tuple(Y.shape)} {Y.dtype}, "
        f"{X.element_size() * X.nelement() / 1e9:.2f} GB on {device}, took {dt:.1f}s")
    return TensorDataset(X, Y)


def tissue_eval(model, val_dl, accelerator, name, n_classes=8):
    """Single-process eval on `val_dl`. Returns metrics dict namespaced by `name`.

    The val DataLoader is NOT prepared via `accelerator.prepare`, so each
    rank holds its own copy of the val set. The model is unwrapped before
    forward — calling the DDP-wrapped module on a single rank would trigger
    a buffer-sync collective that the other ranks (sitting in the trailing
    `wait_for_everyone()`) won't participate in, causing a 10-min NCCL
    timeout. Unwrap bypasses DDP entirely; non-main ranks are pure no-ops
    inside the barrier bookend. Same effect as ssl_58 passing
    `unwrapped.encoder` into ssl_pair_val_eval.
    """
    device = accelerator.device
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    metrics = {}

    if accelerator.is_main_process:
        top1_metric = MulticlassAccuracy(num_classes=n_classes, top_k=1, average='micro').to(device)
        top3_metric = MulticlassAccuracy(num_classes=n_classes, top_k=min(3, n_classes), average='micro').to(device)
        f1_metric = MulticlassF1Score(num_classes=n_classes, average='macro').to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss_sum = 0.0
        n_total = 0

        for x, y in val_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            with torch.no_grad():
                logits = unwrapped(x)
                loss_sum += float(criterion(logits, y).item())
                n_total += int(y.shape[0])
            top1_metric.update(logits, y)
            top3_metric.update(logits, y)
            f1_metric.update(logits, y)

        if n_total > 0:
            metrics[f'{name}/loss'] = loss_sum / n_total
            metrics[f'{name}/top1'] = float(top1_metric.compute().item())
            metrics[f'{name}/top3'] = float(top3_metric.compute().item())
            metrics[f'{name}/macro_f1'] = float(f1_metric.compute().item())

    accelerator.wait_for_everyone()
    return metrics


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT = {
        'd_model': 128, 'n_layers': 4, 'n_head': 4,
        'context': 2048, 'n_classes': 8, 'n_continuous': 4,
        'batch_size': 128, 'total_steps': 100_000, 'ds_limit': 0,
        'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
        'grad_clip': 5.0,
        'eval_every_steps': 5000,
        'resume_every_steps': 5000,
        'dataset_on_gpu': True,
        'chunk_size': 2048,
    }
    c = DEFAULT | config.get('classifier', {})
    config['classifier'] = c
    config['git_hash'] = get_git_revision_hash()

    data_dir = os.path.expandvars(config['data_dir'])
    partition_path = os.path.expandvars(
        config.get('partition_path', os.path.join(data_dir, 'partition.csv'))
    )
    if not os.path.exists(partition_path):
        raise FileNotFoundError(
            f"partition.csv missing at {partition_path}. Build it with "
            f"`python -m scripts.make_tissue_partition ...` before running this experiment."
        )

    resume_from = config.get('resume_from') or None
    if resume_from:
        resume_from = os.path.abspath(os.path.expandvars(str(resume_from)))
        if not os.path.isdir(resume_from):
            raise FileNotFoundError(
                f"resume_from={resume_from!r} is not a directory. It should point "
                f"to an Accelerate state directory (e.g. <exp>/checkpoints/latest)."
            )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with='tensorboard',
        project_dir='training_logs',
        kwargs_handlers=[ddp_kwargs],
    )

    def log(msg):
        if accelerator.is_main_process:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    if accelerator.is_main_process:
        print(config)

    set_seed(42)

    exp_type = config.get('experiment_type', 'supervised')
    exp_name = config.get('experiment_name', 'tissue_experiment')
    project_namespace = f"{exp_type}/{exp_name}"

    tracker = None
    csv_path = None
    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker('tensorboard')
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(config, f)
        tracker.writer.add_text('Full_Config', f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

        # Per-eval CSV: one row per eval tick (including step-0 baseline).
        # Same metrics as the corresponding TB scalars, in a flat file for
        # offline plotting / cross-experiment aggregation. Header written
        # once at run start; rows appended by the eval block at every tick.
        import csv as _csv_mod
        csv_path = os.path.join(run_dir, 'eval_history.csv')
        with open(csv_path, 'w', newline='') as f:
            _csv_mod.writer(f).writerow([
                'step', 'walltime_s',
                'val_s1/loss', 'val_s1/top1', 'val_s1/top3', 'val_s1/macro_f1',
                'val_s2/loss', 'val_s2/top1', 'val_s2/top3', 'val_s2/macro_f1',
            ])

    # --- Partition + datasets ---
    partition = pl.read_csv(partition_path)
    splits = sorted(set(partition['split'].to_list()))
    expected_splits = {'train', 'val_s1', 'val_s2'}
    if not expected_splits.issubset(splits):
        raise ValueError(
            f"partition.csv at {partition_path} has splits {splits}; "
            f"expected at least {sorted(expected_splits)}."
        )

    train_names = partition.filter(pl.col('split') == 'train')['read_name']
    val_s1_names = partition.filter(pl.col('split') == 'val_s1')['read_name']
    val_s2_names = partition.filter(pl.col('split') == 'val_s2')['read_name']

    if accelerator.is_main_process:
        print(
            f"partition splits: train={train_names.len()}, "
            f"val_s1={val_s1_names.len()}, val_s2={val_s2_names.len()}"
        )

    # Raw datasets (no norm, no crop). Apply ds_limit only on the train
    # split — vals are small and should be evaluated in full.
    train_raw = TissueMemmapDataset(
        data_dir, filter_expr=pl.col('read_name').is_in(train_names),
    )
    if c['ds_limit'] and c['ds_limit'] > 0 and c['ds_limit'] < len(train_raw):
        # ds_limit truncates the train pool; deterministic prefix slice.
        # For a randomised subset, build the partition itself with fewer rows.
        train_raw.refs = train_raw.refs[:c['ds_limit']]
    val_s1_raw = TissueMemmapDataset(
        data_dir, filter_expr=pl.col('read_name').is_in(val_s1_names),
    )
    val_s2_raw = TissueMemmapDataset(
        data_dir, filter_expr=pl.col('read_name').is_in(val_s2_names),
    )

    # --- Normalisation ---
    if resume_from:
        norm_path = os.path.join(resume_from, 'norm_stats.pt')
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"resume_from is set but {norm_path} is missing. "
                f"Recomputing stats would shift the data distribution."
            )
        norm_fn = KineticsNorm.load_stats(torch.load(norm_path, map_location='cpu'))
        if accelerator.is_main_process:
            print(f"[resume] loaded norm stats from {norm_path}")
    else:
        norm_fn = KineticsNorm(
            train_raw,
            log_transform=True,
            max_samples=16_384,
            n_continuous=int(c['n_continuous']),
        )
    if accelerator.is_main_process:
        print(f"norm — n_continuous={norm_fn.n_continuous}")
        print(f"norm — means: {norm_fn.means}")
        print(f"norm — stds:  {norm_fn.stds}")

    # --- Wrap with crop + norm ---
    crop_len = int(c['context'])
    train_wrapped = CroppedNormedDataset(train_raw, norm_fn, crop_len=crop_len)
    val_s1_wrapped = CroppedNormedDataset(val_s1_raw, norm_fn, crop_len=crop_len)
    val_s2_wrapped = CroppedNormedDataset(val_s2_raw, norm_fn, crop_len=crop_len)

    # --- DataLoaders: branch on dataset_on_gpu ---
    dataset_on_gpu = bool(c.get('dataset_on_gpu', True))
    if dataset_on_gpu:
        # Each rank materialises its own copy of every split on its GPU.
        # 50k * 2048 * 6 * 4B = ~2.4 GB; trivially fits on each rank.
        train_ds = _materialize_to_gpu(train_wrapped, accelerator.device, log=log, name='train')
        val_s1_ds = _materialize_to_gpu(val_s1_wrapped, accelerator.device, log=log, name='val_s1')
        val_s2_ds = _materialize_to_gpu(val_s2_wrapped, accelerator.device, log=log, name='val_s2')
        # num_workers=0 is required for GPU-resident TensorDatasets:
        # DataLoader workers cannot share GPU memory.
        train_dl = DataLoader(
            train_ds, batch_size=int(c['batch_size']),
            shuffle=True, num_workers=0, drop_last=True,
        )
        val_s1_dl = DataLoader(
            val_s1_ds, batch_size=int(c['batch_size']),
            shuffle=False, num_workers=0, drop_last=False,
        )
        val_s2_dl = DataLoader(
            val_s2_ds, batch_size=int(c['batch_size']),
            shuffle=False, num_workers=0, drop_last=False,
        )
    else:
        # Standard memmap loader path: ChunkedRandomSampler for shard-cache
        # locality, multiple workers + prefetch for IO overlap.
        sampler = ChunkedRandomSampler(train_wrapped, c['chunk_size'], shuffle_within=True)
        train_dl = DataLoader(
            train_wrapped, batch_size=int(c['batch_size']),
            num_workers=6, prefetch_factor=8, pin_memory=True,
            shuffle=False, sampler=sampler, persistent_workers=True,
            drop_last=True,
        )
        val_s1_dl = DataLoader(
            val_s1_wrapped, batch_size=int(c['batch_size']),
            num_workers=2, shuffle=False, drop_last=False,
        )
        val_s2_dl = DataLoader(
            val_s2_wrapped, batch_size=int(c['batch_size']),
            num_workers=2, shuffle=False, drop_last=False,
        )

    # --- Model / loss / optim ---
    model = TissueClassifier(
        d_model=int(c['d_model']),
        n_layers=int(c['n_layers']),
        n_head=int(c['n_head']),
        max_len=int(c['context']),
        n_classes=int(c['n_classes']),
        n_continuous=int(c['n_continuous']),
    )

    if accelerator.is_main_process and tracker is not None:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"CNN receptive field: {model.encoder.cnn.r0} bases")
        tracker.writer.add_scalar('architecture/cnn_receptive_field', model.encoder.cnn.r0, 0)
        tracker.writer.add_scalar('architecture/param_count', n_params, 0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(c['max_lr']),
        weight_decay=float(c['weight_decay']),
    )
    criterion = nn.CrossEntropyLoss()

    # NB: only the train loader is prepared; vals are kept unprepared so
    # each rank sees the full set and metrics are computed on main only.
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    # Forward hook on the encoder submodule to capture the transformer
    # output `c` for diagnostic logging. Same TB keys as ssl_58 so
    # trajectories overlay visually. embed_z_std cratering toward 0 = collapse.
    _diag = {}
    def _capture_c(module, _input, output):
        _diag['c'] = output
    accelerator.unwrap_model(model).encoder.register_forward_hook(_capture_c)

    total_steps = int(c['total_steps'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, total_steps=total_steps, pct_start=float(c['pct_start']),
    )

    progress_state = ProgressState()
    accelerator.register_for_checkpointing(progress_state)

    if resume_from:
        _check_resume_compatible(resume_from, config)
        accelerator.load_state(resume_from)
        if accelerator.is_main_process:
            print(
                f"[resume] restored from {resume_from}: "
                f"epoch={progress_state.epoch}, global_step={progress_state.global_step}"
            )

    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(train_dl)}, Total steps: {total_steps}")
        print(f"Warmup steps: {int(total_steps * c['pct_start'])}")
        print(
            f"Eval every {c['eval_every_steps']} global steps "
            f"(per-eval ckpt + eval_history.csv row), "
            f"resume ckpt every {c['resume_every_steps']} global steps"
        )

    checkpoint_dir = os.path.join(os.path.dirname(config_path), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_dir = os.path.join(checkpoint_dir, 'latest')

    global_step = progress_state.global_step
    nonfinite_skip_count = 0
    run_start_t = time.time()

    def _save_latest(epoch_idx, step_in_epoch_count, global_step_count):
        progress_state.epoch = epoch_idx
        progress_state.step_in_epoch = step_in_epoch_count
        progress_state.global_step = global_step_count
        accelerator.save_state(latest_dir)
        if accelerator.is_main_process:
            with open(os.path.join(latest_dir, 'run_metadata.yaml'), 'w') as f:
                yaml.dump(config, f)
            torch.save(norm_fn.save_stats(), os.path.join(latest_dir, 'norm_stats.pt'))
            log(f"[resume] saved latest/ at global_step {global_step_count} "
                f"(epoch {epoch_idx+1}, step_in_epoch {step_in_epoch_count}/{len(train_dl)})")
        accelerator.wait_for_everyone()

    def _save_milestone(step, epoch_idx, step_in_epoch_count, metrics):
        """Per-eval portable checkpoint. Full model + encoder-only state +
        config + per-eval metrics + KineticsNorm sidecar.
        """
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = os.path.join(checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'model_state_dict': unwrapped.state_dict(),
                'encoder_state_dict': unwrapped.encoder.state_dict(),
                'config': config,
                'epoch': epoch_idx,
                'global_step': step,
                'step_in_epoch': step_in_epoch_count,
                'metrics': metrics,
                **norm_fn.save_stats(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
        accelerator.wait_for_everyone()

    def _log_eval_csv(step, metrics):
        """Append one row per eval tick to <run_dir>/eval_history.csv."""
        if not accelerator.is_main_process or csv_path is None:
            return
        import csv as _csv_mod
        with open(csv_path, 'a', newline='') as f:
            _csv_mod.writer(f).writerow([
                step, time.time() - run_start_t,
                metrics.get('val_s1/loss', float('nan')),
                metrics.get('val_s1/top1', float('nan')),
                metrics.get('val_s1/top3', float('nan')),
                metrics.get('val_s1/macro_f1', float('nan')),
                metrics.get('val_s2/loss', float('nan')),
                metrics.get('val_s2/top1', float('nan')),
                metrics.get('val_s2/top3', float('nan')),
                metrics.get('val_s2/macro_f1', float('nan')),
            ])

    def _run_eval_block(step, epoch_idx, step_in_epoch_count):
        m1 = tissue_eval(model, val_s1_dl, accelerator, name='val_s1',
                         n_classes=int(c['n_classes']))
        m2 = tissue_eval(model, val_s2_dl, accelerator, name='val_s2',
                         n_classes=int(c['n_classes']))
        combined = {**m1, **m2}
        if accelerator.is_main_process:
            accelerator.log(combined, step=step)
            print(
                f"step {step}: "
                f"val_s1 top1={combined.get('val_s1/top1', float('nan')):.4f} "
                f"loss={combined.get('val_s1/loss', float('nan')):.4f} | "
                f"val_s2 top1={combined.get('val_s2/top1', float('nan')):.4f} "
                f"loss={combined.get('val_s2/loss', float('nan')):.4f}"
            )
        _log_eval_csv(step, combined)
        _save_milestone(step, epoch_idx, step_in_epoch_count, combined)
        accelerator.wait_for_everyone()
        model.train()
        return combined

    # Step-0 baseline eval: anchors trajectories at the random-init encoder.
    # 8-class CE expects ~ln(8) = 2.08 loss and ~12.5% top1 by chance.
    if global_step == 0:
        log("=== step 0 baseline eval (random-init model) ===")
        _run_eval_block(0, 0, 0)
        _prev_t = time.perf_counter()
        _window_start_t = _prev_t
    else:
        _prev_t = time.perf_counter()
        _window_start_t = _prev_t

    epoch = progress_state.epoch
    done = global_step >= total_steps
    while not done:
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        skip_n = progress_state.step_in_epoch if epoch == progress_state.epoch else 0
        if skip_n > 0:
            effective_dl = skip_first_batches(train_dl, skip_n)
            log(f"=== epoch {epoch+1} resume at "
                f"step_in_epoch {skip_n}/{len(train_dl)}, "
                f"global_step={global_step}/{total_steps} ===")
        else:
            effective_dl = train_dl
            log(f"=== epoch {epoch+1} start "
                f"({len(train_dl)} steps, global_step={global_step}/{total_steps}) ===")
        _prev_t = time.perf_counter()
        _window_start_t = _prev_t

        for step_offset, batch in enumerate(effective_dl):
            step_in_epoch_done = skip_n + step_offset + 1
            x, y = batch
            y = y.long()

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=float(c['grad_clip']))
            assert grad_norm is not None, "clip_grad_norm_ returned None — model has no params with grads"
            grad_finite = bool(torch.isfinite(grad_norm).item())
            if grad_finite:
                optimizer.step()
                scheduler.step()
            else:
                nonfinite_skip_count += 1

            global_step += 1
            current_loss = accelerator.reduce(loss, reduction='mean').item()
            epoch_loss_sum += current_loss
            epoch_loss_count += 1

            with torch.no_grad():
                c_tensor = _diag['c']
                c_flat = c_tensor.reshape(-1, c_tensor.shape[-1])
                z_std_local = c_flat.std(dim=0).mean()
                z_norm_local = c_flat.norm(dim=1).mean()
            grad_norm_r = accelerator.reduce(grad_norm, reduction='mean').item()
            z_std = accelerator.reduce(z_std_local, reduction='mean').item()
            z_norm = accelerator.reduce(z_norm_local, reduction='mean').item()

            _now = time.perf_counter()
            step_ms = (_now - _prev_t) * 1000.0
            _prev_t = _now
            its = 1000.0 / step_ms if step_ms > 0 else 0.0

            accelerator.log({
                'train_loss': current_loss,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch,
                'grad_norm': grad_norm_r,
                'embed_z_std': z_std,
                'embed_z_norm': z_norm,
                'step_time_ms': step_ms,
                'iters_per_sec': its,
                'nonfinite_skip_count': nonfinite_skip_count,
            }, step=global_step)

            if global_step % LOG_EVERY == 0:
                window_secs = _now - _window_start_t
                window_its = LOG_EVERY / window_secs if window_secs > 0 else 0.0
                _window_start_t = _now
                log(f"ep {epoch+1} "
                    f"step {global_step}/{total_steps} "
                    f"({step_in_epoch_done}/{len(train_dl)}) | "
                    f"loss={current_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{window_its:.2f} it/s "
                    f"grad={grad_norm_r:.2f}")

            eval_every_steps = int(c['eval_every_steps'])
            resume_every_steps = int(c['resume_every_steps'])
            tick_eval = (eval_every_steps > 0
                         and global_step % eval_every_steps == 0)
            tick_resume = (resume_every_steps > 0
                           and global_step % resume_every_steps == 0)

            if tick_eval:
                _run_eval_block(global_step, epoch, step_in_epoch_done)
                _prev_t = time.perf_counter()
                _window_start_t = _prev_t

            if tick_resume:
                _save_latest(
                    epoch_idx=epoch,
                    step_in_epoch_count=step_in_epoch_done,
                    global_step_count=global_step,
                )

            if global_step >= total_steps:
                done = True
                break

        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        accelerator.log({'epoch_avg_loss': avg_loss}, step=global_step)
        log(f"=== epoch {epoch+1} end | "
            f"avg_loss={avg_loss:.4f} over {epoch_loss_count} batches ===")

        epoch += 1
        progress_state.epoch = epoch
        progress_state.step_in_epoch = 0
        progress_state.global_step = global_step

    # --- Final checkpoint ---
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'config': config,
            'epoch': epoch,
            'global_step': global_step,
            **norm_fn.save_stats(),
        }, save_path)
        print(f"Saved final checkpoint to {save_path}")

    accelerator.end_training()


if __name__ == '__main__':
    main()
