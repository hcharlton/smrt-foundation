"""Shared input-masked-prediction training loop for ssl_57.

Combines the learning task of ssl_21 (`Smrt2VecInputMask` + `AgInfoNCE`)
with the modern harness from ssl_55/ssl_56:

  - LayerNorm projection head (`Smrt2VecInputMaskLN`, the masked-pred
    sibling of ssl_55's `SimCLRSmrtLN`). Bounds the per-channel scale of
    the per-position projection feeding `AgInfoNCE`'s `F.normalize`,
    eliminating the same magnitude-runaway pathway ssl_55 fixed for the
    SimCLR head.
  - Non-finite-grad skip on `clip_grad_norm_` (defensive; with the LN
    head the `nonfinite_skip_count` TB scalar should stay at 0).
  - `ChunkedRandomSampler` (sequential shard reads, ~3-6x I/O speedup
    over random `shuffle=True` on the source memmap).
  - Step-based cadences (probe / portable encoder ckpt / Accelerate
    state resume), all triggered on `global_step % N == 0`.
  - Step-cap schedule: `total_steps` is set explicitly in config
    (replacing ssl_55/56's `len(dl) * epochs` derivation). The cosine
    schedule's horizon is `total_steps`. The training loop iterates the
    DataLoader as many epochs as fit and exits when `global_step >=
    total_steps` regardless of epoch boundary.
  - Two SSL pair-val splits ('train' = yoran in-distribution, 'val' =
    ob007 held-out) plus the CpG linear probe, both at every
    `probe_every_steps`.

Adaptation of ssl_56's `ssl_pair_val_eval` for a non-pair-trained
encoder: `Smrt2VecInputMaskLN.forward` returns `(c_proj, targets,
mask_idx)` and applies input masking, neither of which is right at eval
time. The adapter calls `encoder.forward(view)` directly and pools to a
single embedding via center-latent — the same convention the CpG linear
probe uses. No projection (the project head was trained for per-position
context-from-context retrieval, not pair retrieval).

ssl_21 used `p_mask=0.15`; ssl_57 starts with `p_mask=0.05` per the user
decision. Both `p_mask` and `mask_size` are config knobs under
`smrt2vec:` and can be tuned without code changes.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs, skip_first_batches
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


LOG_EVERY = 100  # steps between stdout status lines

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import (
    ShardedMemmapDataset, LabeledMemmapDataset,
    PairedGapMemmapDataset, ChunkedRandomSampler,
)
from smrt_foundation.model import Smrt2VecInputMaskLN, Smrt2VecInputMaskLNSmallRF
from smrt_foundation.loss import AgInfoNCE
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


class NormedDataset(Dataset):
    """Thin wrapper that applies a normalization callable to each sample.

    ssl_21 used the same pattern (an inline class). Lifting it to a
    module-level Dataset so `accelerator.prepare(dl)` can pickle it for
    DataLoader workers without the pickling-an-inner-class warning.
    """
    def __init__(self, inner, norm_fn):
        self.inner = inner
        self.norm_fn = norm_fn

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.norm_fn(self.inner[idx])


class ProgressState:
    """Tracks training progress across interrupted runs.

    Registered with `accelerator.register_for_checkpointing(...)` so the
    counters are part of the Accelerate state directory and restore
    atomically alongside model / optimizer / scheduler / RNG state.

    Three fields:
      - `epoch`: index of the current epoch through the data. Doesn't
        drive the schedule (which is keyed off `total_steps`); only
        bookkeeping for the resume DataLoader skip.
      - `global_step`: total training steps completed.
      - `step_in_epoch`: batches consumed from the start of the current
        epoch (0 at the start of a fresh epoch). Used to
        `skip_first_batches` on the DataLoader after a mid-epoch
        resume, so no already-processed batch is re-run.
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
    config. Emits a warning (not an error) on git hash mismatch; code
    drift that preserves the architecture is usually fine to resume into.
    """
    sidecar = os.path.join(resume_dir, 'run_metadata.yaml')
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"Resume target {resume_dir} has no run_metadata.yaml sidecar; "
            f"refusing to resume (cannot verify architecture match)."
        )
    with open(sidecar, 'r') as f:
        stored = yaml.safe_load(f)
    cur = config.get('smrt2vec', {})
    prev = stored.get('smrt2vec', {})
    arch_keys = ('d_model', 'n_layers', 'n_head', 'context',
                 'p_mask', 'mask_size')
    for k in arch_keys:
        if cur.get(k) != prev.get(k):
            raise RuntimeError(
                f"Refusing to resume: smrt2vec.{k} differs "
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


def linear_probe_eval(encoder, probe_config, config, accelerator, norm_fn):
    """Freeze encoder, train a linear head on labeled CpG data, report accuracy.

    Verbatim from ssl_56 (and ssl_55, and ssl_26): probe_top1 is
    comparable across the SSL family. Center-latent pooling matches
    DirectClassifier convention.
    """
    device = accelerator.device
    encoder.eval()

    pc = probe_config
    probe_limit = pc.get('ds_limit', 500000)

    train_ds = LabeledMemmapDataset(
        config.get('probe_pos_train'), config.get('probe_neg_train'),
        limit=probe_limit, norm_fn=norm_fn, balance=True
    )
    val_ds = LabeledMemmapDataset(
        config.get('probe_pos_val'), config.get('probe_neg_val'),
        limit=probe_limit, norm_fn=norm_fn, balance=True
    )

    train_dl = DataLoader(train_ds, batch_size=pc.get('batch_size', 512), shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=pc.get('batch_size', 512), shuffle=False, num_workers=2)

    d_model = encoder.d_model
    probe_head = nn.Linear(d_model, 1).to(device)
    probe_opt = torch.optim.Adam(probe_head.parameters(), lr=float(pc.get('lr', 3e-3)))
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(pc.get('epochs', 3)):
        probe_head.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                c = encoder.forward(x)
                center = c[:, c.shape[1] // 2, :]
            logits = probe_head(center).squeeze(-1)
            loss = criterion(logits, y)
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()

    probe_head.eval()
    acc_metric = BinaryAccuracy().to(device)
    auroc_metric = BinaryAUROC().to(device)

    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            c = encoder.forward(x)
            center = c[:, c.shape[1] // 2, :]
            logits = probe_head(center).squeeze(-1)
        acc_metric.update(logits > 0, y.long())
        auroc_metric.update(logits, y.long())

    top1 = acc_metric.compute().item()
    auroc = auroc_metric.compute().item()

    del probe_head, probe_opt, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache()

    return top1, auroc


def ssl_pair_val_eval(
    encoder,
    val_data_dir,
    accelerator,
    norm_fn,
    temperature,
    batch_size=1024,
    limit_per_gap=10000,
    name='train',
):
    """Per-gap SSL pair val metrics, adapted from ssl_56 for a non-pair-
    trained encoder.

    Unlike ssl_56's `ssl_pair_val_eval`, which calls the SimCLR model
    directly (it has a `(view1, view2) -> (z1, z2)` forward), here we
    take the bare `SmrtEncoder` and (a) bypass the input-mask path
    (no masking at eval), (b) pool the per-position output via
    center-latent — the same convention the CpG linear probe uses, so
    the pair-val and probe metrics are evaluating the same underlying
    representation. No projection head is applied: the project MLP was
    trained for per-position context-from-context retrieval, not pair
    retrieval, so applying it here would mix two different signals.

    The DataLoader is *not* `accelerator.prepare`'d, so each rank sees
    the full per-gap subset and the metrics are deterministic across
    ranks (no sync needed). `name` prefixes every TB key:
      val_ssl/{name}_loss_gap_{g}
      val_ssl/{name}_top1_gap_{g}
      val_ssl/{name}_pos_cos_gap_{g}
    plus a global summary `val_ssl/{name}_spearman_cos_vs_gap`.
    """
    device = accelerator.device
    encoder.eval()

    full_ds = PairedGapMemmapDataset(val_data_dir, norm_fn=norm_fn)
    unique_gaps = sorted(set(int(g) for g in full_ds.gaps_all))
    del full_ds

    metrics = {}
    pos_cos_at_gap_means = []

    for g in unique_gaps:
        ds_g = PairedGapMemmapDataset(
            val_data_dir, norm_fn=norm_fn,
            gap_filter=[g], limit=limit_per_gap,
        )
        if len(ds_g) == 0:
            continue
        dl = DataLoader(
            ds_g, batch_size=batch_size,
            num_workers=2, shuffle=False, drop_last=False,
        )

        loss_sum = 0.0
        top1_sum = 0
        pos_cos_sum = 0.0
        n_total = 0

        for v1, v2, _gap_batch in dl:
            v1, v2 = v1.to(device), v2.to(device)
            B = v1.shape[0]
            if B < 2:
                continue
            with torch.no_grad():
                c1 = encoder.forward(v1)
                c2 = encoder.forward(v2)
                z1 = c1[:, c1.shape[1] // 2, :]
                z2 = c2[:, c2.shape[1] // 2, :]
                z1n = torch.nn.functional.normalize(z1, dim=-1)
                z2n = torch.nn.functional.normalize(z2, dim=-1)
                sim = (z1n @ z2n.T) / temperature
                targets = torch.arange(B, device=device)
                ce = torch.nn.functional.cross_entropy(sim, targets, reduction='sum')
                top1 = (sim.argmax(dim=1) == targets).long().sum()
                pos_cos = (z1n * z2n).sum(dim=-1).sum()
            loss_sum += float(ce.item())
            top1_sum += int(top1.item())
            pos_cos_sum += float(pos_cos.item())
            n_total += B

        if n_total == 0:
            continue
        mean_loss = loss_sum / n_total
        mean_top1 = top1_sum / n_total
        mean_pos_cos = pos_cos_sum / n_total
        metrics[f'val_ssl/{name}_loss_gap_{g}'] = mean_loss
        metrics[f'val_ssl/{name}_top1_gap_{g}'] = mean_top1
        metrics[f'val_ssl/{name}_pos_cos_gap_{g}'] = mean_pos_cos
        pos_cos_at_gap_means.append((g, mean_pos_cos))

    if len(pos_cos_at_gap_means) >= 3:
        gs_t = torch.tensor([x[0] for x in pos_cos_at_gap_means], dtype=torch.float64)
        ms_t = torch.tensor([x[1] for x in pos_cos_at_gap_means], dtype=torch.float64)
        gs_ranks = gs_t.argsort().argsort().to(torch.float64)
        ms_ranks = ms_t.argsort().argsort().to(torch.float64)
        gs_norm = gs_ranks - gs_ranks.mean()
        ms_norm = ms_ranks - ms_ranks.mean()
        denom = (gs_norm.pow(2).sum().sqrt() * ms_norm.pow(2).sum().sqrt()).item()
        if denom > 0:
            metrics[f'val_ssl/{name}_spearman_cos_vs_gap'] = float(
                (gs_norm * ms_norm).sum().item() / denom
            )

    torch.cuda.empty_cache()
    return metrics


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT = {
        'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 4096,
        'batch_size': 64, 'total_steps': 300_000, 'ds_limit': 0,
        'max_lr': 3e-4, 'temperature': 0.1,
        'p_mask': 0.05, 'mask_size': 10, 'max_negatives': 8192,
        'weight_decay': 0.02, 'pct_start': 0.10,
        'grad_clip': 5.0,
        # Step-based cadences (matches ssl_55/56). At ctx=4096 a full
        # epoch through yoran is ~6k steps at d=128 bs=64; ckpt every
        # 10k steps means ~1.7 epochs between portable milestones.
        'probe_every_steps': 10000,
        'checkpoint_every_steps': 10000,
        'resume_every_steps': 10000,
        'chunk_size': 2048,
        'cnn_variant': 'default',
    }
    c = DEFAULT | config.get('smrt2vec', {})
    config['smrt2vec'] = c
    config['git_hash'] = get_git_revision_hash()

    resume_from = config.get('resume_from') or None
    if resume_from:
        resume_from = os.path.abspath(os.path.expandvars(str(resume_from)))
        if not os.path.isdir(resume_from):
            raise FileNotFoundError(
                f"resume_from={resume_from!r} is not a directory. It should point "
                f"to an Accelerate state directory (e.g. <exp>/checkpoints/latest)."
            )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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

    exp_type = config.get('experiment_type', 'ssl')
    exp_name = config.get('experiment_name', 'ssl_experiment')
    project_namespace = f"{exp_type}/{exp_name}"

    tracker = None
    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker('tensorboard')
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(config, f)
        tracker.writer.add_text('Full_Config', f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # --- Dataset & normalization ---
    dataset_name = config.get('ssl_dataset', 'yoran_raw.memmap')
    memmap_path = f"data/01_processed/ssl_sets/{dataset_name}"

    ds = ShardedMemmapDataset(memmap_path, limit=c['ds_limit'])
    if accelerator.is_main_process:
        print(f"SSL dataset: {len(ds)} reads from {memmap_path}")

    if resume_from:
        _check_resume_compatible(resume_from, config)
        norm_path = os.path.join(resume_from, 'norm_stats.pt')
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"resume_from is set but {norm_path} is missing. "
                f"The checkpoint was produced by an older training loop; "
                f"recompute will skew the data distribution."
            )
        ssl_norm = KineticsNorm.load_stats(torch.load(norm_path, map_location='cpu'))
        if accelerator.is_main_process:
            print(f"[resume] loaded norm stats from {norm_path}")
            print(f"SSL norm — means: {ssl_norm.means}, stds: {ssl_norm.stds}")
    else:
        ssl_norm = KineticsNorm(ds, max_samples=16_384)
        if accelerator.is_main_process:
            print(f"SSL norm — means: {ssl_norm.means}, stds: {ssl_norm.stds}")

    normed_ds = NormedDataset(ds, ssl_norm)

    # ChunkedRandomSampler: contiguous index ranges from a single shard
    # at a time; collapses random disk seeks into sequential cache-
    # resident access. Within-chunk shuffle preserves gradient noise.
    sampler = ChunkedRandomSampler(normed_ds, c['chunk_size'], shuffle_within=True)
    dl = DataLoader(
        normed_ds, batch_size=c['batch_size'], num_workers=4,
        pin_memory=True, prefetch_factor=4,
        shuffle=False, sampler=sampler,
        persistent_workers=True,
    )

    # --- Model / loss / optim ---
    # cnn_variant='default' (RF=107) | 'small_rf' (RF=27, 4x downsampling
    # preserved). The small_rf path swaps SmrtEncoder for SmrtEncoderSmallRF
    # via Smrt2VecInputMaskLNSmallRF. State-dicts are not interchangeable
    # across variants — fresh training only.
    model_cls = (Smrt2VecInputMaskLNSmallRF
                 if c['cnn_variant'] == 'small_rf'
                 else Smrt2VecInputMaskLN)
    model = model_cls(
        d_model=c['d_model'], n_layers=c['n_layers'], n_head=c['n_head'],
        max_len=c['context'],
        p_mask=float(c['p_mask']), mask_size=int(c['mask_size']),
    )

    if accelerator.is_main_process and tracker is not None:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"CNN receptive field: {model.encoder.cnn.r0} bases")
        tracker.writer.add_scalar('architecture/cnn_receptive_field', model.encoder.cnn.r0, 0)
        tracker.writer.add_scalar('architecture/param_count', n_params, 0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(c['max_lr']), weight_decay=float(c['weight_decay'])
    )
    criterion = AgInfoNCE(
        temperature=float(c['temperature']),
        max_negatives=int(c['max_negatives']) if c.get('max_negatives') else None,
    )

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    total_steps = int(c['total_steps'])
    # Do not call accelerator.prepare(scheduler): AcceleratedScheduler.step()
    # advances the wrapped LambdaLR by num_processes per call (default
    # split_batches=False path; verified against accelerate==1.13.0 source),
    # which compresses the schedule horizon by 8x with 8 GPUs. Same fix
    # supervised_40 applied for the ds_grid lineage (`docs/experiment_log.md`).
    # The raw LambdaLR is stepped once per global step in the training loop.
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps=total_steps, pct_start=c['pct_start'])

    progress_state = ProgressState()
    accelerator.register_for_checkpointing(progress_state)

    if resume_from:
        accelerator.load_state(resume_from)
        if accelerator.is_main_process:
            print(
                f"[resume] restored from {resume_from}: "
                f"epoch={progress_state.epoch}, global_step={progress_state.global_step}"
            )

    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(dl)}, Total steps: {total_steps}")
        print(f"Warmup steps: {int(total_steps * c['pct_start'])}")
        print(
            f"Probe every {c['probe_every_steps']} global steps, "
            f"milestone ckpt every {c['checkpoint_every_steps']} global steps, "
            f"resume ckpt every {c['resume_every_steps']} global steps"
        )

    checkpoint_dir = os.path.join(os.path.dirname(config_path), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_dir = os.path.join(checkpoint_dir, 'latest')

    global_step = progress_state.global_step
    probe_history = []
    nonfinite_skip_count = 0

    def _save_latest(epoch_idx, step_in_epoch_count, global_step_count):
        progress_state.epoch = epoch_idx
        progress_state.step_in_epoch = step_in_epoch_count
        progress_state.global_step = global_step_count
        accelerator.save_state(latest_dir)
        if accelerator.is_main_process:
            with open(os.path.join(latest_dir, 'run_metadata.yaml'), 'w') as f:
                yaml.dump(config, f)
            torch.save(ssl_norm.save_stats(), os.path.join(latest_dir, 'norm_stats.pt'))
            log(f"[resume] saved latest/ at global_step {global_step_count} "
                f"(epoch {epoch_idx+1}, step_in_epoch {step_in_epoch_count}/{len(dl)})")
        accelerator.wait_for_everyone()

    # Step-0 baseline eval: probe + pair val on the random-init encoder
    # (fresh runs only — `global_step == 0`). Anchors every trajectory at
    # a calibrated baseline rather than extrapolating back from step
    # `probe_every_steps`. The probe-head signal alone is non-trivially
    # above chance (a linear head on a random CNN+transformer can pick
    # up some class correlation), so the SSL gain is `step_N − step_0`,
    # not `step_N − 0.5`. Also disambiguates "collapsed at chance" from
    # "below random init" when probe falls during training.
    # Skipped on resume: the resumed checkpoint already has its in-run
    # measurements at `probe_every_steps` cadence.
    if global_step == 0 and config.get('probe_pos_train'):
        log("=== step 0 baseline eval (random-init encoder) ===")
        unwrapped = accelerator.unwrap_model(model)
        probe_top1, probe_auroc = linear_probe_eval(
            unwrapped.encoder, config.get('probe', {}),
            config, accelerator, ssl_norm,
        )
        if accelerator.is_main_process:
            probe_history.append((0, probe_top1, probe_auroc))
            accelerator.log({
                'probe_top1': probe_top1,
                'probe_auroc': probe_auroc,
            }, step=0)
            print(
                f"step 0 baseline: probe_top1={probe_top1:.4f}  "
                f"probe_auroc={probe_auroc:.4f}"
            )

        ssl_pair_val_paths = {}
        if config.get('ssl_pair_val_train'):
            ssl_pair_val_paths['train'] = config['ssl_pair_val_train']
        if config.get('ssl_pair_val_val'):
            ssl_pair_val_paths['val'] = config['ssl_pair_val_val']

        if ssl_pair_val_paths:
            combined_metrics = {}
            for split_name, split_path in ssl_pair_val_paths.items():
                m = ssl_pair_val_eval(
                    unwrapped.encoder, split_path,
                    accelerator, ssl_norm,
                    temperature=float(c['temperature']),
                    batch_size=int(c.get('val_pair_batch_size', 1024)),
                    limit_per_gap=int(c.get('val_pair_limit_per_gap', 10000)),
                    name=split_name,
                )
                combined_metrics.update(m)
            if accelerator.is_main_process:
                accelerator.log(combined_metrics, step=0)
                tr_top1 = combined_metrics.get(
                    'val_ssl/train_top1_gap_32', float('nan'))
                va_top1 = combined_metrics.get(
                    'val_ssl/val_top1_gap_32', float('nan'))
                print(
                    f"step 0 baseline: ssl_pair_val "
                    f"train_top1_gap_32={tr_top1:.3f} "
                    f"val_top1_gap_32={va_top1:.3f}"
                )

        accelerator.wait_for_everyone()
        model.train()

    # Step-cap outer loop: re-iterate the DataLoader as many epochs as
    # fit until global_step hits total_steps. Epoch counter is bookkeeping
    # for resume's skip_first_batches; the cosine schedule is keyed off
    # total_steps directly.
    epoch = progress_state.epoch
    done = global_step >= total_steps
    while not done:
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        skip_n = progress_state.step_in_epoch if epoch == progress_state.epoch else 0
        if skip_n > 0:
            effective_dl = skip_first_batches(dl, skip_n)
            log(f"=== epoch {epoch+1} resume at "
                f"step_in_epoch {skip_n}/{len(dl)}, global_step={global_step}/{total_steps} ===")
        else:
            effective_dl = dl
            log(f"=== epoch {epoch+1} start "
                f"({len(dl)} steps, global_step={global_step}/{total_steps}) ===")
        _prev_t = time.perf_counter()
        _window_start_t = _prev_t

        for step_offset, batch in enumerate(effective_dl):
            step_in_epoch_done = skip_n + step_offset + 1

            c_proj, targets, mask_idx = model(batch)
            loss = criterion(c_proj, targets, mask_idx)

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

            # Dimensional-collapse diagnostics on the projected per-position
            # output before AgInfoNCE's F.normalize. Flatten across batch
            # x positions so the std/norm are over the same population the
            # contrastive loss is operating on.
            with torch.no_grad():
                z_flat = c_proj.reshape(-1, c_proj.shape[-1])
                z_std_local = z_flat.std(dim=0).mean()
                z_norm_local = z_flat.norm(dim=1).mean()
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
                    f"({step_in_epoch_done}/{len(dl)}) | "
                    f"loss={current_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{window_its:.2f} it/s "
                    f"grad={grad_norm_r:.2f}")

            probe_every_steps = int(c['probe_every_steps'])
            checkpoint_every_steps = int(c['checkpoint_every_steps'])
            resume_every_steps = int(c['resume_every_steps'])
            tick_probe = (probe_every_steps > 0
                          and global_step % probe_every_steps == 0
                          and config.get('probe_pos_train'))
            tick_ckpt = (checkpoint_every_steps > 0
                         and global_step % checkpoint_every_steps == 0)
            tick_resume = (resume_every_steps > 0
                           and global_step % resume_every_steps == 0)

            if tick_probe or tick_ckpt:
                unwrapped = accelerator.unwrap_model(model)

                if tick_probe:
                    probe_top1, probe_auroc = linear_probe_eval(
                        unwrapped.encoder, config.get('probe', {}),
                        config, accelerator, ssl_norm,
                    )
                    if accelerator.is_main_process:
                        probe_history.append((global_step, probe_top1, probe_auroc))
                        accelerator.log({
                            'probe_top1': probe_top1,
                            'probe_auroc': probe_auroc,
                        }, step=global_step)
                        print(
                            f"step {global_step}: probe_top1={probe_top1:.4f}  "
                            f"probe_auroc={probe_auroc:.4f}  (recent loss={current_loss:.4f})"
                        )

                    # SSL pair val: encoder.forward + center-latent on
                    # the held-out PairedGapMemmapDataset. Two splits:
                    # 'train' = yoran (in-distribution), 'val' = ob007
                    # (held-out source). The pair-val task isn't the
                    # SSL training objective for this experiment (which
                    # is per-position masked prediction, not pair
                    # retrieval), but it remains a useful diagnostic of
                    # whether the encoder produces locally-coherent
                    # representations on the same molecule.
                    ssl_pair_val_paths = {}
                    if config.get('ssl_pair_val_train'):
                        ssl_pair_val_paths['train'] = config['ssl_pair_val_train']
                    if config.get('ssl_pair_val_val'):
                        ssl_pair_val_paths['val'] = config['ssl_pair_val_val']

                    if ssl_pair_val_paths:
                        combined_metrics = {}
                        for split_name, split_path in ssl_pair_val_paths.items():
                            m = ssl_pair_val_eval(
                                unwrapped.encoder, split_path,
                                accelerator, ssl_norm,
                                temperature=float(c['temperature']),
                                batch_size=int(c.get('val_pair_batch_size', 1024)),
                                limit_per_gap=int(c.get('val_pair_limit_per_gap', 10000)),
                                name=split_name,
                            )
                            combined_metrics.update(m)
                        if accelerator.is_main_process:
                            accelerator.log(combined_metrics, step=global_step)
                            tr_top1 = combined_metrics.get(
                                'val_ssl/train_top1_gap_32', float('nan'))
                            va_top1 = combined_metrics.get(
                                'val_ssl/val_top1_gap_32', float('nan'))
                            tr_rho = combined_metrics.get(
                                'val_ssl/train_spearman_cos_vs_gap', float('nan'))
                            va_rho = combined_metrics.get(
                                'val_ssl/val_spearman_cos_vs_gap', float('nan'))
                            print(
                                f"step {global_step}: ssl_pair_val "
                                f"train_top1_gap_32={tr_top1:.3f} "
                                f"val_top1_gap_32={va_top1:.3f} "
                                f"train_rho={tr_rho:.3f} val_rho={va_rho:.3f}"
                            )

                    accelerator.wait_for_everyone()
                    model.train()
                    _prev_t = time.perf_counter()
                    _window_start_t = _prev_t

                if tick_ckpt:
                    # Portable encoder-only milestone. Load recipe:
                    #   import torch
                    #   from smrt_foundation.model import SmrtEncoder
                    #   ckpt = torch.load('step_<N>.pt', map_location='cpu')
                    #   cfg = ckpt['config']['smrt2vec']
                    #   enc = SmrtEncoder(cfg['d_model'], cfg['n_layers'],
                    #                     cfg['n_head'], cfg['context'])
                    #   enc.load_state_dict(ckpt['encoder_state_dict'])
                    #   # Use ckpt['means'] / ckpt['stds'] with KineticsNorm
                    #   # so downstream normalization matches pre-training.
                    if accelerator.is_main_process:
                        save_path = os.path.join(checkpoint_dir, f'step_{global_step}.pt')
                        torch.save({
                            'encoder_state_dict': unwrapped.encoder.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'global_step': global_step,
                            'step_in_epoch': step_in_epoch_done,
                            **ssl_norm.save_stats(),
                        }, save_path)
                        print(f"Saved encoder checkpoint to {save_path}")
                    accelerator.wait_for_everyone()

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

    # --- Final checkpoint + pass-criterion summary ---
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'config': config,
            'epoch': epoch,
            'global_step': global_step,
            **ssl_norm.save_stats(),
        }, save_path)
        print(f"Saved final encoder checkpoint to {save_path}")

        if len(probe_history) >= 3:
            last = probe_history[-3:]
            print("Last 3 probe evals:", last)
            end_top1 = last[-1][1]
            non_dec = all(last[i][1] >= last[i - 1][1] - 0.005 for i in range(1, len(last)))
            passed = end_top1 >= 0.63 and non_dec
            print(f"Pass criterion: probe_top1 >= 0.63 AND non-decreasing -> {'PASS' if passed else 'FAIL'} (end {end_top1:.4f})")

    accelerator.end_training()


if __name__ == '__main__':
    main()
