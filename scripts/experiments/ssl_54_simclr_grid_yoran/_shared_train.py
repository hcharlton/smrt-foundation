"""Shared SimCLR v1 training loop for the Round-2 step-metric scoping grid.

Derived from ssl_52_simclr_grid_r1.1_gradclip/_shared_train.py with three
targeted round-2 changes:

  1. **Full epoch**: `ds_limit=0` in the per-size configs, so an "epoch"
     iterates all ~100M SSL reads (up from the 20M cap in rounds 1/1.1).

  2. **Step-based metrics**: probe and portable milestone checkpoints
     are now triggered every `probe_every_steps` / `checkpoint_every_steps`
     global steps respectively, not every N epochs. At the R2 data
     budget an epoch spans ~24k steps, so epoch-based cadences
     undersample the probe curve at the large end of the size grid —
     that's the overfit signal the prior round was missing. Resume
     state (the Accelerate `latest/` directory) stays epoch-based for
     simplicity: mid-epoch Accelerate resume via `skip_first_batches`
     is fiddly and gains little when ChunkedRandomSampler makes a full
     epoch restart cheap.

  3. **ChunkedRandomSampler**: the DataLoader uses
     `ChunkedRandomSampler(chunk_size=2048, shuffle_within=True)` in
     place of `shuffle=True`. The IO intervention sweep
     (profile_ssl51_io_interventions) showed this alone buys ~3.66× on
     sustained throughput at d128 (1.46 → 5.33 it/s) by converting
     random shard access into sequential reads from the same shard.
     Within-chunk shuffle preserves gradient noise; only cross-chunk
     order is pre-determined per epoch.

Pass criterion for size_d128_L4 is unchanged: probe_top1 >= 0.63 at
end AND non-decreasing over the last 3 probe evaluations.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs, skip_first_batches
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


LOG_EVERY = 100  # steps between stdout status lines

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import (
    ShardedMemmapDataset, LabeledMemmapDataset, PairedViewDataset,
    ChunkedRandomSampler,
)
from smrt_foundation.model import SimCLRSmrt
from smrt_foundation.loss import NTXent
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm, build_rc_lookup
from smrt_foundation.augment import AugmentationPolicy


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


class ProgressState:
    """Tracks training progress across interrupted runs.

    Registered with `accelerator.register_for_checkpointing(...)` so the
    counters are part of the Accelerate state directory and restore
    atomically alongside model / optimizer / scheduler / RNG state.

    Three fields:
      - `epoch`: index of the next epoch to run. When saving mid-epoch
        this holds the current (in-progress) epoch index; when saving
        at epoch end it holds `epoch + 1`.
      - `global_step`: total training steps completed across all epochs.
      - `step_in_epoch`: batches consumed from the start of the current
        epoch (0 at the start of a fresh epoch). Used to
        `skip_first_batches` on the DataLoader after a mid-epoch
        resume, so no already-processed batch is re-run.

    R2 reminder: ds_limit=0 means one epoch is ~24k steps (~1.3 h at
    ChunkedRandomSampler throughput). Step-based resume every
    `resume_every_steps` bounds crash-loss regardless of where in the
    epoch the crash lands — epoch-based resume at R1-size ds_limit
    (short epochs) was fine, but at R2 it would lose up to 1.3 h.
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

    Reads `{resume_dir}/run_metadata.yaml`, which the training loop writes
    alongside every `latest/` checkpoint. Returns the stored metadata dict.
    """
    sidecar = os.path.join(resume_dir, 'run_metadata.yaml')
    if not os.path.exists(sidecar):
        raise RuntimeError(
            f"Resume target {resume_dir} has no run_metadata.yaml sidecar; "
            f"refusing to resume (cannot verify architecture match)."
        )
    with open(sidecar, 'r') as f:
        stored = yaml.safe_load(f)
    cur = config.get('simclr', {})
    prev = stored.get('simclr', {})
    arch_keys = ('d_model', 'n_layers', 'n_head', 'context',
                 'projection_dim', 'projection_layers')
    for k in arch_keys:
        if cur.get(k) != prev.get(k):
            raise RuntimeError(
                f"Refusing to resume: simclr.{k} differs "
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

    Mirrors `ssl_26_cpg_contrastive/train.py:37-94` so probe_top1 is
    comparable across the SSL family. Uses the center-latent pooling
    convention consistent with `DirectClassifier`.
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


def build_policy(aug_cfg, data_config_path):
    """Instantiate AugmentationPolicy. Loads the data config lookup tables
    only when revcomp is enabled; otherwise `rc_lookup` stays None so the
    revcomp branch is unreachable.
    """
    rc_lookup = None
    if aug_cfg.get('revcomp_p', 0.0) > 0.0:
        with open(data_config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        rc_lookup = torch.as_tensor(build_rc_lookup(data_config), dtype=torch.long)
    return AugmentationPolicy(
        target_len=aug_cfg['target_len'],
        rc_lookup=rc_lookup,
        revcomp_p=aug_cfg.get('revcomp_p', 0.0),
        channel_dropout_p=aug_cfg.get('channel_dropout_p', 0.2),
        gaussian_noise_p=aug_cfg.get('gaussian_noise_p', 0.8),
        gaussian_noise_sigma=aug_cfg.get('gaussian_noise_sigma', 0.1),
        blur_p=aug_cfg.get('blur_p', 0.5),
        blur_sigma_range=tuple(aug_cfg.get('blur_sigma_range', (0.2, 2.0))),
    )


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT = {
        'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 32,
        'projection_dim': 128, 'projection_layers': 2,
        'temperature': 0.1,
        'epochs': 50, 'ds_limit': 0,
        'batch_size': 512, 'max_lr': 3e-4,
        'weight_decay': 1e-4, 'pct_start': 0.05,
        'grad_clip': float('inf'),
        # --- Round-2 step-based cadences ---
        # Probe, portable milestone checkpoint, and resume state are all
        # step-based. At R2's full-dataset scale one epoch is ~24k
        # steps (~1.3 h) — epoch-based resume would lose up to a full
        # epoch of work on a crash, which is worse than necessary.
        # Step-based resume with skip_first_batches picks back up at
        # the exact saved step on restart.
        'probe_every_steps': 10000,
        'checkpoint_every_steps': 10000,
        'resume_every_steps': 10000,
        # Sampler shard-chunk granularity. ChunkedRandomSampler yields
        # contiguous index ranges from a single shard; 2048 matches the
        # chunk size used in the IO-intervention sweep and lines up with
        # a typical shard's samples per load.
        'chunk_size': 2048,
    }
    c = DEFAULT | config.get('simclr', {})
    config['simclr'] = c
    config['git_hash'] = get_git_revision_hash()

    # Resolve and validate the resume target early so we fail fast (before
    # spinning up the dataset) if the user pointed at a missing or
    # incompatible checkpoint.
    resume_from = config.get('resume_from') or None
    if resume_from:
        resume_from = os.path.abspath(os.path.expandvars(str(resume_from)))
        if not os.path.isdir(resume_from):
            raise FileNotFoundError(
                f"resume_from={resume_from!r} is not a directory. It should point "
                f"to an Accelerate state directory (e.g. <exp>/checkpoints/latest)."
            )

    # Pass the active context through to the augmentation policy so the
    # target_len stays in one place (simclr.context) rather than duplicated
    # across sections.
    aug_cfg = dict(config.get('augment', {}))
    aug_cfg['target_len'] = c['context']

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

    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker('tensorboard')
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(config, f)
        tracker.writer.add_text('Full_Config', f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

    # --- Dataset & normalization ---
    dataset_name = config.get('ssl_dataset', 'ob007_raw.memmap')
    memmap_path = f"data/01_processed/ssl_sets/{dataset_name}"

    ds = ShardedMemmapDataset(memmap_path, limit=c['ds_limit'])
    if accelerator.is_main_process:
        print(f"SSL dataset: {len(ds)} reads from {memmap_path}")

    # On a fresh run we compute normalisation stats from the dataset; on
    # resume we load them from the checkpoint sidecar so the post-norm
    # data distribution is identical to pre-crash (the KineticsNorm
    # constructor draws a random subset of the dataset and would otherwise
    # produce a slightly different distribution on every restart).
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

    policy = build_policy(aug_cfg, config.get('data_config', 'configs/data.yaml'))
    paired_ds = PairedViewDataset(ds, policy=policy, norm_fn=ssl_norm)

    # Round-2 IO fix: ChunkedRandomSampler yields contiguous index ranges
    # from a single shard at a time. Workers read one shard then advance,
    # collapsing random seeks into sequential cache-resident access.
    # Within-chunk shuffle keeps gradient noise identical to shuffle=True;
    # only cross-chunk order is pre-determined per epoch.
    sampler = ChunkedRandomSampler(paired_ds, c['chunk_size'], shuffle_within=True)
    dl = DataLoader(
        paired_ds, batch_size=c['batch_size'], num_workers=4,
        pin_memory=True, prefetch_factor=4,
        shuffle=False, sampler=sampler,
        persistent_workers=True,
    )

    # --- Model / loss / optim ---
    model = SimCLRSmrt(
        d_model=c['d_model'], n_layers=c['n_layers'], n_head=c['n_head'],
        max_len=c['context'],
        projection_dim=c['projection_dim'], projection_layers=c['projection_layers'],
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"CNN receptive field: {model.encoder.cnn.r0} bases")
        tracker.writer.add_scalar('architecture/cnn_receptive_field', model.encoder.cnn.r0, 0)
        tracker.writer.add_scalar('architecture/param_count', n_params, 0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(c['max_lr']), weight_decay=float(c['weight_decay'])
    )
    criterion = NTXent(temperature=float(c['temperature']))

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    total_steps = len(dl) * c['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps=total_steps, pct_start=c['pct_start'])
    scheduler = accelerator.prepare(scheduler)

    # Register a tiny counter object so Accelerate includes our epoch /
    # global_step inside its save_state snapshot. `register_for_checkpointing`
    # must be called before `load_state`.
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

    def _save_latest(epoch_idx, step_in_epoch_count, global_step_count):
        """Write the Accelerate state dir + sidecars. Mutates progress_state
        so the saved snapshot reflects exactly (epoch_idx, step_in_epoch_count,
        global_step_count).
        """
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

    for epoch in range(progress_state.epoch, c['epochs']):
        model.train()
        # Use sum/count rather than sum/len(dl) so a resumed partial epoch
        # reports the correct mean over only the batches actually processed.
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # On resume into a mid-epoch save, skip_first_batches advances the
        # DataLoader past batches already processed before the crash. After
        # the first (resumed) epoch, progress_state.step_in_epoch is reset
        # to 0 so subsequent epochs iterate from the start.
        skip_n = progress_state.step_in_epoch if epoch == progress_state.epoch else 0
        if skip_n > 0:
            effective_dl = skip_first_batches(dl, skip_n)
            log(f"=== epoch {epoch+1}/{c['epochs']} resume at "
                f"step_in_epoch {skip_n}/{len(dl)}, global_step={global_step} ===")
        else:
            effective_dl = dl
            log(f"=== epoch {epoch+1}/{c['epochs']} start "
                f"({len(dl)} steps, global_step={global_step}) ===")
        _prev_t = time.perf_counter()
        _window_start_t = _prev_t

        for step_offset, (v1, v2) in enumerate(effective_dl):
            # Number of batches consumed from the start of this epoch
            # once the current step completes.
            step_in_epoch_done = skip_n + step_offset + 1

            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            accelerator.backward(loss)
            # Also reports pre-clip L2 norm for instability diagnosis.
            # simclr.grad_clip defaults to inf (no-op) when unset.
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=float(c['grad_clip']))
            optimizer.step()
            scheduler.step()

            global_step += 1
            current_loss = accelerator.reduce(loss, reduction='mean').item()
            epoch_loss_sum += current_loss
            epoch_loss_count += 1

            # Dimensional-collapse diagnostics on projection outputs before
            # F.normalize inside NTXent. `embed_z_std` is the mean over
            # channels of per-channel std across the batch — drops toward 0
            # when channels collapse. `embed_z_norm` tracks overall scale.
            with torch.no_grad():
                z_all = torch.cat([z1, z2], dim=0)
                z_std_local = z_all.std(dim=0).mean()
                z_norm_local = z_all.norm(dim=1).mean()
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
            }, step=global_step)

            if global_step % LOG_EVERY == 0:
                # Window-averaged it/s over the last LOG_EVERY steps.
                # `its` (single-step) goes to TB above for fine-grained analysis,
                # but instantaneous readings of one step in 100 are misleading
                # in stdout when the pipeline is bursty — so stdout shows the
                # true sustained rate instead.
                window_secs = _now - _window_start_t
                window_its = LOG_EVERY / window_secs if window_secs > 0 else 0.0
                _window_start_t = _now
                log(f"ep {epoch+1}/{c['epochs']} "
                    f"step {global_step}/{total_steps} "
                    f"({step_in_epoch_done}/{len(dl)}) | "
                    f"loss={current_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{window_its:.2f} it/s "
                    f"grad={grad_norm_r:.2f}")

            # --- Step-based triggers: decide once, unwrap once, act. ---
            # All three cadences are computed against the post-increment
            # global_step so an inner check like `global_step % 10000 == 0`
            # fires at steps 10000, 20000, ... — never at step 0.
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

            # Nest probe + ckpt under a single unwrap: both use
            # `unwrapped.encoder`, and the nested form makes the "either
            # condition implies unwrapped is bound" invariant obvious to
            # readers (and the type checker).
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
                    accelerator.wait_for_everyone()
                    model.train()
                    # Reset step-time timer so probe wall time doesn't
                    # pollute the next it/s reading.
                    _prev_t = time.perf_counter()
                    _window_start_t = _prev_t

                if tick_ckpt:
                    # Portable milestone: encoder-only. SimCLR discards
                    # the projection head for downstream use
                    # (Chen et al. 2020 §4.2), so the encoder is the
                    # only thing a fine-tuning or inference script
                    # should load from these files. Load recipe:
                    #
                    #   import torch
                    #   from smrt_foundation.model import SmrtEncoder
                    #   ckpt = torch.load('step_<N>.pt', map_location='cpu')
                    #   cfg = ckpt['config']['simclr']
                    #   enc = SmrtEncoder(cfg['d_model'], cfg['n_layers'],
                    #                     cfg['n_head'], cfg['context'])
                    #   enc.load_state_dict(ckpt['encoder_state_dict'])
                    #   # Use ckpt['means'] / ckpt['stds'] with
                    #   # KineticsNorm so downstream normalization
                    #   # matches pre-training.
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

        avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        accelerator.log({'epoch_avg_loss': avg_loss}, step=global_step)
        log(f"=== epoch {epoch+1}/{c['epochs']} end | "
            f"avg_loss={avg_loss:.4f} over {epoch_loss_count} batches ===")

        # Epoch fully complete. Clear the within-epoch counter so the
        # next iteration (and any future resume after this boundary)
        # starts from batch 0 of the next epoch.
        progress_state.epoch = epoch + 1
        progress_state.step_in_epoch = 0
        progress_state.global_step = global_step

    # --- Final checkpoint + pass-criterion summary ---
    # Same encoder-only format as the step milestones; see the load
    # recipe in the tick_ckpt block above.
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'config': config,
            'epoch': c['epochs'],
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
            print(f"Pass criterion: probe_top1 >= 0.63 AND non-decreasing → {'PASS' if passed else 'FAIL'} (end {end_top1:.4f})")

    accelerator.end_training()


if __name__ == '__main__':
    main()
