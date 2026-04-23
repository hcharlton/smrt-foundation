"""Shared SimCLR v1 training loop for the Round-1 scoping grid.

Used by each size_*/train.py in this directory; identical in structure to
ssl_50_simclr_pilot/train.py. The only thing that varies between the
pilot and the four grid sizes is the config.yaml — specifically
simclr.d_model, simclr.n_layers, simclr.n_head, and simclr.batch_size.

Two views per read are drawn via AugmentationPolicy (random_subcrop +
kinetic noise/dropout/blur; revcomp off by default — see augment.py).
Encoder is the shared SmrtEncoder (RF=107, identical to exp 29) so a
passing checkpoint's encoder.state_dict() drops directly into
DirectClassifier for downstream fine-tuning.

Pass criterion for size_d128_L4 (mirrors the plan): probe_top1 >= 0.63
at end AND non-decreasing over the last 3 probe evaluations. 0.63 matches
ssl_26's contrastive-on-CpG result; falling below that reproduces a known
failure mode (declining probe) and indicates the augmentation policy is
destroying methylation signal.
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


LOG_EVERY = 100  # steps between stdout status lines

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import ShardedMemmapDataset, LabeledMemmapDataset, PairedViewDataset
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
    epoch counter and global_step are part of the Accelerate state
    directory and restore atomically alongside model / optimizer /
    scheduler / RNG state. `epoch` records the number of fully-completed
    epochs (so the training loop resumes at `range(progress_state.epoch,
    c['epochs'])`).
    """

    def __init__(self):
        self.epoch = 0
        self.global_step = 0

    def state_dict(self):
        return {'epoch': int(self.epoch), 'global_step': int(self.global_step)}

    def load_state_dict(self, sd):
        self.epoch = int(sd.get('epoch', 0))
        self.global_step = int(sd.get('global_step', 0))


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
        'epochs': 100, 'ds_limit': 0,
        'batch_size': 512, 'max_lr': 3e-4,
        'weight_decay': 1e-4, 'pct_start': 0.05,
        'checkpoint_every': 20, 'probe_every': 5,
        # How often to overwrite the `latest/` Accelerate state dir (in
        # epochs). This is the resume cadence — bounded wall-time loss on a
        # crashed/preempted run. Defaults to `checkpoint_every` when unset
        # (so current behaviour is preserved). Per-size grid configs set a
        # smaller number so the largest model never loses more than ~5 h.
        'resume_every': None,
    }
    c = DEFAULT | config.get('simclr', {})
    if c['resume_every'] is None:
        c['resume_every'] = c['checkpoint_every']
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

    dl = DataLoader(
        paired_ds, batch_size=c['batch_size'], num_workers=4,
        pin_memory=True, prefetch_factor=4, shuffle=True,
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
            f"Probe every {c['probe_every']} epochs, "
            f"milestone ckpt every {c['checkpoint_every']} epochs, "
            f"resume ckpt every {c['resume_every']} epochs"
        )

    checkpoint_dir = os.path.join(os.path.dirname(config_path), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_dir = os.path.join(checkpoint_dir, 'latest')

    global_step = progress_state.global_step
    probe_history = []

    for epoch in range(progress_state.epoch, c['epochs']):
        model.train()
        epoch_loss = 0.0
        log(f"=== epoch {epoch+1}/{c['epochs']} start "
            f"({len(dl)} steps, global_step={global_step}) ===")
        _prev_t = time.perf_counter()
        _window_start_t = _prev_t

        for step_in_epoch, (v1, v2) in enumerate(dl):
            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            accelerator.backward(loss)
            # Unclipped total grad L2 norm for instability diagnosis.
            # max_norm=inf = report only; swap in a finite value to clip.
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            optimizer.step()
            scheduler.step()

            global_step += 1
            current_loss = accelerator.reduce(loss, reduction='mean').item()
            epoch_loss += current_loss

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
                    f"({step_in_epoch+1}/{len(dl)}) | "
                    f"loss={current_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{window_its:.2f} it/s "
                    f"grad={grad_norm_r:.2f}")

        avg_loss = epoch_loss / max(len(dl), 1)
        accelerator.log({'epoch_avg_loss': avg_loss}, step=global_step)
        log(f"=== epoch {epoch+1}/{c['epochs']} end | avg_loss={avg_loss:.4f} ===")

        # --- Probe ---
        if (epoch + 1) % c['probe_every'] == 0 and config.get('probe_pos_train'):
            unwrapped = accelerator.unwrap_model(model)
            probe_top1, probe_auroc = linear_probe_eval(
                unwrapped.encoder, config.get('probe', {}), config, accelerator, ssl_norm
            )
            if accelerator.is_main_process:
                probe_history.append((epoch + 1, probe_top1, probe_auroc))
                accelerator.log({
                    'probe_top1': probe_top1,
                    'probe_auroc': probe_auroc,
                }, step=global_step)
                print(f"Epoch {epoch + 1}: ntxent_loss={avg_loss:.4f}  probe_top1={probe_top1:.4f}  probe_auroc={probe_auroc:.4f}")
            accelerator.wait_for_everyone()
            model.train()
        elif accelerator.is_main_process and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: ntxent_loss={avg_loss:.4f}")

        # Keep progress_state current *before* saving either checkpoint
        # kind, so a resume lands at the start of the next unfinished epoch.
        progress_state.epoch = epoch + 1
        progress_state.global_step = global_step

        # --- Resume checkpoint (Accelerate state dir; overwritten each tick) ---
        if (epoch + 1) % c['resume_every'] == 0:
            accelerator.save_state(latest_dir)
            if accelerator.is_main_process:
                with open(os.path.join(latest_dir, 'run_metadata.yaml'), 'w') as f:
                    yaml.dump(config, f)
                torch.save(ssl_norm.save_stats(), os.path.join(latest_dir, 'norm_stats.pt'))
                print(f"[resume] saved latest/ at epoch {epoch + 1}")
            accelerator.wait_for_everyone()

        # --- Milestone checkpoint (portable single-file for downstream use) ---
        if (epoch + 1) % c['checkpoint_every'] == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')
            torch.save({
                'model_state_dict': unwrapped.state_dict(),
                'encoder_state_dict': unwrapped.encoder.state_dict(),
                'config': config,
                'epoch': epoch + 1,
                **ssl_norm.save_stats(),
            }, save_path)
            print(f"Saved milestone checkpoint to {save_path}")
        accelerator.wait_for_everyone()

    # --- Final checkpoint + pass-criterion summary ---
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': unwrapped.state_dict(),
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'config': config,
            'epoch': c['epochs'],
            **ssl_norm.save_stats(),
        }, save_path)
        print(f"Saved final checkpoint to {save_path}")

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
