"""Dual-sample masked-autoencoder training loop for ssl_61.

Mirrors the ssl_58 autoencoder harness exactly (architecture, loss, probe,
pair-val, step-0 baseline, atomic Accelerate resume, encoder-only milestone
saves), with two changes that fall out of training on two source samples
instead of one:

1. **Per-source z-normalization.** Each source (e.g. DA1 Sequel II, yoran
   Revio) is wrapped in its own `KineticsNorm` sampled from its own data.
   That keeps the encoder's input distribution consistent in *each source's
   own z-space*, which is what a downstream classifier on that source
   would also do. The two normalized streams are then composed into a
   single ConcatDataset.

2. **Balanced 50/50 source mixing.** `BalancedChunkedSampler` round-robins
   chunks across sources at chunk granularity, oversampling the smaller
   source by re-shuffling its chunk list. Without this, `ConcatDataset`
   plus `ChunkedRandomSampler` would sample uniformly by index — yielding
   exposure proportional to dataset sizes, which collapses the
   experiment's "does mixing help?" signal when one source is much
   larger than the other.

The CpG probe data is Sequel II-derived, so the probe uses the Sequel II
source's norm (here keyed `da1`). Pair-val splits keep yoran's norm for
both yoran (in-dist) and ob007 (held-out) so the trajectory overlays
cleanly against ssl_58's diagnostics.

Pass criterion: probe_top1 >= ssl_58's d-size match, non-decreasing over
last 3 evals.
"""

import os
import sys
import time
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator, DistributedDataParallelKwargs, skip_first_batches
from accelerate.utils import set_seed
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


LOG_EVERY = 100

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import (
    ShardedMemmapDataset, LabeledMemmapDataset,
    PairedGapMemmapDataset, NormedDataset, BalancedChunkedSampler,
)
from smrt_foundation.model import SmrtAutoencoderSmallRF
from smrt_foundation.loss import MaskedReconstructionLoss
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm
from smrt_foundation.utils import (
    get_git_revision_hash, ProgressState, check_resume_compatible,
)


ARCH_KEYS = ('d_model', 'n_layers', 'n_head', 'context', 'p_mask', 'mask_size')

PROBE_NORM_SOURCE = 'da1'
PAIR_VAL_NORM_SOURCE = 'yoran'


def linear_probe_eval(encoder, probe_config, config, accelerator, norm_fn):
    """Freeze encoder, train a linear head on labeled CpG data, report accuracy.

    Center-latent pooling matches DirectClassifier convention and is shared
    across the SSL family. `norm_fn` is the source-matched normalizer for
    the probe data (CpG is Sequel II-derived; we pass the Sequel II source's
    KineticsNorm).
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
    """Per-gap SSL pair val metrics. Identical to ssl_58's version.

    `norm_fn` is the pair-val source's matched normalizer. For ssl_61 we
    use yoran's norm for both yoran (in-dist) and ob007 (held-out) so the
    trajectory matches ssl_58's pair-val curves visually.
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


def _resolve_source_path(config_path):
    """Map a config-declared memmap path to its staged location if the
    SMRT_SSL_MEMMAP_DIR env var is set (run.sh stages each source under
    that dir, preserving basenames). Otherwise return the config path.
    """
    staged_root = os.environ.get('SMRT_SSL_MEMMAP_DIR')
    if staged_root:
        return os.path.join(staged_root, os.path.basename(config_path))
    return config_path


def main():
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    DEFAULT = {
        'd_model': 128, 'n_layers': 4, 'n_head': 4, 'context': 512,
        'batch_size': 512, 'total_steps': 1_000_000, 'ds_limit': 0,
        'schedule_steps': 0,
        'max_lr': 3e-4,
        'p_mask': 0.15, 'mask_size': 10,
        'weight_decay': 0.02, 'pct_start': 0.03,
        'grad_clip': 5.0,
        'val_pair_temperature': 0.1,
        'probe_every_steps': 10000,
        'checkpoint_every_steps': 10000,
        'resume_every_steps': 10000,
        'chunk_size': 2048,
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
    csv_path = None
    train_csv_path = None
    pair_csv_path = None
    if accelerator.is_main_process:
        accelerator.init_trackers(project_namespace)
        tracker = accelerator.get_tracker('tensorboard')
        run_dir = tracker.writer.log_dir
        with open(os.path.join(run_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(config, f)
        tracker.writer.add_text('Full_Config', f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)

        import csv as _csv_mod
        csv_path = os.path.join(run_dir, 'probe_history.csv')
        with open(csv_path, 'w', newline='') as f:
            _csv_mod.writer(f).writerow([
                'step', 'probe_top1', 'probe_auroc',
                'val_ssl_train_top1_gap_32', 'val_ssl_val_top1_gap_32',
                'val_ssl_train_spearman_cos_vs_gap', 'val_ssl_val_spearman_cos_vs_gap',
            ])

        train_csv_path = os.path.join(run_dir, 'train_history.csv')
        with open(train_csv_path, 'w', newline='') as f:
            _csv_mod.writer(f).writerow([
                'step', 'train_loss', 'learning_rate', 'grad_norm',
                'embed_z_std', 'embed_z_norm',
                'step_time_ms', 'iters_per_sec', 'nonfinite_skip_count',
            ])

        pair_csv_path = os.path.join(run_dir, 'pair_val_history.csv')
        with open(pair_csv_path, 'w', newline='') as f:
            _csv_mod.writer(f).writerow([
                'step', 'split', 'gap', 'loss', 'top1', 'pos_cos', 'spearman_cos_vs_gap',
            ])

    ssl_datasets_cfg = config.get('ssl_datasets')
    if not ssl_datasets_cfg or not isinstance(ssl_datasets_cfg, dict):
        raise ValueError(
            "ssl_61 requires `ssl_datasets:` (dict of source_name -> "
            "memmap path) in config.yaml. Got: {!r}".format(ssl_datasets_cfg)
        )
    source_names = list(ssl_datasets_cfg.keys())
    log(f"SSL sources: {source_names}")

    per_source_ds = {}
    per_source_norm = {}
    per_source_normed = {}
    for src, raw_path in ssl_datasets_cfg.items():
        resolved = _resolve_source_path(raw_path)
        ds = ShardedMemmapDataset(resolved, limit=c['ds_limit'])
        per_source_ds[src] = ds
        if accelerator.is_main_process:
            print(f"  [{src}] {len(ds)} reads from {resolved}")

        if resume_from:
            stat_path = os.path.join(resume_from, f'norm_stats_{src}.pt')
            if not os.path.exists(stat_path):
                raise FileNotFoundError(
                    f"resume_from is set but {stat_path} is missing. "
                    f"Per-source norm stats must be persisted for resume."
                )
            per_source_norm[src] = KineticsNorm.load_stats(torch.load(stat_path, map_location='cpu'))
            if accelerator.is_main_process:
                print(
                    f"  [{src}] loaded norm from {stat_path} -- "
                    f"means: {per_source_norm[src].means}, stds: {per_source_norm[src].stds}"
                )
        else:
            per_source_norm[src] = KineticsNorm(ds, max_samples=16_384)
            if accelerator.is_main_process:
                print(
                    f"  [{src}] computed norm -- "
                    f"means: {per_source_norm[src].means}, stds: {per_source_norm[src].stds}"
                )

        per_source_normed[src] = NormedDataset(
            ds, per_source_norm[src], crop_len=int(c['context']),
        )

    if resume_from:
        check_resume_compatible(resume_from, config, ARCH_KEYS)

    combined_ds = ConcatDataset([per_source_normed[s] for s in source_names])
    source_lengths = [len(per_source_normed[s]) for s in source_names]

    sampler = BalancedChunkedSampler(
        source_lengths=source_lengths,
        chunk_size=c['chunk_size'],
        shuffle_within=True,
    )

    dl = DataLoader(
        combined_ds, batch_size=c['batch_size'], num_workers=6,
        pin_memory=True, prefetch_factor=8,
        shuffle=False, sampler=sampler,
        persistent_workers=True,
    )

    model = SmrtAutoencoderSmallRF(
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
    criterion = MaskedReconstructionLoss()

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    _diag = {}
    def _capture_c(module, _input, output):
        _diag['c'] = output
    accelerator.unwrap_model(model).encoder.register_forward_hook(_capture_c)

    total_steps = int(c['total_steps'])
    schedule_steps = int(c.get('schedule_steps') or 0) or total_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps=schedule_steps, pct_start=c['pct_start'])

    progress_state = ProgressState()
    accelerator.register_for_checkpointing(progress_state)

    if resume_from:
        accelerator.load_state(resume_from)
        steps_to_replay = progress_state.global_step - (scheduler.last_epoch + 1)
        for _ in range(max(0, steps_to_replay)):
            scheduler.step()
        if accelerator.is_main_process:
            print(
                f"[resume] restored from {resume_from}: "
                f"epoch={progress_state.epoch}, global_step={progress_state.global_step}, "
                f"scheduler.last_epoch={scheduler.last_epoch} (replayed {steps_to_replay} step()s), "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )

    if accelerator.is_main_process:
        print(f"Steps per epoch: {len(dl)}, Total steps: {total_steps}")
        print(f"Warmup steps: {int(schedule_steps * c['pct_start'])}")
        if schedule_steps != total_steps:
            print(
                f"Schedule steps: {schedule_steps} (cosine reaches min_lr at step "
                f"{schedule_steps}, then constant min_lr through step {total_steps})"
            )
        print(
            f"Probe every {c['probe_every_steps']} global steps "
            f"(probe uses '{PROBE_NORM_SOURCE}' norm; pair-val uses '{PAIR_VAL_NORM_SOURCE}' norm), "
            f"resume ckpt every {c['resume_every_steps']} global steps"
        )

    if PROBE_NORM_SOURCE not in per_source_norm:
        raise ValueError(
            f"PROBE_NORM_SOURCE='{PROBE_NORM_SOURCE}' but ssl_datasets has "
            f"keys {list(per_source_norm.keys())}. Add '{PROBE_NORM_SOURCE}' or "
            f"change PROBE_NORM_SOURCE at the top of _shared_train.py."
        )
    if PAIR_VAL_NORM_SOURCE not in per_source_norm:
        raise ValueError(
            f"PAIR_VAL_NORM_SOURCE='{PAIR_VAL_NORM_SOURCE}' but ssl_datasets has "
            f"keys {list(per_source_norm.keys())}."
        )
    probe_norm = per_source_norm[PROBE_NORM_SOURCE]
    pair_val_norm = per_source_norm[PAIR_VAL_NORM_SOURCE]

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
            for src, norm in per_source_norm.items():
                torch.save(norm.save_stats(), os.path.join(latest_dir, f'norm_stats_{src}.pt'))
            log(f"[resume] saved latest/ at global_step {global_step_count} "
                f"(epoch {epoch_idx+1}, step_in_epoch {step_in_epoch_count}/{len(dl)})")
        accelerator.wait_for_everyone()

    def _save_milestone(step, epoch_idx, step_in_epoch_count):
        """Portable milestone for downstream inference.

        Bundles encoder + decoder + per-source norm stats so downstream
        code can pick whichever source's norm matches its data
        distribution. Format:
            ckpt['norms'] = {source_name: {'norm_means', 'norm_stds',
                                           'norm_log_transform',
                                           'norm_n_continuous'}}
        """
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = os.path.join(checkpoint_dir, f'step_{step}.pt')
            torch.save({
                'encoder_state_dict': unwrapped.encoder.state_dict(),
                'decoder_state_dict': unwrapped.decoder.state_dict(),
                'mask_config': {
                    'p_mask': float(c['p_mask']),
                    'mask_size': int(c['mask_size']),
                },
                'config': config,
                'epoch': epoch_idx,
                'global_step': step,
                'step_in_epoch': step_in_epoch_count,
                'norms': {src: norm.save_stats() for src, norm in per_source_norm.items()},
            }, save_path)
            print(f"Saved encoder+decoder checkpoint to {save_path}")
        accelerator.wait_for_everyone()

    def _log_eval_csv(step, probe_top1, probe_auroc, pair_metrics):
        if not accelerator.is_main_process or csv_path is None:
            return
        import csv as _csv_mod
        with open(csv_path, 'a', newline='') as f:
            _csv_mod.writer(f).writerow([
                step, probe_top1, probe_auroc,
                pair_metrics.get('val_ssl/train_top1_gap_32', float('nan')),
                pair_metrics.get('val_ssl/val_top1_gap_32', float('nan')),
                pair_metrics.get('val_ssl/train_spearman_cos_vs_gap', float('nan')),
                pair_metrics.get('val_ssl/val_spearman_cos_vs_gap', float('nan')),
            ])

    def _log_pair_val_long(step, pair_metrics):
        if not accelerator.is_main_process or pair_csv_path is None or not pair_metrics:
            return
        import csv as _csv_mod
        grouped = {}
        spearmans = {}
        for k, v in pair_metrics.items():
            if not k.startswith('val_ssl/'):
                continue
            stem = k[len('val_ssl/'):]
            if stem.endswith('_spearman_cos_vs_gap'):
                split = stem[:-len('_spearman_cos_vs_gap')]
                spearmans[split] = v
                continue
            for metric_name in ('loss', 'top1', 'pos_cos'):
                marker = f'_{metric_name}_gap_'
                if marker in stem:
                    split, gap_str = stem.split(marker, 1)
                    try:
                        gap = int(gap_str)
                    except ValueError:
                        continue
                    grouped.setdefault((split, gap), {})[metric_name] = v
                    break
        with open(pair_csv_path, 'a', newline='') as f:
            w = _csv_mod.writer(f)
            for (split, gap), m in sorted(grouped.items()):
                w.writerow([
                    step, split, gap,
                    m.get('loss', float('nan')),
                    m.get('top1', float('nan')),
                    m.get('pos_cos', float('nan')),
                    '',
                ])
            for split, rho in sorted(spearmans.items()):
                w.writerow([step, split, '', '', '', '', rho])

    def _log_train_csv(step, train_loss, lr, grad_norm_v, z_std_v, z_norm_v,
                       step_ms, its, nf_skip):
        if not accelerator.is_main_process or train_csv_path is None:
            return
        import csv as _csv_mod
        with open(train_csv_path, 'a', newline='') as f:
            _csv_mod.writer(f).writerow([
                step, train_loss, lr, grad_norm_v, z_std_v, z_norm_v,
                step_ms, its, nf_skip,
            ])

    if global_step == 0 and config.get('probe_pos_train'):
        log("=== step 0 baseline eval (random-init encoder) ===")
        unwrapped = accelerator.unwrap_model(model)
        probe_top1, probe_auroc = linear_probe_eval(
            unwrapped.encoder, config.get('probe', {}),
            config, accelerator, probe_norm,
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

        combined_metrics = {}
        if ssl_pair_val_paths:
            for split_name, split_path in ssl_pair_val_paths.items():
                m = ssl_pair_val_eval(
                    unwrapped.encoder, split_path,
                    accelerator, pair_val_norm,
                    temperature=float(c['val_pair_temperature']),
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

        _log_eval_csv(0, probe_top1, probe_auroc, combined_metrics)
        _log_pair_val_long(0, combined_metrics)
        _save_milestone(0, 0, 0)

        accelerator.wait_for_everyone()
        model.train()

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

            kin_recon, kin_target, mask = model(batch)
            loss = criterion(kin_recon, kin_target, mask)

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
                _log_train_csv(
                    global_step, current_loss, scheduler.get_last_lr()[0],
                    grad_norm_r, z_std, z_norm,
                    step_ms, its, nonfinite_skip_count,
                )
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
            resume_every_steps = int(c['resume_every_steps'])
            tick_probe = (probe_every_steps > 0
                          and global_step % probe_every_steps == 0
                          and config.get('probe_pos_train'))
            tick_resume = (resume_every_steps > 0
                           and global_step % resume_every_steps == 0)

            if tick_probe:
                unwrapped = accelerator.unwrap_model(model)

                probe_top1, probe_auroc = linear_probe_eval(
                    unwrapped.encoder, config.get('probe', {}),
                    config, accelerator, probe_norm,
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

                ssl_pair_val_paths = {}
                if config.get('ssl_pair_val_train'):
                    ssl_pair_val_paths['train'] = config['ssl_pair_val_train']
                if config.get('ssl_pair_val_val'):
                    ssl_pair_val_paths['val'] = config['ssl_pair_val_val']

                combined_metrics = {}
                if ssl_pair_val_paths:
                    for split_name, split_path in ssl_pair_val_paths.items():
                        m = ssl_pair_val_eval(
                            unwrapped.encoder, split_path,
                            accelerator, pair_val_norm,
                            temperature=float(c['val_pair_temperature']),
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

                _log_eval_csv(global_step, probe_top1, probe_auroc, combined_metrics)
                _log_pair_val_long(global_step, combined_metrics)
                _save_milestone(global_step, epoch, step_in_epoch_done)

                accelerator.wait_for_everyone()
                model.train()
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

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save({
            'encoder_state_dict': unwrapped.encoder.state_dict(),
            'decoder_state_dict': unwrapped.decoder.state_dict(),
            'mask_config': {
                'p_mask': float(c['p_mask']),
                'mask_size': int(c['mask_size']),
            },
            'config': config,
            'epoch': epoch,
            'global_step': global_step,
            'norms': {src: norm.save_stats() for src, norm in per_source_norm.items()},
        }, save_path)
        print(f"Saved final encoder+decoder checkpoint to {save_path}")

        if len(probe_history) >= 3:
            last = probe_history[-3:]
            print("Last 3 probe evals:", last)
            end_top1 = last[-1][1]
            non_dec = all(last[i][1] >= last[i - 1][1] - 0.005 for i in range(1, len(last)))
            passed = end_top1 >= 0.67 and non_dec
            print(f"Pass criterion: probe_top1 >= 0.67 AND non-decreasing -> {'PASS' if passed else 'FAIL'} (end {end_top1:.4f})")

    accelerator.end_training()


if __name__ == '__main__':
    main()
