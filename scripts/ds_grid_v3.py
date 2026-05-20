"""
Treatment-aware fine-tune grid trainer (v3).

Built for the F-series fine-tune revamp (supervised_53_finetune_revamp/*).
Extends ds_grid_v2 with a per-arch `treatment` knob that selects:

  treatment = 'midlayer'      -> DirectClassifierMidLayer + per-arch layer_idx
  treatment = 'lpft_lldr'     -> DirectClassifier(SmallRF) + LP-FT two-stage
                                 with layer-wise LR decay on stage 2
  treatment = 'decoder_init'  -> DirectClassifierWithDecoder + ssl_58 decoder
                                 weight load from `decoder_state_dict`
  treatment = 'big_head'      -> DirectClassifierBigHead + supervised_20 recipe
  treatment = 'baseline'      -> v2 default behaviour (unchanged from v2)

Why a v3 instead of patching v2: ds_grid_v2 explicitly forbids `finetune:`
(two-stage gradual unfreeze) — see scripts/ds_grid_v2.py:104. F2 needs a
two-stage path. Cleanest fix is a sibling script that v2's stable
existing experiments don't depend on.

Init checkpoint resolution:
  - Literal path under `checkpoint:` works as in v2.
  - `checkpoint_template:` with `{arch}` placeholder works as in v2.
  - NEW: special value `'auto_best'` for either field resolves at runtime
    via scripts.utils.select_best_ssl_checkpoint. Requires the init spec
    to carry an `ssl_exp_dirs` dict mapping each arch_name to the SSL
    experiment directory whose probe_history.csv should be searched:

        ssl_58_best:
          checkpoint: 'auto_best'
          ssl_exp_dirs:
            d128_L4: 'scripts/experiments/ssl_58_autoencoder_grid/size_d128_L4_long'
            d512_L8: 'scripts/experiments/ssl_58_autoencoder_grid/size_d512_L8_long'
            ...

    This is the explicit replacement for an earlier implicit form that
    constructed `<ssl_exp_root>/size_<arch_name>/` automatically; the
    dict form makes it obvious which SSL run each arch is fine-tuning
    from (base vs _long vs _finished_cosine vs ssl_59_mae).

Usage (via thin per-experiment train.py wrapper):
  bash run.sh scripts/experiments/supervised_53_finetune_revamp/<treatment>/
"""

import sys
import os
import csv
import yaml
from dataclasses import dataclass, field
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
from smrt_foundation.model import (
    DirectClassifier, DirectClassifierSmallRF,
    DirectClassifierMidLayer, DirectClassifierWithDecoder,
    DirectClassifierBigHead,
)
from smrt_foundation.optim import get_cosine_schedule_with_warmup
from smrt_foundation.normalization import KineticsNorm

from scripts.ds_grid import (
    compute_step_budget,
    compute_batch_size,
    preload_to_gpu,
    build_metrics,
    evaluate,
    get_git_revision_hash,
)
from scripts.utils.select_best_ssl_checkpoint import select_best_ssl_checkpoint


REQUIRED_DATA_KEYS = ['pos_data_train', 'neg_data_train', 'pos_data_val', 'neg_data_val']
REQUIRED_TOPLEVEL = ['architectures', 'inits', 'train_sizes', 'classifier', 'scaling']

DEFAULT_CLASSIFIER = {
    'context': 32,
    'weight_decay': 0.02,
    'pct_start': 0.1,
    'cnn_variant': 'small_rf',     # ssl_58 / ssl_59 use small-RF; F-series defaults to small-RF.
    'treatment': 'baseline',       # v2-equivalent if not overridden.
    # F1 (midlayer): per-arch layer_idx in arch_override.
    # F2 (lpft_lldr): warmup_epochs (head-only) and lldr_decay live here.
    'warmup_epochs': 5,
    'lldr_decay': 0.7,
}

VALID_TREATMENTS = {'baseline', 'midlayer', 'lpft_lldr', 'decoder_init', 'big_head'}


# ---------------------------------------------------------------------------
# Combo (one work item: arch x init x train_size, with treatment metadata)
# ---------------------------------------------------------------------------

@dataclass
class Combo:
    arch_name: str
    init_name: str
    train_size: int
    arch_cfg: dict
    checkpoint: Optional[str]


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
        "v3 requires classifier.max_lr (single max LR; LLDR scales it per layer)."
    )
    config['classifier'] = DEFAULT_CLASSIFIER | config['classifier']
    treatment = config['classifier']['treatment']
    if treatment not in VALID_TREATMENTS:
        raise ValueError(
            f"Unknown treatment={treatment!r}. Valid: {sorted(VALID_TREATMENTS)}"
        )
    config['git_hash'] = get_git_revision_hash()
    return config


def _merge_arch_overrides(classifier_base, arch_override):
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


def _resolve_checkpoint(ckpt_spec: str, arch_name: str, ssl_exp_dir: Optional[str]) -> Optional[str]:
    """Resolve a checkpoint spec to an absolute path.

    Special values:
      - None -> random init.
      - 'auto_best' -> pick best-probe milestone for this arch from
        `ssl_exp_dir` (the explicit SSL experiment directory for this
        arch, looked up from `init_spec.ssl_exp_dirs[arch_name]`).
      - literal path or `{arch}`-template -> v2 behaviour.
    """
    if ckpt_spec is None:
        return None
    if ckpt_spec == 'auto_best':
        if not ssl_exp_dir:
            raise ValueError(
                f"checkpoint='auto_best' for arch={arch_name!r} requires "
                f"init_spec.ssl_exp_dirs[{arch_name!r}] to point at an SSL "
                f"experiment directory."
            )
        return select_best_ssl_checkpoint(ssl_exp_dir)
    if '{arch}' in ckpt_spec:
        return os.path.abspath(os.path.expandvars(ckpt_spec.format(arch=arch_name)))
    return os.path.abspath(os.path.expandvars(ckpt_spec))


def expand_combos(config):
    skip_pairs = {
        (s['arch'], s['init']) for s in config.get('skip', [])
    }
    classifier_base = config['classifier']
    train_sizes = list(config['train_sizes'])

    combos = []
    missing_checkpoints = []
    deferred_auto_best = []  # (arch, init) entries we couldn't resolve locally.

    for arch_name, arch_override in config['architectures'].items():
        arch_cfg = _merge_arch_overrides(classifier_base, arch_override)
        for init_name, init_spec in config['inits'].items():
            if (arch_name, init_name) in skip_pairs:
                continue

            ckpt_spec = init_spec.get('checkpoint')
            if ckpt_spec is None and init_spec.get('checkpoint_template') is not None:
                ckpt_spec = init_spec['checkpoint_template']

            ssl_exp_dirs = init_spec.get('ssl_exp_dirs') or {}
            ssl_exp_dir = ssl_exp_dirs.get(arch_name)

            if ckpt_spec == 'auto_best' and ssl_exp_dir and not os.path.isdir(ssl_exp_dir):
                # Cluster-side resolve at runtime; remember to validate later.
                deferred_auto_best.append((arch_name, init_name))
                checkpoint = 'auto_best'  # unresolved sentinel; resolved in worker.
                # Stash the explicit per-arch SSL dir onto arch_cfg so the
                # worker can resolve without re-reading the init spec.
                arch_cfg = {**arch_cfg, '_ssl_exp_dir': ssl_exp_dir}
            else:
                try:
                    checkpoint = _resolve_checkpoint(ckpt_spec, arch_name, ssl_exp_dir)
                except FileNotFoundError as e:
                    missing_checkpoints.append((arch_name, init_name, str(e)))
                    checkpoint = None

            if checkpoint is not None and checkpoint != 'auto_best':
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

    if missing_checkpoints and not os.environ.get('DS_GRID_DRY_RUN'):
        # auto_best deferred entries are OK — they resolve in the worker on the
        # cluster. Real missing-path failures still error out fast.
        if any(p != 'auto_best' for _, _, p in missing_checkpoints):
            lines = "\n".join(f"  {a} / {i}: {p}" for a, i, p in missing_checkpoints)
            raise FileNotFoundError(
                f"{len(missing_checkpoints)} checkpoint paths do not exist:\n{lines}"
            )

    combos.sort(key=lambda c: _combo_weight(c, config['scaling']), reverse=True)
    return combos


def _combo_weight(combo, scaling):
    c = combo.arch_cfg
    if 'bs_floor' in c and 'bs_k' in c:
        bs = compute_batch_size(combo.train_size, c['batch_size'], c['bs_floor'], c['bs_k'])
    else:
        bs = min(c['batch_size'], combo.train_size)
    steps = compute_step_budget(combo.train_size, bs, scaling)[0]
    return steps * (c['d_model'] ** 2) * c['n_layers']


# ---------------------------------------------------------------------------
# Treatment-aware model construction + checkpoint loading
# ---------------------------------------------------------------------------

def _build_model(arch_cfg):
    """Instantiate the right classifier class for the treatment."""
    treatment = arch_cfg.get('treatment', 'baseline')
    cnn_variant = arch_cfg.get('cnn_variant', 'small_rf')
    common = dict(
        d_model=arch_cfg['d_model'], n_layers=arch_cfg['n_layers'],
        n_head=arch_cfg['n_head'], max_len=arch_cfg['context'],
    )
    if treatment == 'midlayer':
        layer_idx = arch_cfg.get('layer_idx', arch_cfg['n_layers'] - 1)
        return DirectClassifierMidLayer(
            **common, layer_idx=layer_idx, cnn_variant=cnn_variant,
        )
    if treatment == 'decoder_init':
        return DirectClassifierWithDecoder(**common, cnn_variant=cnn_variant)
    if treatment == 'big_head':
        return DirectClassifierBigHead(**common, cnn_variant=cnn_variant)
    # 'baseline' and 'lpft_lldr' both use the standard DirectClassifier (or
    # SmallRF). The two-stage logic in 'lpft_lldr' is in the optimizer wiring,
    # not the model class.
    if cnn_variant == 'small_rf':
        return DirectClassifierSmallRF(**common)
    return DirectClassifier(**common)


def _load_pretrained(model, checkpoint_path, treatment):
    """Encoder weight load + (for decoder_init) decoder upsample weight load.

    Same PE-shape-mismatch tolerance as v1's load_pretrained_encoder. Returns
    the model unchanged if checkpoint_path is None.
    """
    if checkpoint_path is None:
        return model
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    enc_sd = ckpt['encoder_state_dict']
    enc_sd = {k: v for k, v in enc_sd.items()
              if not (k == 'pe.pe' and v.shape != model.encoder.pe.pe.shape)}
    missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
    print(f"  Loaded encoder from {checkpoint_path}")
    if missing:
        print(f"    Encoder missing (expected for PE): {missing[:4]}{'...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"    Encoder unexpected: {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}")

    if treatment == 'decoder_init':
        # ssl_58: 'decoder_state_dict' has keys like 'upsample.0.weight' (from
        # SmrtDecoder.upsample, which is the nn.Sequential at the top of
        # SmrtDecoder). DirectClassifierWithDecoder exposes the same module
        # under 'decoder_upsample.*', so we map the prefix.
        # ssl_59 (MAE): 'decoder_state_dict' bundles decoder_blocks/decoder_pe/
        # decoder_upsample/mask_token. We pick out 'decoder_upsample.upsample.*'
        # and remap to our 'decoder_upsample.*'.
        dec_sd = ckpt.get('decoder_state_dict', {})
        remapped = {}
        for k, v in dec_sd.items():
            if k.startswith('upsample.'):
                remapped[k.replace('upsample.', 'decoder_upsample.', 1)] = v
            elif k.startswith('decoder_upsample.upsample.'):
                remapped[k.replace('decoder_upsample.upsample.', 'decoder_upsample.', 1)] = v
        if remapped:
            mu, ux = model.load_state_dict(remapped, strict=False)
            print(f"  Loaded decoder upsample ({len(remapped)} tensors)")
            if ux:
                print(f"    Decoder unexpected: {ux[:4]}")
        else:
            print(f"  WARNING: decoder_init treatment but checkpoint has no usable decoder weights")
    return model


def _build_optimizer(model, arch_cfg, total_steps):
    """Single-stage AdamW for baseline/midlayer/decoder_init/big_head;
    LP-FT two-stage parameter groups for lpft_lldr.

    Returns (optimizer, scheduler, stage_boundary_step). For non-LPFT
    treatments stage_boundary_step is 0 (entire run is stage 2). For LPFT
    it's the step at which encoder unfreezes and LLDR kicks in.
    """
    treatment = arch_cfg.get('treatment', 'baseline')
    max_lr = float(arch_cfg['max_lr'])
    weight_decay = float(arch_cfg['weight_decay'])
    pct_start = float(arch_cfg['pct_start'])

    if treatment != 'lpft_lldr':
        opt = torch.optim.AdamW(
            model.parameters(), lr=max_lr, weight_decay=weight_decay,
        )
        sched = get_cosine_schedule_with_warmup(
            opt, total_steps=total_steps, pct_start=pct_start,
        )
        return opt, sched, 0

    # LP-FT: stage 1 trains the head only at max_lr.
    # stage 2 unfreezes the encoder with layer-wise LR decay.
    n_layers = int(arch_cfg['n_layers'])
    decay = float(arch_cfg.get('lldr_decay', 0.7))

    # LR per "layer group": index 0 = head (full max_lr); 1..n_layers = transformer
    # blocks top-down (block n_layers-1 = top = max_lr * decay^1, ..., block 0 =
    # bottom = max_lr * decay^n_layers); n_layers+1 = embed + cnn (deepest).
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith('head.') or n.startswith('head_ln.')]
    block_params = []
    for i in range(n_layers):
        block_params.append([
            p for n, p in model.named_parameters()
            if n.startswith(f'encoder.blocks.{i}.')
        ])
    base_params = [p for n, p in model.named_parameters()
                   if n.startswith('encoder.embed.')
                   or n.startswith('encoder.pe.')
                   or n.startswith('encoder.cnn.')
                   or n.startswith('encoder.layer_norm_target.')]

    param_groups = [{'params': head_params, 'lr': max_lr, 'name': 'head'}]
    for i in range(n_layers):
        depth = n_layers - i  # top layer (i=n_layers-1) gets depth=1
        param_groups.append({
            'params': block_params[i],
            'lr': max_lr * (decay ** depth),
            'name': f'block_{i}',
        })
    param_groups.append({
        'params': base_params,
        'lr': max_lr * (decay ** (n_layers + 1)),
        'name': 'base',
    })

    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    sched = get_cosine_schedule_with_warmup(
        opt, total_steps=total_steps, pct_start=pct_start,
    )
    # Stage boundary in steps: warmup_epochs * (steps_per_epoch). The worker
    # passes the step count; here we just return the *fraction* and let the
    # worker compute the absolute step.
    warmup_epochs = int(arch_cfg.get('warmup_epochs', 5))
    return opt, sched, warmup_epochs  # interpreted as epochs in worker


def _set_encoder_requires_grad(model, requires_grad: bool):
    for p in model.encoder.parameters():
        p.requires_grad = requires_grad


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def train_one_combo(rank, combo, config, experiment_dir, tb_dir):
    c = combo.arch_cfg
    treatment = c.get('treatment', 'baseline')
    has_pretrained = combo.checkpoint is not None
    has_batch_scaling = 'bs_floor' in c and 'bs_k' in c

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    combo_dir = os.path.join(
        experiment_dir, combo.init_name, combo.arch_name, f'n{combo.train_size}',
    )
    os.makedirs(combo_dir, exist_ok=True)

    tag = f"[GPU {rank} {combo.init_name}/{combo.arch_name}/n={combo.train_size:,}]"
    print(f"{tag} Starting | treatment={treatment}")

    # Resolve auto_best lazily on the cluster (where the checkpoints exist).
    if combo.checkpoint == 'auto_best':
        ssl_dir = c.get('_ssl_exp_dir')
        if not ssl_dir:
            raise RuntimeError(f"{tag} auto_best requested but _ssl_exp_dir is unset")
        combo.checkpoint = select_best_ssl_checkpoint(ssl_dir)
        print(f"{tag} Resolved auto_best -> {combo.checkpoint}")

    if combo.checkpoint:
        print(f"{tag} Pretrained: {combo.checkpoint}")
    else:
        print(f"{tag} Encoder: random init")

    set_seed(42)

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
    print(f"{tag} Preloading training data ({len(train_ds)} samples)...")
    train_x, train_y = preload_to_gpu(train_ds, device)
    n_train = len(train_x)
    val_x, val_y = preload_to_gpu(val_ds, device)
    print(f"{tag} Preloaded -- train: {train_x.shape}, val: {val_x.shape}")
    del train_ds, val_ds

    if has_batch_scaling:
        train_bs = compute_batch_size(n_train, c['batch_size'], c['bs_floor'], c['bs_k'])
    else:
        train_bs = min(c['batch_size'], n_train)

    total_steps, steps_per_epoch, eval_steps = compute_step_budget(
        n_train, train_bs, config['scaling'],
    )
    print(f"{tag} Train bs={train_bs}, steps/epoch={steps_per_epoch}, "
          f"total_steps={total_steps:,}, epochs={total_steps / max(steps_per_epoch, 1):.1f}")

    model = _build_model(c)
    if has_pretrained:
        _load_pretrained(model, combo.checkpoint, treatment)
    model = model.to(device)
    model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, RF={model.encoder.cnn.r0}")

    criterion = nn.BCEWithLogitsLoss()
    metrics = build_metrics(device)

    optimizer, scheduler, stage_boundary = _build_optimizer(model, c, total_steps)

    is_lpft = treatment == 'lpft_lldr'
    boundary_step = stage_boundary * steps_per_epoch if is_lpft else 0
    if is_lpft:
        print(f"{tag} LP-FT: stage1 head-only for {stage_boundary} epochs "
              f"({boundary_step} steps); stage2 unfreezes encoder with "
              f"LLDR decay={c.get('lldr_decay', 0.7)}")
        _set_encoder_requires_grad(model, False)

    tb_run_dir = os.path.join(
        tb_dir, combo.init_name, combo.arch_name, f'n{combo.train_size}',
    )
    writer = SummaryWriter(log_dir=tb_run_dir)
    writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
    writer.add_text("combo",
                    f"arch={combo.arch_name} init={combo.init_name} "
                    f"n={combo.train_size} treatment={treatment}", 0)
    writer.add_scalar("train_size", combo.train_size, 0)
    writer.add_scalar("train_bs", train_bs, 0)
    writer.add_scalar("max_lr", float(c['max_lr']), 0)

    csv_path = os.path.join(combo_dir, 'results.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'arch_name', 'init_name', 'treatment', 'train_size', 'train_bs',
        'max_lr', 'eval_point', 'step', 'stage',
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
    stage = 1
    best_val_acc = float('-inf')

    print(f"{tag} Eval schedule ({n_evals} points): {eval_steps[:5]}{'...' if n_evals > 5 else ''}")

    for step in range(1, total_steps + 1):
        # LP-FT stage transition.
        if is_lpft and stage == 1 and step > boundary_step:
            stage = 2
            _set_encoder_requires_grad(model, True)
            print(f"{tag} LP-FT stage2: encoder unfrozen at step {step}")

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
            writer.add_scalar("train/stage", stage, step)

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
                combo.arch_name, combo.init_name, treatment,
                combo.train_size, train_bs, float(c['max_lr']),
                eval_point, step, stage,
                f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}',
                f'{eval_results["f1"]:.6f}', f'{eval_results["auroc"]:.6f}',
                f'{eval_results["auprc"]:.6f}', f'{eval_results["accuracy"]:.6f}',
                f'{epochs_completed:.1f}',
            ])
            csv_file.flush()

            print(f"{tag} [{eval_point}/{n_evals}] step={step:,} "
                  f"loss={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                  f"acc={eval_results['accuracy']:.4f} "
                  f"auroc={eval_results['auroc']:.4f}")

            if eval_results['accuracy'] > best_val_acc:
                best_val_acc = eval_results['accuracy']
                unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save({
                    'model_state_dict': unwrapped.state_dict(),
                    'encoder_state_dict': unwrapped.encoder.state_dict(),
                    'config': config,
                    'arch_cfg': c,
                    'treatment': treatment,
                    'step': step,
                    'best_val_accuracy': best_val_acc,
                    **norm_fn.save_stats(),
                }, os.path.join(combo_dir, 'best_ckpt.pt'))

    csv_file.close()
    print(f"{tag} Done. {eval_point} checkpoints recorded.")
    writer.close()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def assign_combos(combos, world_size, scaling):
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


def merge_csvs_v3(experiment_dir, combos):
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
        print(f"WARNING: {len(missing)} per-combo CSVs missing during merge")
    if header is None:
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


def main(config_path):
    config = load_config(config_path)
    combos = expand_combos(config)
    experiment_dir = os.path.dirname(os.path.abspath(config_path))
    tb_dir = os.path.join(experiment_dir, 'training_logs')
    os.makedirs(tb_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    forced = os.environ.get('DS_GRID_FORCE_WORLD_SIZE')
    if forced and os.environ.get('DS_GRID_DRY_RUN'):
        num_gpus = int(forced)
    world_size = max(1, min(num_gpus, len(combos)))

    init_counts = {}
    arch_counts = {}
    for combo in combos:
        init_counts[combo.init_name] = init_counts.get(combo.init_name, 0) + 1
        arch_counts[combo.arch_name] = arch_counts.get(combo.arch_name, 0) + 1
    print(f"Experiment dir: {experiment_dir}")
    print(f"Treatment: {config['classifier']['treatment']}")
    print(f"Total combos: {len(combos)}")
    print(f"  by init: {init_counts}")
    print(f"  by arch: {arch_counts}")
    print(f"Train sizes: {config['train_sizes']}")
    print(f"GPUs: {num_gpus} available, {world_size} workers")

    assignments, workloads = assign_combos(combos, world_size, config['scaling'])
    if workloads:
        max_w, min_w = max(workloads), min(workloads)
        balance = (max_w - min_w) / max_w if max_w > 0 else 0.0
        print(f"Load balance: max/min weight {max_w:.2e}/{min_w:.2e} (spread {balance:.1%})")

    if os.environ.get('DS_GRID_DRY_RUN'):
        print("[DRY RUN] exiting before worker dispatch.")
        return

    if num_gpus == 0:
        raise RuntimeError("No CUDA devices visible. ds_grid_v3 requires GPUs.")

    if world_size > 1:
        mp.spawn(worker_fn, args=(assignments, config, experiment_dir, tb_dir),
                 nprocs=world_size)
    else:
        worker_fn(0, assignments, config, experiment_dir, tb_dir)

    merge_csvs_v3(experiment_dir, combos)
    print(f"TensorBoard logs in {tb_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
