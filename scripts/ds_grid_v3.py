"""
Treatment-aware fine-tune grid trainer (v3).

Built for the F-series fine-tune revamp (supervised_53_finetune_revamp/*) and
the d=768 LR x recipe ablations (supervised_54 / supervised_55). A per-arch
`treatment` knob selects the classifier head, the encoder-adaptation mode, and
the optimizer strategy (see the TREATMENTS registry below):

  treatment = 'midlayer'      -> DirectClassifierMidLayer + per-arch layer_idx
  treatment = 'lpft_lldr'     -> DirectClassifier(SmallRF) + LP-FT two-stage
                                 with layer-wise LR decay on stage 2
  treatment = 'decoder_init'  -> DirectClassifierWithDecoder + ssl_58 decoder
                                 weight load from `decoder_state_dict`
  treatment = 'big_head'      -> DirectClassifierBigHead + supervised_20 recipe
  treatment = 'linear_probe'  -> encoder frozen the whole run, head-only opt
  treatment = 'baseline'      -> v2 default behaviour (unchanged from v2)

Why a v3 instead of patching v2: ds_grid_v2 explicitly forbids `finetune:`
(two-stage gradual unfreeze) — see scripts/ds_grid_v2.py:104. F2 needs a
two-stage path. Cleanest fix is a sibling script that v2's stable
existing experiments don't depend on.

Optimizer LR customisation (non-LLDR treatments):
  - `classifier.head_lr` / `classifier.encoder_lr` — when both are set and
    differ, build two AdamW param groups (head, encoder) under one cosine
    schedule. When they match (or are unset) the single-group v2 path is used.

Regularisation:
  - `classifier.label_smoothing` (default 0.0) — BCE targets are smoothed
    toward 0.5 in the training loop; 0.0 reproduces the un-smoothed behaviour.

Init checkpoint resolution:
  - Literal path under `checkpoint:` works as in v2.
  - `checkpoint_template:` with `{arch}` placeholder works as in v2.
  - Special value `'auto_best'` for either field resolves at runtime via
    scripts.utils.select_best_ssl_checkpoint. Requires the init spec to carry
    an `ssl_exp_dirs` dict mapping each arch_name to the SSL experiment
    directory whose probe_history.csv should be searched:

        ssl_58_best:
          checkpoint: 'auto_best'
          ssl_exp_dirs:
            d128_L4: 'scripts/experiments/ssl_58_autoencoder_grid/size_d128_L4_long'
            d512_L8: 'scripts/experiments/ssl_58_autoencoder_grid/size_d512_L8_long'
            ...

Usage (via thin per-experiment train.py wrapper):
  bash run.sh scripts/experiments/supervised_53_finetune_revamp/<treatment>/
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
    # supervised_54 decoupled-LR knobs. Both default to None, which means
    # "use max_lr for the whole model" (v2-equivalent). Set both to a number
    # to drive the head and encoder at different uniform LRs without invoking
    # lpft_lldr's per-layer decay.
    'head_lr': None,
    'encoder_lr': None,
    # supervised_55 regularisation knob. 0.0 reproduces un-smoothed BCE; a
    # positive value smooths targets toward 0.5 in the training loop.
    'label_smoothing': 0.0,
}


# ---------------------------------------------------------------------------
# Treatment registry
# ---------------------------------------------------------------------------
#
# Each treatment maps to (a) how the encoder is adapted and (b) which optimizer
# strategy builds the param groups. The classifier *class* is chosen in
# `_build_model` (it also depends on cnn_variant for the shared classes), but
# everything else about a treatment lives here so adding/altering one is a
# single-table edit rather than five scattered branches.

@dataclass(frozen=True)
class TreatmentSpec:
    encoder_mode: str        # 'full' | 'frozen' | 'staged'
    optimizer_strategy: str  # 'standard' | 'linear_probe' | 'lldr'


TREATMENTS = {
    'baseline':     TreatmentSpec(encoder_mode='full',   optimizer_strategy='standard'),
    'midlayer':     TreatmentSpec(encoder_mode='full',   optimizer_strategy='standard'),
    'decoder_init': TreatmentSpec(encoder_mode='full',   optimizer_strategy='standard'),
    'big_head':     TreatmentSpec(encoder_mode='full',   optimizer_strategy='standard'),
    'lpft_lldr':    TreatmentSpec(encoder_mode='staged',  optimizer_strategy='lldr'),
    'linear_probe': TreatmentSpec(encoder_mode='frozen',  optimizer_strategy='linear_probe'),
}

VALID_TREATMENTS = set(TREATMENTS)


# ---------------------------------------------------------------------------
# Combo + compute plan (one work item: arch x init x train_size)
# ---------------------------------------------------------------------------

@dataclass
class Combo:
    arch_name: str
    init_name: str
    train_size: int
    arch_cfg: dict
    checkpoint: Optional[str]
    # Set only when `checkpoint == 'auto_best'` could not be resolved locally
    # (the SSL dir doesn't exist on the submitting host); the worker resolves
    # it on the cluster. Replaces v2's `arch_cfg['_ssl_exp_dir']` side-channel.
    ssl_exp_dir: Optional[str] = None


@dataclass
class ComputePlan:
    train_bs: int
    total_steps: int
    steps_per_epoch: int
    eval_steps: list


def plan_compute(arch_cfg, n, scaling):
    """Resolve batch size + step budget for `n` training examples.

    Centralises the batch-size / step-budget arithmetic that was previously
    duplicated in `_combo_weight` (sort key) and the worker. `n` is the
    requested train_size when planning/weighting and the *actual* loaded count
    in the worker (they can differ when the dataset is smaller than requested).
    """
    if 'bs_floor' in arch_cfg and 'bs_k' in arch_cfg:
        bs = compute_batch_size(n, arch_cfg['batch_size'], arch_cfg['bs_floor'], arch_cfg['bs_k'])
    else:
        bs = min(arch_cfg['batch_size'], n)
    total_steps, steps_per_epoch, eval_steps = compute_step_budget(n, bs, scaling)
    return ComputePlan(train_bs=bs, total_steps=total_steps,
                       steps_per_epoch=steps_per_epoch, eval_steps=eval_steps)


def _combo_weight(combo, scaling):
    plan = plan_compute(combo.arch_cfg, combo.train_size, scaling)
    c = combo.arch_cfg
    return plan.total_steps * (c['d_model'] ** 2) * c['n_layers']


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


def _resolve_checkpoint(ckpt_spec: Optional[str], arch_name: str, ssl_exp_dir: Optional[str]) -> Optional[str]:
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
            deferred_ssl_dir = None

            if ckpt_spec == 'auto_best' and ssl_exp_dir and not os.path.isdir(ssl_exp_dir):
                # Cluster-side resolve at runtime; the worker resolves the
                # unresolved sentinel via combo.ssl_exp_dir.
                checkpoint = 'auto_best'
                deferred_ssl_dir = ssl_exp_dir
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
                    ssl_exp_dir=deferred_ssl_dir,
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
    # 'baseline', 'lpft_lldr', and 'linear_probe' all use the standard
    # DirectClassifier (or SmallRF). The two-stage logic in 'lpft_lldr' and the
    # permanent encoder freeze in 'linear_probe' live in the optimizer wiring
    # and the training loop, not the model class.
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


# ---------------------------------------------------------------------------
# Optimizer strategies
# ---------------------------------------------------------------------------

def _is_head_param(name: str) -> bool:
    """Match parameters that belong to the classification head.

    The DirectClassifier* family stores the head module as ``head`` (a
    Sequential) and, in `DirectClassifierMidLayer`, an additional pre-head
    LayerNorm at ``head_ln``. Both are part of the classifier and stay
    trainable under `linear_probe`. Anything else (encoder.*, decoder_upsample.*
    in F3) counts as encoder-side and gets frozen / picks up `encoder_lr`.
    """
    return name.startswith('head.') or name.startswith('head_ln.')


def _opt_linear_probe(model, head_lr, weight_decay, total_steps, pct_start):
    """Head-only optimizer; the encoder is frozen in the worker and stays so."""
    head_params = [p for n, p in model.named_parameters() if _is_head_param(n)]
    opt = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, total_steps=total_steps, pct_start=pct_start)
    return opt, sched, 0


def _opt_standard(model, max_lr, head_lr, encoder_lr, weight_decay, total_steps, pct_start):
    """Single-group AdamW, or two groups when head_lr != encoder_lr.

    Decoupled groups share one cosine schedule; cosine scaling is per-group on
    each group's initial_lr, so head and encoder keep their own LRs through
    warmup + decay. When the LRs match, the single-group v2 path (lr=max_lr) is
    preserved verbatim.
    """
    if head_lr != encoder_lr:
        head_params = [p for n, p in model.named_parameters() if _is_head_param(n)]
        enc_params = [p for n, p in model.named_parameters() if not _is_head_param(n)]
        opt = torch.optim.AdamW(
            [
                {'params': head_params, 'lr': head_lr, 'name': 'head'},
                {'params': enc_params,  'lr': encoder_lr, 'name': 'encoder'},
            ],
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=max_lr, weight_decay=weight_decay,
        )
    sched = get_cosine_schedule_with_warmup(opt, total_steps=total_steps, pct_start=pct_start)
    return opt, sched, 0


def _opt_lldr(model, arch_cfg, max_lr, weight_decay, total_steps, pct_start):
    """LP-FT ladder: head at full LR, transformer blocks decayed top-down,
    embed/cnn deepest. Returns warmup_epochs (head-only stage length) as the
    third element; the worker converts it to an absolute step boundary."""
    n_layers = int(arch_cfg['n_layers'])
    decay = float(arch_cfg.get('lldr_decay', 0.7))

    # LR per "layer group": index 0 = head (full max_lr); 1..n_layers = transformer
    # blocks top-down (block n_layers-1 = top = max_lr * decay^1, ..., block 0 =
    # bottom = max_lr * decay^n_layers); n_layers+1 = embed + cnn (deepest).
    head_params = [p for n, p in model.named_parameters() if _is_head_param(n)]
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
    sched = get_cosine_schedule_with_warmup(opt, total_steps=total_steps, pct_start=pct_start)
    warmup_epochs = int(arch_cfg.get('warmup_epochs', 5))
    return opt, sched, warmup_epochs  # interpreted as epochs in worker


def _build_optimizer(model, arch_cfg, total_steps):
    """Dispatch to the optimizer strategy for this treatment.

    Returns (optimizer, scheduler, stage_boundary). `stage_boundary` is the
    head-only warmup length in *epochs* for lpft_lldr; 0 for every other
    strategy (the whole run is one stage).
    """
    treatment = arch_cfg.get('treatment', 'baseline')
    spec = TREATMENTS.get(treatment)
    strategy = spec.optimizer_strategy if spec else 'standard'

    max_lr = float(arch_cfg['max_lr'])
    weight_decay = float(arch_cfg['weight_decay'])
    pct_start = float(arch_cfg['pct_start'])
    head_lr_raw = arch_cfg.get('head_lr')
    encoder_lr_raw = arch_cfg.get('encoder_lr')
    head_lr = float(head_lr_raw) if head_lr_raw is not None else max_lr
    encoder_lr = float(encoder_lr_raw) if encoder_lr_raw is not None else max_lr

    if strategy == 'linear_probe':
        return _opt_linear_probe(model, head_lr, weight_decay, total_steps, pct_start)
    if strategy == 'lldr':
        return _opt_lldr(model, arch_cfg, max_lr, weight_decay, total_steps, pct_start)
    return _opt_standard(model, max_lr, head_lr, encoder_lr, weight_decay, total_steps, pct_start)


def _set_encoder_requires_grad(model, requires_grad: bool):
    for p in model.encoder.parameters():
        p.requires_grad = requires_grad


def _smooth_targets(target, label_smoothing):
    """Smooth binary targets toward 0.5: {0,1} -> {ls/2, 1-ls/2}.

    `label_smoothing <= 0` is the identity (un-smoothed BCE), preserving the
    pre-supervised_55 behaviour for configs that don't set the knob.
    """
    if label_smoothing <= 0.0:
        return target
    return target * (1.0 - label_smoothing) + 0.5 * label_smoothing


# ---------------------------------------------------------------------------
# Per-run logging (TensorBoard + per-combo results.csv)
# ---------------------------------------------------------------------------

CSV_HEADER = [
    'arch_name', 'init_name', 'treatment', 'train_size', 'train_bs',
    'max_lr', 'head_lr', 'encoder_lr', 'label_smoothing',
    'eval_point', 'step', 'stage',
    'train_loss', 'val_loss',
    'val_f1', 'val_auroc', 'val_auprc', 'val_accuracy',
    'epochs_completed',
]


class RunLogger:
    """Wraps the SummaryWriter + per-combo results.csv so the training loop
    reads as logic, not boilerplate. Logging only — checkpoint saving lives in
    the worker."""

    def __init__(self, tb_run_dir, combo_dir, config, combo, train_bs,
                 max_lr, head_lr, encoder_lr, label_smoothing, treatment):
        self.combo = combo
        self.treatment = treatment
        self.train_bs = train_bs
        self.max_lr = max_lr
        self.head_lr = head_lr
        self.encoder_lr = encoder_lr
        self.label_smoothing = label_smoothing

        self.writer = SummaryWriter(log_dir=tb_run_dir)
        self.writer.add_text("config", f"```yaml\n{yaml.dump(config, indent=2)}\n```", 0)
        self.writer.add_text(
            "combo",
            f"arch={combo.arch_name} init={combo.init_name} "
            f"n={combo.train_size} treatment={treatment}", 0,
        )
        self.writer.add_scalar("train_size", combo.train_size, 0)
        self.writer.add_scalar("train_bs", train_bs, 0)
        self.writer.add_scalar("max_lr", max_lr, 0)
        self.writer.add_scalar("head_lr", head_lr, 0)
        self.writer.add_scalar("encoder_lr", encoder_lr, 0)
        self.writer.add_scalar("label_smoothing", label_smoothing, 0)

        self._csv_file = open(os.path.join(combo_dir, 'results.csv'), 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_HEADER)
        self._csv_file.flush()

    def log_train(self, step, loss, lr, stage):
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/lr", lr, step)
        self.writer.add_scalar("train/stage", stage, step)

    def log_eval(self, eval_point, n_evals, step, stage, avg_train_loss,
                 avg_val_loss, eval_results, epochs_completed, tag):
        self.writer.add_scalar("train/interval_avg_loss", avg_train_loss, step)
        self.writer.add_scalar("eval/loss", avg_val_loss, step)
        for name, val in eval_results.items():
            self.writer.add_scalar(f"eval/{name}", val, step)

        self._csv_writer.writerow([
            self.combo.arch_name, self.combo.init_name, self.treatment,
            self.combo.train_size, self.train_bs, self.max_lr,
            self.head_lr, self.encoder_lr, self.label_smoothing,
            eval_point, step, stage,
            f'{avg_train_loss:.6f}', f'{avg_val_loss:.6f}',
            f'{eval_results["f1"]:.6f}', f'{eval_results["auroc"]:.6f}',
            f'{eval_results["auprc"]:.6f}', f'{eval_results["accuracy"]:.6f}',
            f'{epochs_completed:.1f}',
        ])
        self._csv_file.flush()

        print(f"{tag} [{eval_point}/{n_evals}] step={step:,} "
              f"loss={avg_train_loss:.4f} val={avg_val_loss:.4f} "
              f"acc={eval_results['accuracy']:.4f} "
              f"auroc={eval_results['auroc']:.4f}")

    def close(self):
        self._csv_file.close()
        self.writer.close()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _resolve_runtime_checkpoint(combo, tag):
    """Resolve a deferred `auto_best` checkpoint on the cluster (where the SSL
    dirs exist). No-op for already-resolved checkpoints."""
    if combo.checkpoint == 'auto_best':
        if not combo.ssl_exp_dir:
            raise RuntimeError(f"{tag} auto_best requested but ssl_exp_dir is unset")
        combo.checkpoint = select_best_ssl_checkpoint(combo.ssl_exp_dir)
        print(f"{tag} Resolved auto_best -> {combo.checkpoint}")


def _prepare_data(combo, config, device, tag):
    """Compute the train-derived norm, then preload balanced train/val to GPU.

    Returns (train_x, train_y, val_x, val_y). The norm_fn is returned too so
    the best-checkpoint save can bundle its stats."""
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
    val_x, val_y = preload_to_gpu(val_ds, device)
    print(f"{tag} Preloaded -- train: {train_x.shape}, val: {val_x.shape}")
    del train_ds, val_ds
    return train_x, train_y, val_x, val_y, norm_fn


def _run_training_loop(model, optimizer, scheduler, criterion, metrics,
                       train_x, train_y, val_x, val_y, plan, c, treatment,
                       boundary_step, logger, combo_dir, norm_fn, config, tag):
    """The step loop: sample-with-replacement batches, eval on a geomspace
    schedule, save the best-val_acc checkpoint. Returns the number of evals."""
    encoder_mode = TREATMENTS.get(treatment, TREATMENTS['baseline']).encoder_mode
    train_bs = plan.train_bs
    total_steps = plan.total_steps
    steps_per_epoch = plan.steps_per_epoch
    eval_steps_set = set(plan.eval_steps)
    n_evals = len(plan.eval_steps)
    eval_bs = c['batch_size']
    label_smoothing = float(c.get('label_smoothing', 0.0))
    n_train = len(train_x)

    interval_loss = 0.0
    steps_since_last_eval = 0
    eval_point = 0
    stage = 1
    best_val_acc = float('-inf')

    print(f"{tag} Eval schedule ({n_evals} points): "
          f"{plan.eval_steps[:5]}{'...' if n_evals > 5 else ''}")

    for step in range(1, total_steps + 1):
        # LP-FT stage transition (staged treatments only).
        if encoder_mode == 'staged' and stage == 1 and step > boundary_step:
            stage = 2
            _set_encoder_requires_grad(model, True)
            print(f"{tag} LP-FT stage2: encoder unfrozen at step {step}")

        model.train()
        idx = torch.randint(0, n_train, (train_bs,), device=train_x.device)
        x = train_x[idx]
        y = train_y[idx]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            target = _smooth_targets(y.unsqueeze(1), label_smoothing)
            loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        interval_loss += loss.item()
        steps_since_last_eval += 1

        if step % 100 == 0:
            logger.log_train(step, loss.item(), scheduler.get_last_lr()[0], stage)

        if step in eval_steps_set:
            eval_point += 1
            avg_train_loss = interval_loss / steps_since_last_eval
            interval_loss = 0.0
            steps_since_last_eval = 0

            eval_results, avg_val_loss = evaluate(
                model, val_x, val_y, eval_bs, metrics, criterion,
            )
            epochs_completed = step / max(steps_per_epoch, 1)
            logger.log_eval(eval_point, n_evals, step, stage, avg_train_loss,
                            avg_val_loss, eval_results, epochs_completed, tag)

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

    return eval_point


def train_one_combo(rank, combo, config, experiment_dir, tb_dir):
    c = combo.arch_cfg
    treatment = c.get('treatment', 'baseline')

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    combo_dir = os.path.join(
        experiment_dir, combo.init_name, combo.arch_name, f'n{combo.train_size}',
    )
    os.makedirs(combo_dir, exist_ok=True)
    tb_run_dir = os.path.join(
        tb_dir, combo.init_name, combo.arch_name, f'n{combo.train_size}',
    )

    tag = f"[GPU {rank} {combo.init_name}/{combo.arch_name}/n={combo.train_size:,}]"
    print(f"{tag} Starting | treatment={treatment}")

    _resolve_runtime_checkpoint(combo, tag)
    print(f"{tag} Pretrained: {combo.checkpoint}" if combo.checkpoint
          else f"{tag} Encoder: random init")

    set_seed(42)

    train_x, train_y, val_x, val_y, norm_fn = _prepare_data(combo, config, device, tag)
    n_train = len(train_x)
    plan = plan_compute(c, n_train, config['scaling'])
    print(f"{tag} Train bs={plan.train_bs}, steps/epoch={plan.steps_per_epoch}, "
          f"total_steps={plan.total_steps:,}, "
          f"epochs={plan.total_steps / max(plan.steps_per_epoch, 1):.1f}")

    # Model.
    model = _build_model(c)
    if combo.checkpoint is not None:
        _load_pretrained(model, combo.checkpoint, treatment)
    model = model.to(device)
    model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} Parameters: {n_params:,}, RF={model.encoder.cnn.r0}")

    criterion = nn.BCEWithLogitsLoss()
    metrics = build_metrics(device)

    # Optimizer + encoder-adaptation mode.
    optimizer, scheduler, stage_boundary = _build_optimizer(model, c, plan.total_steps)
    encoder_mode = TREATMENTS.get(treatment, TREATMENTS['baseline']).encoder_mode
    boundary_step = stage_boundary * plan.steps_per_epoch if encoder_mode == 'staged' else 0
    if encoder_mode == 'staged':
        print(f"{tag} LP-FT: stage1 head-only for {stage_boundary} epochs "
              f"({boundary_step} steps); stage2 unfreezes encoder with "
              f"LLDR decay={c.get('lldr_decay', 0.7)}")
        _set_encoder_requires_grad(model, False)
    elif encoder_mode == 'frozen':
        print(f"{tag} linear_probe: encoder frozen for the whole run; "
              f"only head params will be trained")
        _set_encoder_requires_grad(model, False)

    # Resolve effective head/encoder LRs for logging. None => max_lr. Frozen
    # encoder logs encoder_lr=0.0.
    max_lr = float(c['max_lr'])
    head_lr = float(c['head_lr']) if c.get('head_lr') is not None else max_lr
    encoder_lr = (
        0.0 if encoder_mode == 'frozen' else
        (float(c['encoder_lr']) if c.get('encoder_lr') is not None else max_lr)
    )
    label_smoothing = float(c.get('label_smoothing', 0.0))

    logger = RunLogger(
        tb_run_dir, combo_dir, config, combo, plan.train_bs,
        max_lr, head_lr, encoder_lr, label_smoothing, treatment,
    )

    eval_count = _run_training_loop(
        model, optimizer, scheduler, criterion, metrics,
        train_x, train_y, val_x, val_y, plan, c, treatment,
        boundary_step, logger, combo_dir, norm_fn, config, tag,
    )

    logger.close()
    print(f"{tag} Done. {eval_count} checkpoints recorded.")


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
