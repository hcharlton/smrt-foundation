"""Tests for the supervised_54 additions to scripts/ds_grid_v3.py.

Covers two harness features added so the d=768 LR x recipe ablation can run
without further code changes:

  - `treatment='linear_probe'`: encoder permanently frozen, optimizer only
    touches the classifier head, no stage transition.
  - Decoupled `head_lr` / `encoder_lr` on non-LLDR treatments: two parameter
    groups when the two LRs differ, single group when they match (back-compat).

These tests construct optimizers directly via `_build_optimizer` without any
training data or GPU, so they run quickly in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.ds_grid_v3 import _build_optimizer, _is_head_param, VALID_TREATMENTS
from smrt_foundation.model import (
    DirectClassifierSmallRF,
    DirectClassifierMidLayer,
)


D_MODEL = 32
N_LAYERS = 4
N_HEAD = 2
CONTEXT = 32


def _base_cfg(**overrides):
    cfg = {
        'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_head': N_HEAD,
        'context': CONTEXT,
        'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
        'treatment': 'baseline',
        'head_lr': None, 'encoder_lr': None,
        'warmup_epochs': 5, 'lldr_decay': 0.7,
    }
    cfg.update(overrides)
    return cfg


class TestIsHeadParam:
    """The head-param matcher must accept both `head.*` and `head_ln.*`
    without false-positive matches against unrelated names like `encoder.*`."""

    def test_matches_head_sequential(self):
        assert _is_head_param('head.0.weight')
        assert _is_head_param('head.2.bias')

    def test_matches_head_ln(self):
        assert _is_head_param('head_ln.weight')
        assert _is_head_param('head_ln.bias')

    def test_rejects_encoder(self):
        assert not _is_head_param('encoder.blocks.0.attn.c_attn.weight')
        assert not _is_head_param('encoder.cnn.extractor.0.conv1.weight')
        assert not _is_head_param('encoder.embed.layernorm.weight')

    def test_rejects_decoder_upsample(self):
        # F3 (DirectClassifierWithDecoder) has decoder_upsample sitting between
        # encoder and head — it must NOT be treated as head; otherwise
        # `linear_probe` would inadvertently train the upsample stack too.
        assert not _is_head_param('decoder_upsample.0.weight')


class TestLinearProbeRegistered:
    def test_in_valid_treatments(self):
        assert 'linear_probe' in VALID_TREATMENTS


class TestLinearProbeOptimizer:
    """`linear_probe` optimizer should only touch head parameters; encoder
    params don't even appear in any param group."""

    def test_only_head_params_with_smallrf_classifier(self):
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = _base_cfg(treatment='linear_probe')
        opt, sched, boundary = _build_optimizer(model, cfg, total_steps=1000)
        assert boundary == 0
        assert len(opt.param_groups) == 1
        opt_param_ids = {id(p) for p in opt.param_groups[0]['params']}
        # Every head param is in the optimizer.
        head_ids = {id(p) for n, p in model.named_parameters() if _is_head_param(n)}
        assert head_ids.issubset(opt_param_ids), (
            "linear_probe optimizer is missing some head params"
        )
        # No encoder param is in the optimizer.
        enc_ids = {id(p) for n, p in model.named_parameters() if not _is_head_param(n)}
        assert opt_param_ids.isdisjoint(enc_ids), (
            "linear_probe optimizer should not see encoder params; "
            "freeze must happen via requires_grad in train_one_combo too, but "
            "the optimizer must not iterate over them in the first place."
        )

    def test_only_head_params_with_midlayer_classifier(self):
        # In supervised_54, treatment='linear_probe' uses DirectClassifierSmallRF
        # by dispatch — but if a future config combined midlayer with LP-style
        # freezing, head_ln must also count as head. This guards against
        # someone forgetting head_ln on a midlayer linear_probe.
        model = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=1, cnn_variant='small_rf',
        )
        cfg = _base_cfg(treatment='linear_probe')
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        opt_param_ids = {id(p) for p in opt.param_groups[0]['params']}
        # head_ln must be in there.
        ln_ids = {id(p) for n, p in model.named_parameters() if n.startswith('head_ln.')}
        assert ln_ids.issubset(opt_param_ids)

    def test_head_lr_respected(self):
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = _base_cfg(treatment='linear_probe', head_lr=1e-2)
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        # `lr` reflects the current cosine-warmup scaling (0 at step 0);
        # `initial_lr` is the configured starting LR before any scheduler step.
        assert pytest.approx(opt.param_groups[0]['initial_lr'], rel=1e-9) == 1e-2

    def test_head_lr_falls_back_to_max_lr(self):
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = _base_cfg(treatment='linear_probe')  # head_lr=None
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        # Should fall back to max_lr (3e-3).
        assert pytest.approx(opt.param_groups[0]['initial_lr'], rel=1e-9) == 3e-3


class TestDecoupledLR:
    """Non-LLDR treatments: when head_lr != encoder_lr, build 2 groups."""

    def test_two_groups_when_lrs_differ(self):
        model = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=1, cnn_variant='small_rf',
        )
        cfg = _base_cfg(treatment='midlayer', head_lr=3e-3, encoder_lr=3e-4)
        opt, _, boundary = _build_optimizer(model, cfg, total_steps=1000)
        assert boundary == 0
        assert len(opt.param_groups) == 2
        named = {g.get('name'): g for g in opt.param_groups}
        assert {'head', 'encoder'} <= set(named.keys())
        # `initial_lr` is the configured pre-scheduler base; `lr` is 0 at
        # warmup step 0.
        assert pytest.approx(named['head']['initial_lr'], rel=1e-9) == 3e-3
        assert pytest.approx(named['encoder']['initial_lr'], rel=1e-9) == 3e-4

    def test_head_group_contains_head_and_head_ln(self):
        model = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=1, cnn_variant='small_rf',
        )
        cfg = _base_cfg(treatment='midlayer', head_lr=3e-3, encoder_lr=3e-4)
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        named = {g.get('name'): g for g in opt.param_groups}
        head_ids = {id(p) for p in named['head']['params']}
        expected_head_ids = {
            id(p) for n, p in model.named_parameters() if _is_head_param(n)
        }
        assert head_ids == expected_head_ids

    def test_encoder_group_excludes_head(self):
        model = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=1, cnn_variant='small_rf',
        )
        cfg = _base_cfg(treatment='midlayer', head_lr=3e-3, encoder_lr=3e-4)
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        named = {g.get('name'): g for g in opt.param_groups}
        enc_ids = {id(p) for p in named['encoder']['params']}
        head_ids = {id(p) for n, p in model.named_parameters() if _is_head_param(n)}
        assert enc_ids.isdisjoint(head_ids)
        # Every non-head param ends up in the encoder group.
        expected_enc_ids = {
            id(p) for n, p in model.named_parameters() if not _is_head_param(n)
        }
        assert enc_ids == expected_enc_ids

    def test_single_group_when_lrs_match(self):
        # Equal head_lr / encoder_lr should preserve the v2 single-group
        # codepath, since adding a degenerate split changes nothing
        # behaviourally but would clutter the optimizer state.
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = _base_cfg(treatment='baseline', head_lr=3e-3, encoder_lr=3e-3)
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        assert len(opt.param_groups) == 1

    def test_single_group_when_both_none(self):
        # Back-compat: configs that never set head_lr/encoder_lr (e.g. existing
        # supervised_53 entries) should produce a single param group.
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = _base_cfg(treatment='baseline')  # both None
        opt, _, _ = _build_optimizer(model, cfg, total_steps=1000)
        assert len(opt.param_groups) == 1
        assert pytest.approx(opt.param_groups[0]['initial_lr'], rel=1e-9) == 3e-3

    def test_cosine_decays_both_groups(self):
        """Step the scheduler past warmup; both head and encoder LRs must
        scale down together (cosine is per-group on initial_lr)."""
        model = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=1, cnn_variant='small_rf',
        )
        cfg = _base_cfg(treatment='midlayer', head_lr=3e-3, encoder_lr=3e-4)
        opt, sched, _ = _build_optimizer(model, cfg, total_steps=100)
        head_initial = opt.param_groups[0]['initial_lr']
        enc_initial = opt.param_groups[1]['initial_lr']
        # Step well past warmup (pct_start=0.1 -> warmup ends at step 10).
        for _ in range(80):
            sched.step()
        head_after = opt.param_groups[0]['lr']
        enc_after = opt.param_groups[1]['lr']
        # Both groups should have decayed by the same cosine factor relative
        # to their respective `initial_lr` bases.
        head_ratio = head_after / head_initial
        enc_ratio = enc_after / enc_initial
        assert pytest.approx(head_ratio, rel=1e-4) == enc_ratio, (
            f"cosine should scale both groups by the same factor; "
            f"got head_ratio={head_ratio:.6f} enc_ratio={enc_ratio:.6f}"
        )

    def test_lpft_lldr_unaffected_by_decoupled_knobs(self):
        # When head_lr/encoder_lr are set but treatment=='lpft_lldr', the
        # LLDR branch should ignore them and build its standard LLDR param
        # group ladder. This guards against accidental cross-talk between
        # the two LR-customisation paths.
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = _base_cfg(
            treatment='lpft_lldr',
            head_lr=3e-3, encoder_lr=3e-4,  # ignored under lpft_lldr
            warmup_epochs=5, lldr_decay=0.7,
        )
        opt, _, warmup_epochs = _build_optimizer(model, cfg, total_steps=1000)
        assert warmup_epochs == 5
        # Expect head + N_LAYERS block groups + base = N_LAYERS + 2 groups.
        assert len(opt.param_groups) == N_LAYERS + 2

    def test_baseline_treatment_back_compat(self):
        # Regression: a config with no head_lr/encoder_lr fields must produce
        # the original single-group baseline behaviour with lr=max_lr.
        model = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = {
            'treatment': 'baseline',
            'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
            # head_lr / encoder_lr deliberately absent (not None) — exercises
            # the .get() default path in _build_optimizer.
        }
        opt, _, boundary = _build_optimizer(model, cfg, total_steps=1000)
        assert boundary == 0
        assert len(opt.param_groups) == 1
        assert pytest.approx(opt.param_groups[0]['initial_lr'], rel=1e-9) == 3e-3
