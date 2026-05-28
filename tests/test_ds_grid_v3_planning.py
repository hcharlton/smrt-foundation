"""Characterization tests for the supervised_55 refactor of scripts/ds_grid_v3.py.

Locks the behaviour of the planning/expansion helpers that the in-place
refactor restructured, plus the new label-smoothing helper:

  - `plan_compute`: batch-size + step-budget resolution (with and without the
    bs_floor/bs_k scaling knobs), cross-checked against `compute_step_budget`.
  - `expand_combos`: arch x init x train_size fan-out, `skip` pairs, and
    `auto_best` deferral routing the SSL dir onto `Combo.ssl_exp_dir`.
  - `_smooth_targets`: {0,1} -> {ls/2, 1-ls/2}, identity at ls=0.

Pure-Python: no GPU, no data, no checkpoints (auto_best is pointed at a
non-existent dir so it defers instead of resolving).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.ds_grid_v3 import (
    DEFAULT_CLASSIFIER,
    Combo,
    ComputePlan,
    plan_compute,
    expand_combos,
    _smooth_targets,
)
from scripts.ds_grid import compute_step_budget


SCALING = {
    'val_limit': 1_000_000,
    'max_epochs': 20,
    'min_steps': 100,
    'max_steps': 400_000,
    'n_evals': 40,
    'first_eval_step': 100,
}


def _classifier(**overrides):
    return DEFAULT_CLASSIFIER | {'max_lr': 3e-3, 'batch_size': 4096, **overrides}


def _config(**overrides):
    cfg = {
        'architectures': {
            'd768_L8': {'d_model': 768, 'n_layers': 8, 'n_head': 12, 'layer_idx': 3},
        },
        'inits': {
            'random': {'checkpoint': None},
            'ssl_best': {
                'checkpoint': 'auto_best',
                # Non-existent on purpose: forces the deferral path instead of
                # calling select_best_ssl_checkpoint at expand time.
                'ssl_exp_dirs': {'d768_L8': '/nonexistent/ssl/dir/size_d768_L8_long'},
            },
        },
        'train_sizes': [10_000, 100_000],
        'classifier': _classifier(),
        'scaling': SCALING,
        'skip': [],
    }
    cfg.update(overrides)
    return cfg


class TestPlanCompute:
    def test_no_scaling_caps_at_train_size(self):
        cfg = _classifier(batch_size=4096)
        plan = plan_compute(cfg, n=10_000, scaling=SCALING)
        assert isinstance(plan, ComputePlan)
        assert plan.train_bs == 4096
        ts, spe, ev = compute_step_budget(10_000, 4096, SCALING)
        assert plan.total_steps == ts
        assert plan.steps_per_epoch == spe
        assert plan.eval_steps == ev

    def test_batch_capped_below_train_size(self):
        cfg = _classifier(batch_size=4096)
        plan = plan_compute(cfg, n=512, scaling=SCALING)
        assert plan.train_bs == 512  # min(batch_size, n)

    def test_bs_floor_bs_k_path(self):
        cfg = _classifier(batch_size=4096, bs_floor=256, bs_k=8)
        # bs = min(4096, max(256, n // 8)) then min(., n)
        plan = plan_compute(cfg, n=100_000, scaling=SCALING)
        assert plan.train_bs == min(4096, max(256, 100_000 // 8))
        assert plan.train_bs == 4096


class TestExpandCombos:
    def test_fan_out_count(self):
        combos = expand_combos(_config())
        # 1 arch x 2 inits x 2 sizes
        assert len(combos) == 4

    def test_random_init_has_no_checkpoint(self):
        combos = expand_combos(_config())
        randoms = [c for c in combos if c.init_name == 'random']
        assert len(randoms) == 2
        assert all(c.checkpoint is None for c in randoms)
        assert all(c.ssl_exp_dir is None for c in randoms)

    def test_auto_best_defers_with_ssl_dir(self):
        combos = expand_combos(_config())
        ssl = [c for c in combos if c.init_name == 'ssl_best']
        assert len(ssl) == 2
        assert all(c.checkpoint == 'auto_best' for c in ssl)
        assert all(
            c.ssl_exp_dir == '/nonexistent/ssl/dir/size_d768_L8_long' for c in ssl
        )

    def test_train_sizes_fan_out(self):
        combos = expand_combos(_config())
        sizes = sorted({c.train_size for c in combos})
        assert sizes == [10_000, 100_000]

    def test_skip_pair_excluded(self):
        cfg = _config(skip=[{'arch': 'd768_L8', 'init': 'random'}])
        combos = expand_combos(cfg)
        assert all(c.init_name != 'random' for c in combos)
        assert len(combos) == 2  # only ssl_best x 2 sizes

    def test_combos_are_dataclass_instances(self):
        combos = expand_combos(_config())
        assert all(isinstance(c, Combo) for c in combos)


class TestSmoothTargets:
    def test_identity_at_zero(self):
        t = torch.tensor([[0.0], [1.0], [0.0]])
        out = _smooth_targets(t, 0.0)
        assert torch.equal(out, t)

    def test_negative_is_identity(self):
        t = torch.tensor([[0.0], [1.0]])
        assert torch.equal(_smooth_targets(t, -0.1), t)

    def test_maps_to_eps_band(self):
        t = torch.tensor([[0.0], [1.0]])
        out = _smooth_targets(t, 0.1)
        # 0 -> ls/2 = 0.05 ; 1 -> 1 - ls/2 = 0.95
        assert pytest.approx(out[0].item(), abs=1e-6) == 0.05
        assert pytest.approx(out[1].item(), abs=1e-6) == 0.95
