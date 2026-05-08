"""Tests for the fine-tune revamp (supervised_53_finetune_revamp / F-series).

Covers:
  - SmrtEncoder.forward_to_layer parity at the final layer (Phase 0.3)
  - DirectClassifierMidLayer accepts a layer_idx and reads from there (F1)
  - DirectClassifierWithDecoder loads ssl_58-style decoder weights (F3)
  - DirectClassifierBigHead has the expected 3-layer head structure (F4)
  - LP-FT + LLDR optimizer wiring produces strictly-decreasing per-layer LRs (F2)
  - select_best_ssl_checkpoint smoke (Phase 3.0) — temp dir with a fake CSV.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from smrt_foundation.model import (
    SmrtEncoder, SmrtEncoderSmallRF,
    DirectClassifierMidLayer, DirectClassifierWithDecoder,
    DirectClassifierBigHead, SmrtDecoder,
)


D_MODEL = 32
N_LAYERS = 4
N_HEAD = 2
CONTEXT = 32
BATCH = 4


@pytest.fixture
def sample_input():
    x = torch.zeros(BATCH, CONTEXT, 4)
    x[..., 0] = torch.randint(0, 5, (BATCH, CONTEXT)).float()
    x[..., 1] = torch.randn(BATCH, CONTEXT)
    x[..., 2] = torch.randn(BATCH, CONTEXT)
    return x


class TestForwardToLayer:
    """Phase 0.3: SmrtEncoder.forward_to_layer."""

    def test_returns_correct_shape(self, sample_input):
        enc = SmrtEncoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT).eval()
        with torch.no_grad():
            for li in range(N_LAYERS):
                c = enc.forward_to_layer(sample_input, li)
            # Final output shape is [B, T/4, d] (4x downsample)
            assert c.shape == (BATCH, CONTEXT // 4, D_MODEL)

    def test_final_layer_matches_forward(self, sample_input):
        enc = SmrtEncoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT).eval()
        with torch.no_grad():
            via_forward = enc(sample_input)
            via_to_layer = enc.forward_to_layer(sample_input, N_LAYERS - 1)
            via_neg1 = enc.forward_to_layer(sample_input, -1)
        assert torch.allclose(via_forward, via_to_layer, atol=1e-6)
        assert torch.allclose(via_forward, via_neg1, atol=1e-6)

    def test_middle_layer_differs_from_final(self, sample_input):
        enc = SmrtEncoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT).eval()
        with torch.no_grad():
            mid = enc.forward_to_layer(sample_input, 0)
            final = enc.forward_to_layer(sample_input, N_LAYERS - 1)
        assert not torch.allclose(mid, final, atol=1e-3)

    def test_smallrf_inherits(self, sample_input):
        enc = SmrtEncoderSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT).eval()
        with torch.no_grad():
            c = enc.forward_to_layer(sample_input, 0)
        assert c.shape == (BATCH, CONTEXT // 4, D_MODEL)


class TestDirectClassifierMidLayer:
    """F1."""

    def test_forward_shape(self, sample_input):
        cls = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=1,
        )
        out = cls(sample_input)
        assert out.shape == (BATCH, 1)

    def test_layer_idx_changes_output(self, sample_input):
        torch.manual_seed(0)
        cls0 = DirectClassifierMidLayer(D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=0).eval()
        torch.manual_seed(0)
        cls3 = DirectClassifierMidLayer(D_MODEL, N_LAYERS, N_HEAD, CONTEXT, layer_idx=3).eval()
        # Re-seed the encoder weights identically; layer_idx is the only
        # difference. The two outputs should differ on a non-trivial input.
        with torch.no_grad():
            assert not torch.allclose(cls0(sample_input), cls3(sample_input), atol=1e-3)

    def test_loads_smrtencoder_smallrf_state_dict(self):
        """Encoder portion should accept a SmrtEncoderSmallRF state_dict (the
        ssl_58 / ssl_59 case). PE shape mismatch tolerated by strict=False."""
        ssl_enc = SmrtEncoderSmallRF(D_MODEL, N_LAYERS, N_HEAD, max_len=512)  # ssl context
        cls = DirectClassifierMidLayer(
            D_MODEL, N_LAYERS, N_HEAD, max_len=CONTEXT,  # supervised context
            layer_idx=1, cnn_variant='small_rf',
        )
        sd = {k: v for k, v in ssl_enc.state_dict().items()
              if not (k == 'pe.pe' and v.shape != cls.encoder.pe.pe.shape)}
        missing, unexpected = cls.encoder.load_state_dict(sd, strict=False)
        # Only PE should be missing (different context length).
        assert all('pe.pe' in m for m in missing), f"unexpected missing: {missing}"
        assert not unexpected


class TestDirectClassifierWithDecoder:
    """F3."""

    def test_forward_shape(self, sample_input):
        cls = DirectClassifierWithDecoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        assert cls(sample_input).shape == (BATCH, 1)

    def test_loads_smrtdecoder_upsample(self):
        """SSL decoder is `SmrtDecoder` whose `.upsample.*` keys should map onto
        DirectClassifierWithDecoder's `decoder_upsample.*` after a prefix
        rename."""
        sd = SmrtDecoder(D_MODEL).state_dict()
        # Filter to upsample.* and remap.
        remapped = {
            k.replace('upsample.', 'decoder_upsample.', 1): v
            for k, v in sd.items() if k.startswith('upsample.')
        }
        cls = DirectClassifierWithDecoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        missing, unexpected = cls.load_state_dict(remapped, strict=False)
        # Encoder + head are missing (we only loaded decoder_upsample).
        assert all(m.startswith(('encoder.', 'head.')) for m in missing)
        assert not unexpected, f"unexpected: {unexpected}"


class TestDirectClassifierBigHead:
    """F4."""

    def test_forward_shape(self, sample_input):
        cls = DirectClassifierBigHead(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        assert cls(sample_input).shape == (BATCH, 1)

    def test_head_has_three_linear_layers(self):
        cls = DirectClassifierBigHead(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        linears = [m for m in cls.head if isinstance(m, nn.Linear)]
        assert len(linears) == 3

    def test_head_keeps_full_width(self):
        cls = DirectClassifierBigHead(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        linears = [m for m in cls.head if isinstance(m, nn.Linear)]
        # Linear(d, d) -> Linear(d, d/2) -> Linear(d/2, 1)
        assert linears[0].in_features == D_MODEL and linears[0].out_features == D_MODEL
        assert linears[1].in_features == D_MODEL and linears[1].out_features == D_MODEL // 2
        assert linears[2].in_features == D_MODEL // 2 and linears[2].out_features == 1


class TestLPFTOptimizer:
    """F2: layer-wise LR decay produces strictly-decreasing LRs from head to base."""

    def test_lr_decay_monotone(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.ds_grid_v3 import _build_optimizer

        from smrt_foundation.model import DirectClassifierSmallRF
        cls = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = {
            'treatment': 'lpft_lldr',
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_head': N_HEAD,
            'context': CONTEXT,
            'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
            'warmup_epochs': 5, 'lldr_decay': 0.7,
        }
        opt, sched, warmup_epochs = _build_optimizer(cls, cfg, total_steps=1000)
        assert warmup_epochs == 5
        # Read `initial_lr` (the base before scheduler scaling). Group order:
        # head, block_0, block_1, ..., block_{L-1}, base. block_i gets
        # max_lr * decay^(L - i): block_0=max_lr*decay^L (smallest of blocks),
        # block_{L-1}=max_lr*decay^1 (largest of blocks), base=max_lr*decay^(L+1).
        names = [g.get('name', '') for g in opt.param_groups]
        lrs = [g.get('initial_lr', g['lr']) for g in opt.param_groups]
        head_idx = names.index('head')
        base_idx = names.index('base')
        block_lrs = sorted(
            [(int(n.split('_')[1]), lr) for n, lr in zip(names, lrs) if n.startswith('block_')],
            key=lambda kv: kv[0],
        )
        # block_0 (deepest) < block_{L-1} (top, just under head)
        assert block_lrs[0][1] < block_lrs[-1][1], block_lrs
        # head > top block > deepest block > base
        assert lrs[head_idx] > block_lrs[-1][1] > block_lrs[0][1] > lrs[base_idx]

    def test_baseline_treatment_single_group(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.ds_grid_v3 import _build_optimizer
        from smrt_foundation.model import DirectClassifierSmallRF
        cls = DirectClassifierSmallRF(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        cfg = {
            'treatment': 'baseline',
            'max_lr': 3e-3, 'weight_decay': 0.02, 'pct_start': 0.1,
        }
        opt, _, boundary = _build_optimizer(cls, cfg, total_steps=1000)
        assert boundary == 0
        assert len(opt.param_groups) == 1


class TestSelectBestCheckpoint:
    """Phase 3.0: best-checkpoint resolver with a synthetic probe_history.csv."""

    def test_picks_max_top1(self, tmp_path: Path):
        from scripts.utils.select_best_ssl_checkpoint import select_best_ssl_checkpoint

        exp_dir = tmp_path / 'exp'
        ckpt_dir = exp_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True)
        for step in (0, 100, 200, 300, 400):
            (ckpt_dir / f'step_{step}.pt').write_bytes(b'x')
        (exp_dir / 'config.yaml').write_text(
            "experiment_type: ssl\nexperiment_name: testexp\n"
        )

        tb_logs = tmp_path / 'tb' / 'ssl' / 'testexp' / 'run_0'
        tb_logs.mkdir(parents=True)
        with open(tb_logs / 'probe_history.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'probe_top1', 'probe_auroc'])
            w.writerow([0, 0.50, 0.55])
            w.writerow([100, 0.55, 0.60])
            w.writerow([200, 0.62, 0.68])  # peak
            w.writerow([300, 0.60, 0.66])
            w.writerow([400, 0.59, 0.64])

        path = select_best_ssl_checkpoint(
            exp_dir, training_logs_root=str(tmp_path / 'tb'),
        )
        assert path.endswith('step_200.pt')

    def test_falls_back_to_final_when_no_csv(self, tmp_path: Path):
        from scripts.utils.select_best_ssl_checkpoint import select_best_ssl_checkpoint
        exp_dir = tmp_path / 'exp'
        ckpt_dir = exp_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / 'final_model.pt').write_bytes(b'x')
        (exp_dir / 'config.yaml').write_text(
            "experiment_type: ssl\nexperiment_name: testexp\n"
        )
        path = select_best_ssl_checkpoint(
            exp_dir, training_logs_root=str(tmp_path / 'tb_missing'),
        )
        assert path.endswith('final_model.pt')
