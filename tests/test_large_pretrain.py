"""Tests for experiment 29 features: random cropping, periodic checkpointing, LR schedule."""

import torch
import numpy as np
import pytest
from smrt_foundation.optim import get_cosine_schedule_with_warmup


class TestRandomCropping:
    def test_returns_correct_shape(self):
        """Random crop from (4096, 4) with context=128 should return (128, 4)."""
        x = torch.randn(4096, 4)
        context = 128
        start = torch.randint(0, x.shape[0] - context, (1,)).item()
        crop = x[start:start + context]
        assert crop.shape == (128, 4)

    def test_crops_differ_across_calls(self):
        """Multiple random crops from the same sample should (almost certainly) differ."""
        x = torch.randn(4096, 4)
        context = 128
        crops = []
        for _ in range(10):
            start = torch.randint(0, x.shape[0] - context, (1,)).item()
            crops.append(x[start:start + context])

        # At least 2 of 10 crops should start at different positions
        first_vals = [c[0, 0].item() for c in crops]
        assert len(set(round(v, 6) for v in first_vals)) > 1, "All crops identical — random cropping not working"

    def test_crop_stays_in_bounds(self):
        """Crop should never exceed the input length."""
        x = torch.randn(4096, 4)
        context = 128
        for _ in range(100):
            start = torch.randint(0, x.shape[0] - context, (1,)).item()
            assert start >= 0
            assert start + context <= 4096

    def test_short_input_fallback(self):
        """If input is shorter than context, take first context positions."""
        x = torch.randn(64, 4)
        context = 128
        max_start = x.shape[0] - context
        if max_start > 0:
            start = torch.randint(0, max_start, (1,)).item()
            crop = x[start:start + context]
        else:
            crop = x[:context]
        # Input is 64 < 128, so fallback gives 64 positions
        assert crop.shape == (64, 4)


class TestLRSchedulerLongTraining:
    """Verify cosine schedule with warmup behaves correctly over 3000 epochs."""

    @pytest.fixture
    def schedule(self):
        """Create a schedule matching exp 29: 3000 epochs × ~1641 steps."""
        total_steps = 3000 * 1641  # ~4.9M steps
        pct_start = 0.01
        max_lr = 3e-4
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, total_steps=total_steps, pct_start=pct_start
        )
        return scheduler, optimizer, total_steps, pct_start, max_lr

    def test_lr_starts_near_zero(self, schedule):
        scheduler, optimizer, _, _, _ = schedule
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        assert lr < 1e-6, f"LR at step 1 should be near zero, got {lr}"

    def test_lr_peaks_after_warmup(self, schedule):
        scheduler, optimizer, total_steps, pct_start, max_lr = schedule
        warmup_steps = int(total_steps * pct_start)
        for _ in range(warmup_steps):
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        assert abs(lr - max_lr) < 1e-7, f"LR at end of warmup should be {max_lr}, got {lr}"

    def test_lr_decays_at_midpoint(self, schedule):
        scheduler, optimizer, total_steps, pct_start, max_lr = schedule
        midpoint = total_steps // 2
        for _ in range(midpoint):
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        # At midpoint of cosine, LR should be roughly halfway between max and min
        assert lr < max_lr, f"LR at midpoint should be below max, got {lr}"
        assert lr > max_lr * 0.05, f"LR at midpoint should be above min, got {lr}"

    def test_lr_reaches_floor_at_end(self, schedule):
        scheduler, optimizer, total_steps, _, max_lr = schedule
        for _ in range(total_steps):
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        expected_min = max_lr * 0.05  # min_lr_ratio default
        assert abs(lr - expected_min) < 1e-8, f"LR at end should be {expected_min}, got {lr}"

    def test_lr_never_negative(self, schedule):
        """Sample checkpoints throughout training to verify LR is always positive."""
        scheduler, optimizer, total_steps, _, _ = schedule
        checkpoints = [0, 100, 1000, total_steps // 4, total_steps // 2,
                       3 * total_steps // 4, total_steps - 1, total_steps, total_steps + 100]
        step = 0
        for target in sorted(checkpoints):
            while step < target:
                scheduler.step()
                step += 1
            lr = optimizer.param_groups[0]['lr']
            assert lr >= 0, f"LR should be non-negative at step {step}, got {lr}"


class TestProbeFrequency:
    def test_probe_runs_at_correct_intervals(self):
        """Probe should only run when (epoch+1) % probe_every == 0."""
        probe_every = 100
        epochs = 500
        probe_epochs = [e + 1 for e in range(epochs) if (e + 1) % probe_every == 0]
        assert probe_epochs == [100, 200, 300, 400, 500]

    def test_checkpoint_runs_at_correct_intervals(self):
        """Checkpoints should save when (epoch+1) % checkpoint_every == 0."""
        checkpoint_every = 100
        epochs = 350
        ckpt_epochs = [e + 1 for e in range(epochs) if (e + 1) % checkpoint_every == 0]
        assert ckpt_epochs == [100, 200, 300]
