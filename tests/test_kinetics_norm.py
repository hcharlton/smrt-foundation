"""
Equivalence tests for KineticsNorm vs ZNorm and SSLNorm.

Verifies that KineticsNorm produces identical statistics and identical
batch outputs as the original classes on their respective dataset types.

Run:
    python -m pytest tests/test_kinetics_norm.py -v -s
"""

import os
import sys
import torch
import numpy as np
import pytest
from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset, ShardedMemmapDataset
from smrt_foundation.normalization import ZNorm, KineticsNorm

# Import SSLNorm from the experiment script
sys.path.insert(0, os.path.join(module_path, 'scripts', 'experiments', 'ssl_21_pretrain'))
from train import SSLNorm


# ---------------------------------------------------------------------------
# Data paths (use subset data available locally)
# ---------------------------------------------------------------------------

POS_MEMMAP = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
NEG_MEMMAP = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'
SSL_MEMMAP_CANDIDATES = [
    'data/01_processed/ssl_sets/cpg_pos_subset.memmap',  # small zarr-derived
    'data/01_processed/val_sets/cpg_pos_subset.memmap/train',  # reuse as SSL-shaped
]


def find_ssl_memmap():
    for p in SSL_MEMMAP_CANDIDATES:
        if os.path.isdir(p) and any(f.endswith('.npy') for f in os.listdir(p)):
            return p
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def labeled_ds():
    if not os.path.isdir(POS_MEMMAP) or not os.path.isdir(NEG_MEMMAP):
        pytest.skip(f"Subset memmap data not found at {POS_MEMMAP}")
    return LabeledMemmapDataset(POS_MEMMAP, NEG_MEMMAP, limit=10000)


@pytest.fixture(scope="session")
def ssl_ds():
    path = find_ssl_memmap()
    if path is None:
        pytest.skip("No SSL-shaped memmap data found locally")
    return ShardedMemmapDataset(path, limit=10000)


# ---------------------------------------------------------------------------
# Test: ZNorm equivalence on labeled data
# ---------------------------------------------------------------------------

class TestZNormEquivalence:
    """KineticsNorm should match ZNorm on LabeledMemmapDataset."""

    def test_statistics_match(self, labeled_ds):
        """means and stds for kinetics channels should be identical."""
        torch.manual_seed(42)
        znorm = ZNorm(labeled_ds, log_transform=True)

        torch.manual_seed(42)
        knorm = KineticsNorm(labeled_ds, log_transform=True)

        # For CpG windows there's no padding, so the padding filter
        # in KineticsNorm shouldn't change the result
        torch.testing.assert_close(
            knorm.means[[1, 2]], znorm.means[[1, 2]],
            atol=1e-4, rtol=1e-4,
            msg="KineticsNorm means don't match ZNorm"
        )
        torch.testing.assert_close(
            knorm.stds[[1, 2]], znorm.stds[[1, 2]],
            atol=1e-4, rtol=1e-4,
            msg="KineticsNorm stds don't match ZNorm"
        )

    def test_batch_output_identical(self, labeled_ds):
        """Applying each norm to the same batch should produce identical tensors."""
        torch.manual_seed(42)
        znorm = ZNorm(labeled_ds, log_transform=True)

        torch.manual_seed(42)
        knorm = KineticsNorm(labeled_ds, log_transform=True)

        # Get a batch
        x_raw, y = labeled_ds[0]

        x_z = znorm(x_raw.clone())
        x_k = knorm(x_raw.clone())

        torch.testing.assert_close(
            x_k, x_z,
            atol=1e-5, rtol=1e-5,
            msg="KineticsNorm output doesn't match ZNorm output"
        )

    def test_multiple_samples(self, labeled_ds):
        """Check across 50 samples."""
        torch.manual_seed(42)
        znorm = ZNorm(labeled_ds, log_transform=True)

        torch.manual_seed(42)
        knorm = KineticsNorm(labeled_ds, log_transform=True)

        for i in range(min(50, len(labeled_ds))):
            x_raw, _ = labeled_ds[i]
            x_z = znorm(x_raw.clone())
            x_k = knorm(x_raw.clone())
            torch.testing.assert_close(
                x_k, x_z, atol=1e-5, rtol=1e-5,
                msg=f"Mismatch at sample {i}"
            )


# ---------------------------------------------------------------------------
# Test: SSLNorm equivalence on SSL data
# ---------------------------------------------------------------------------

class TestSSLNormEquivalence:
    """KineticsNorm should match SSLNorm on ShardedMemmapDataset."""

    def test_statistics_match(self, ssl_ds):
        """means and stds for kinetics channels should be identical."""
        torch.manual_seed(42)
        sslnorm = SSLNorm(ssl_ds)

        torch.manual_seed(42)
        knorm = KineticsNorm(ssl_ds, log_transform=True)

        torch.testing.assert_close(
            knorm.means[[1, 2]], sslnorm.means[[1, 2]],
            atol=1e-4, rtol=1e-4,
            msg="KineticsNorm means don't match SSLNorm"
        )
        torch.testing.assert_close(
            knorm.stds[[1, 2]], sslnorm.stds[[1, 2]],
            atol=1e-4, rtol=1e-4,
            msg="KineticsNorm stds don't match SSLNorm"
        )

    def test_batch_output_identical(self, ssl_ds):
        """Applying each norm to the same batch should produce identical tensors."""
        torch.manual_seed(42)
        sslnorm = SSLNorm(ssl_ds)

        torch.manual_seed(42)
        knorm = KineticsNorm(ssl_ds, log_transform=True)

        x_raw = ssl_ds[0]

        x_s = sslnorm(x_raw.clone())
        x_k = knorm(x_raw.clone())

        torch.testing.assert_close(
            x_k, x_s,
            atol=1e-5, rtol=1e-5,
            msg="KineticsNorm output doesn't match SSLNorm output"
        )

    def test_multiple_samples(self, ssl_ds):
        """Check across 50 samples."""
        torch.manual_seed(42)
        sslnorm = SSLNorm(ssl_ds)

        torch.manual_seed(42)
        knorm = KineticsNorm(ssl_ds, log_transform=True)

        for i in range(min(50, len(ssl_ds))):
            x_raw = ssl_ds[i]
            x_s = sslnorm(x_raw.clone())
            x_k = knorm(x_raw.clone())
            torch.testing.assert_close(
                x_k, x_s, atol=1e-5, rtol=1e-5,
                msg=f"Mismatch at sample {i}"
            )


# ---------------------------------------------------------------------------
# Test: KineticsNorm basic properties
# ---------------------------------------------------------------------------

class TestKineticsNormProperties:
    """Verify KineticsNorm has expected behavior."""

    def test_seq_and_mask_unchanged(self, labeled_ds):
        """Sequence tokens (col 0) and mask (col 3) should not be modified."""
        torch.manual_seed(42)
        knorm = KineticsNorm(labeled_ds, log_transform=True)

        x_raw, _ = labeled_ds[0]
        x_orig = x_raw.clone()
        x_normed = knorm(x_raw.clone())

        torch.testing.assert_close(x_normed[:, 0], x_orig[:, 0], msg="Seq tokens modified")
        torch.testing.assert_close(x_normed[:, 3], x_orig[:, 3], msg="Mask modified")

    def test_kinetics_are_modified(self, labeled_ds):
        """Kinetics channels (cols 1, 2) should be different after normalization."""
        torch.manual_seed(42)
        knorm = KineticsNorm(labeled_ds, log_transform=True)

        x_raw, _ = labeled_ds[0]
        x_orig = x_raw.clone()
        x_normed = knorm(x_raw.clone())

        assert not torch.equal(x_normed[:, 1], x_orig[:, 1]), "Col 1 unchanged"
        assert not torch.equal(x_normed[:, 2], x_orig[:, 2]), "Col 2 unchanged"

    def test_no_log_transform_mode(self, labeled_ds):
        """With log_transform=False, should just z-score without log1p."""
        torch.manual_seed(42)
        knorm = KineticsNorm(labeled_ds, log_transform=False)

        x_raw, _ = labeled_ds[0]
        x_normed = knorm(x_raw.clone())

        # Values should still be modified (z-scored)
        assert not torch.equal(x_normed[:, 1], x_raw[:, 1])
