"""
End-to-end tests for zarr_to_methyl_memmap_v2.py.

Verifies every step from Zarr → shards → LabeledMemmapDataset → DataLoader.
Uses Zarr fixtures (not BAMs — BAMs may be mocked on Gefion).

Run:
    python -m pytest tests/test_zarr_to_methyl_memmap_v2.py -v -s
"""

import os
import glob
import tempfile

import numpy as np
import pytest
import torch
import yaml
import zarr

from scripts.zarr_to_methyl_memmap_v2 import (
    zarr_to_methyl_memmap_v2,
    find_cpg_positions,
    extract_cpg_windows,
    build_rc_lookup,
)
from smrt_foundation.dataset import LabeledMemmapDataset

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data.yaml')
ZARR_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '01_processed', 'ssl_sets')
POS_ZARR_CANDIDATES = ['cpg_pos_subset.zarr', 'cpg_pos.zarr']
NEG_ZARR_CANDIDATES = ['cpg_neg_subset.zarr', 'cpg_neg.zarr']
CONTEXT = 32
SHARD_SIZE = 4096  # small for testing


def _find_zarr(candidates):
    for name in candidates:
        path = os.path.join(ZARR_DIR, name)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def pos_zarr():
    path = _find_zarr(POS_ZARR_CANDIDATES)
    if path is None:
        pytest.skip(f"No pos Zarr found in {ZARR_DIR}")
    return path


@pytest.fixture(scope="session")
def neg_zarr():
    path = _find_zarr(NEG_ZARR_CANDIDATES)
    if path is None:
        pytest.skip(f"No neg Zarr found in {ZARR_DIR}")
    return path


@pytest.fixture(scope="session")
def pos_shards(pos_zarr, config, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("v2_pos"))
    zarr_to_methyl_memmap_v2(
        pos_zarr, out, config,
        context=CONTEXT, shard_size=SHARD_SIZE,
        val_pct=0.2, seed=42,
    )
    return out


@pytest.fixture(scope="session")
def neg_shards(neg_zarr, config, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("v2_neg"))
    zarr_to_methyl_memmap_v2(
        neg_zarr, out, config,
        context=CONTEXT, shard_size=SHARD_SIZE,
        val_pct=0.2, seed=42,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_shards(directory):
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    if not paths:
        return np.empty((0,))
    return np.concatenate([np.load(p) for p in paths], axis=0)


def get_first_read(zarr_path):
    """Return the first read's data array and its feature list."""
    root = zarr.open(zarr_path, mode='r')
    indptr = root['indptr'][:]
    feats = root.attrs['features']
    start, end = int(indptr[0]), int(indptr[1])
    data = root['data'][start:end, :]
    return data, feats


# ---------------------------------------------------------------------------
# Stage 1: CpG extraction correctness
# ---------------------------------------------------------------------------

class TestCpgExtraction:
    """Verify CpG window finding and counting."""

    def test_all_windows_have_cg_at_center(self, pos_shards):
        """Every shard row should have C=1, G=2 at the center."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("No data")

        pad = (CONTEXT - 2) // 2
        seq = all_data[:, :, 0]
        assert (seq[:, pad] == 1.0).all(), "Not all windows have C at center"
        assert (seq[:, pad + 1] == 2.0).all(), "Not all windows have G at center+1"

    def test_window_count_matches_zarr(self, pos_zarr, pos_shards, config):
        """Total windows should equal 2x the number of CpG sites found in Zarr
        (one forward + one reverse view per CpG)."""
        root = zarr.open(pos_zarr, mode='r')
        indptr = root['indptr'][:]
        z_data = root['data']
        feats = root.attrs['features']
        seq_idx = feats.index('seq')

        # Count CpG sites across all reads
        total_cpg = 0
        n_reads = len(indptr) - 1
        for r in range(n_reads):
            start, end = int(indptr[r]), int(indptr[r + 1])
            seq = z_data[start:end, seq_idx]
            total_cpg += sum(1 for _ in find_cpg_positions(seq, CONTEXT))

        # Count shard windows (train + val)
        train_data = load_all_shards(os.path.join(pos_shards, "train"))
        val_data = load_all_shards(os.path.join(pos_shards, "val"))
        total_windows = (len(train_data) if train_data.ndim == 3 else 0) + \
                        (len(val_data) if val_data.ndim == 3 else 0)

        assert total_windows == 2 * total_cpg, (
            f"Expected 2 * {total_cpg} = {2*total_cpg} windows, got {total_windows}"
        )

    def test_forward_and_reverse_alternate(self, pos_shards):
        """Windows are written in pairs: fwd then rev. Both should have CG at center."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) < 2:
            pytest.skip("Need at least 2 windows")

        pad = (CONTEXT - 2) // 2
        # Check pairs
        for i in range(0, min(20, len(all_data)), 2):
            fwd = all_data[i]
            rev = all_data[i + 1]
            # Both should have CG at center
            assert fwd[pad, 0] == 1.0 and fwd[pad + 1, 0] == 2.0, f"Fwd window {i} missing CG"
            assert rev[pad, 0] == 1.0 and rev[pad + 1, 0] == 2.0, f"Rev window {i+1} missing CG"


# ---------------------------------------------------------------------------
# Stage 2: Kinetics alignment
# ---------------------------------------------------------------------------

class TestKineticsAlignment:
    """Verify kinetics values match the Zarr source at correct positions."""

    def test_forward_kinetics_match_zarr(self, pos_zarr, pos_shards, config):
        """Forward view fi/fp should match Zarr fi/fp at [win_start:win_end]."""
        read_data, feats = get_first_read(pos_zarr)
        feature_idx = {f: feats.index(f) for f in ['seq', 'fi', 'fp', 'ri', 'rp']}

        seq = read_data[:, feature_idx['seq']]
        fi = read_data[:, feature_idx['fi']]
        fp = read_data[:, feature_idx['fp']]

        # Get first CpG
        cpg_iter = find_cpg_positions(seq, CONTEXT)
        first = next(cpg_iter, None)
        if first is None:
            pytest.skip("No CpG in first read")
        win_start, win_end, _, _ = first

        expected_fi = fi[win_start:win_end].astype(np.float16)
        expected_fp = fp[win_start:win_end].astype(np.float16)

        # Load first forward window from shards (index 0)
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            all_data = load_all_shards(os.path.join(pos_shards, "val"))
        if len(all_data) == 0:
            pytest.skip("No windows")

        fwd_row = all_data[0]
        np.testing.assert_array_equal(
            fwd_row[:, 1], expected_fi,
            err_msg="Forward fi doesn't match Zarr"
        )
        np.testing.assert_array_equal(
            fwd_row[:, 2], expected_fp,
            err_msg="Forward fp doesn't match Zarr"
        )

    def test_reverse_kinetics_match_zarr(self, pos_zarr, pos_shards, config):
        """Reverse view should have ri/rp from reverse-indexed positions, flipped."""
        read_data, feats = get_first_read(pos_zarr)
        feature_idx = {f: feats.index(f) for f in ['seq', 'fi', 'fp', 'ri', 'rp']}

        seq = read_data[:, feature_idx['seq']]
        ri = read_data[:, feature_idx['ri']]
        rp = read_data[:, feature_idx['rp']]

        cpg_iter = find_cpg_positions(seq, CONTEXT)
        first = next(cpg_iter, None)
        if first is None:
            pytest.skip("No CpG in first read")
        _, _, rev_start, rev_end = first

        # Legacy: ri[L-win_end:L-win_start] then flip
        expected_ri = ri[rev_start:rev_end][::-1].astype(np.float16)
        expected_rp = rp[rev_start:rev_end][::-1].astype(np.float16)

        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            all_data = load_all_shards(os.path.join(pos_shards, "val"))
        if len(all_data) < 2:
            pytest.skip("Need at least 2 windows")

        rev_row = all_data[1]  # second row = reverse view
        np.testing.assert_array_equal(
            rev_row[:, 1], expected_ri,
            err_msg="Reverse ri doesn't match expected"
        )
        np.testing.assert_array_equal(
            rev_row[:, 2], expected_rp,
            err_msg="Reverse rp doesn't match expected"
        )

    def test_reverse_seq_is_rc_of_forward(self, pos_shards, config):
        """Reverse view sequence should be the reverse complement of forward."""
        token_map = config['data']['token_map']
        rc_map = config['data']['rc_map']
        rc_lookup = build_rc_lookup(token_map, rc_map)

        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) < 2:
            pytest.skip("Need at least 2 windows")

        for i in range(0, min(20, len(all_data)), 2):
            fwd_seq = all_data[i, :, 0].astype(np.uint8)
            rev_seq = all_data[i + 1, :, 0].astype(np.uint8)
            expected_rc = rc_lookup[fwd_seq][::-1]
            np.testing.assert_array_equal(
                rev_seq, expected_rc,
                err_msg=f"Rev seq at pair {i//2} is not RC of fwd"
            )


# ---------------------------------------------------------------------------
# Stage 3: Shard integrity
# ---------------------------------------------------------------------------

class TestShardIntegrity:
    """Verify shard format and metadata."""

    def test_shard_shape(self, pos_shards):
        """Each shard should have shape (N, context, 4)."""
        paths = sorted(glob.glob(os.path.join(pos_shards, "train", "shard_*.npy")))
        if not paths:
            pytest.skip("No shards")
        for p in paths:
            arr = np.load(p)
            assert arr.ndim == 3, f"Expected 3D, got {arr.ndim}D"
            assert arr.shape[1] == CONTEXT, f"Expected context={CONTEXT}, got {arr.shape[1]}"
            assert arr.shape[2] == 4, f"Expected 4 features, got {arr.shape[2]}"

    def test_mask_is_zero(self, pos_shards):
        """Mask channel (column 3) should be 0.0 for all positions."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("No data")
        assert (all_data[:, :, 3] == 0.0).all(), "Mask channel has non-zero values"

    def test_no_nan_inf(self, pos_shards):
        """No NaN or Inf values in shards."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("No data")
        assert np.isfinite(all_data).all(), "Found NaN or Inf in shards"

    def test_seq_tokens_valid(self, pos_shards):
        """Sequence tokens should be in {0, 1, 2, 3, 4}."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("No data")
        seq = all_data[:, :, 0]
        assert seq.min() >= 0.0, f"Seq token below 0: {seq.min()}"
        assert seq.max() <= 4.0, f"Seq token above 4: {seq.max()}"

    def test_schema_written(self, pos_shards):
        """Schema JSON should exist in each split dir."""
        for split in ("train", "val"):
            schema_path = os.path.join(pos_shards, split, "schema.json")
            assert os.path.exists(schema_path), f"Missing {schema_path}"

    def test_train_val_no_read_overlap(self, pos_zarr, pos_shards, config):
        """Train and val should have no overlapping reads.

        We check this indirectly: count windows per split and verify
        they sum to the total (no duplication).
        """
        root = zarr.open(pos_zarr, mode='r')
        indptr = root['indptr'][:]
        z_data = root['data']
        feats = root.attrs['features']
        seq_idx = feats.index('seq')

        total_cpg = 0
        for r in range(len(indptr) - 1):
            start, end = int(indptr[r]), int(indptr[r + 1])
            seq = z_data[start:end, seq_idx]
            total_cpg += sum(1 for _ in find_cpg_positions(seq, CONTEXT))

        train_n = len(load_all_shards(os.path.join(pos_shards, "train"))) if \
            load_all_shards(os.path.join(pos_shards, "train")).ndim == 3 else 0
        val_n = len(load_all_shards(os.path.join(pos_shards, "val"))) if \
            load_all_shards(os.path.join(pos_shards, "val")).ndim == 3 else 0

        assert train_n + val_n == 2 * total_cpg, (
            f"Train ({train_n}) + val ({val_n}) != 2 * total CpGs ({total_cpg})"
        )


# ---------------------------------------------------------------------------
# Stage 4: DataLoader integration
# ---------------------------------------------------------------------------

class TestDataLoaderIntegration:
    """Verify LabeledMemmapDataset loads shards correctly."""

    def test_dataset_loads(self, pos_shards, neg_shards):
        """LabeledMemmapDataset should instantiate without error."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(pos_shards, "train"),
            neg_dir=os.path.join(neg_shards, "train"),
        )
        assert len(ds) > 0, "Dataset is empty"

    def test_sample_shape(self, pos_shards, neg_shards):
        """Each sample should be (context, 4) float32 with a scalar label."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(pos_shards, "train"),
            neg_dir=os.path.join(neg_shards, "train"),
        )
        x, y = ds[0]
        assert x.shape == (CONTEXT, 4), f"Expected ({CONTEXT}, 4), got {x.shape}"
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert y.item() in (0.0, 1.0)

    def test_model_input_channels(self, pos_shards, neg_shards):
        """Verify the model's expected channel layout:
        x[...,0] = seq tokens, x[...,1:3] = kinetics, x[...,3] = mask."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(pos_shards, "train"),
            neg_dir=os.path.join(neg_shards, "train"),
        )
        x, _ = ds[0]

        # Seq tokens: integers 0-4
        seq = x[:, 0]
        assert seq.min() >= 0.0 and seq.max() <= 4.0, "Seq out of range"
        assert (seq == seq.int().float()).all(), "Seq tokens not integer-valued"

        # Kinetics: non-negative (raw uint8 values)
        kin = x[:, 1:3]
        assert kin.min() >= 0.0, "Negative kinetics"
        assert kin.max() <= 255.0, "Kinetics above uint8 range"

        # Mask: all zero
        mask = x[:, 3]
        assert (mask == 0.0).all(), "Mask not zero"

    def test_positive_label_is_one(self, pos_shards, neg_shards):
        """First pos_len samples should have label 1.0."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(pos_shards, "train"),
            neg_dir=os.path.join(neg_shards, "train"),
        )
        _, y = ds[0]
        assert y.item() == 1.0, f"First sample label should be 1.0, got {y.item()}"

    def test_negative_label_is_zero(self, pos_shards, neg_shards):
        """Samples after pos_len should have label 0.0."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(pos_shards, "train"),
            neg_dir=os.path.join(neg_shards, "train"),
        )
        _, y = ds[ds.pos_len]  # first negative sample
        assert y.item() == 0.0, f"First neg sample label should be 0.0, got {y.item()}"
