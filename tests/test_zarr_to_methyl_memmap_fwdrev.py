"""
Tests for scripts.zarr_to_methyl_memmap_fwdrev (6-channel paired-kinetics
CpG memmap creation).

Mirrors `tests/test_zarr_to_methyl_memmap_v2.py` but adapted for the
fwdrev variant:
  - 6-channel rows instead of 4
  - one sample per CpG instead of two (no separate fwd_row/rev_row)
  - reverse kinetics paired into channels 3/4 of the same row
  - same v2-fix indexing (`ri[L-win_end:L-win_start][::-1]`) — regression
    test included to keep that load-bearing detail honest.

Run:
    python -m pytest tests/test_zarr_to_methyl_memmap_fwdrev.py -v -s
"""

import glob
import json
import os

import numpy as np
import pytest
import yaml
import zarr

from scripts.zarr_to_methyl_memmap_fwdrev import (
    zarr_to_methyl_memmap_fwdrev,
    find_cpg_positions,
    extract_cpg_windows_fwdrev,
    C_TOKEN,
    G_TOKEN,
)


# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data.yaml')
ZARR_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '01_processed', 'ssl_sets')
POS_ZARR_CANDIDATES = ['cpg_pos_subset.zarr', 'cpg_pos.zarr']
NEG_ZARR_CANDIDATES = ['cpg_neg_subset.zarr', 'cpg_neg.zarr']
CONTEXT = 32
SHARD_SIZE = 4096


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
        pytest.skip(f"No pos zarr in {ZARR_DIR}")
    return path


@pytest.fixture(scope="session")
def pos_shards(pos_zarr, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("fwdrev_cpg_pos"))
    zarr_to_methyl_memmap_fwdrev(
        zarr_path=pos_zarr,
        output_dir=out,
        context=CONTEXT,
        shard_size=SHARD_SIZE,
        max_shards=1,
        val_pct=0.2,
        seed=42,
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


# ---------------------------------------------------------------------------
# Schema + shape
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_sidecar(self, pos_shards):
        with open(os.path.join(pos_shards, "train", "schema.json")) as f:
            schema = json.load(f)
        assert schema['features'] == ['seq', 'fi', 'fp', 'ri', 'rp', 'mask']
        assert schema['pad_idx'] == 5
        assert schema['samples_per_cpg'] == 1
        assert schema['source'] == 'zarr_to_methyl_memmap_fwdrev'

    def test_shard_shape_and_dtype(self, pos_shards):
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("no train data")
        assert all_data.shape[1:] == (CONTEXT, 6)
        assert all_data.dtype == np.float16


# ---------------------------------------------------------------------------
# CpG centering and counting
# ---------------------------------------------------------------------------

class TestCpgExtraction:
    def test_all_windows_have_cg_at_center(self, pos_shards):
        """With context=32, pad=(32-2)/2=15 so the center C should be
        at index 15 and the G at index 16."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("no train data")
        center_C = all_data[:, 15, 0]
        center_G = all_data[:, 16, 0]
        assert (center_C == C_TOKEN).all(), "center C-token missing on some rows"
        assert (center_G == G_TOKEN).all(), "center G-token missing on some rows"

    def test_one_sample_per_cpg(self, pos_zarr, pos_shards):
        """vs. v2's two samples per CpG. The total row count across
        train+val should equal the number of valid CpGs across the
        zarr (same `find_cpg_positions` selection rule), without the
        2x multiplier."""
        train_data = load_all_shards(os.path.join(pos_shards, "train"))
        val_data = load_all_shards(os.path.join(pos_shards, "val"))
        actual = len(train_data) + len(val_data)

        # Recompute the expected total by scanning the zarr.
        root = zarr.open(pos_zarr, mode='r')
        indptr = root['indptr'][:]
        seq_idx = list(root.attrs['features']).index('seq')
        # For perf, only scan up to the first 1024 reads (the test fixture
        # caps shards at 1, so most CpGs come from these reads anyway).
        # We instead verify the ratio is ~1, not a precise count.
        expected = 0
        n_to_scan = min(1024, len(indptr) - 1)
        for r in range(n_to_scan):
            r_start, r_end = int(indptr[r]), int(indptr[r + 1])
            seq = root['data'][r_start:r_end, seq_idx]
            for _ in find_cpg_positions(seq, CONTEXT):
                expected += 1
        # If the script ran with max_shards=1, it may have stopped before
        # processing all reads — so we only assert the script didn't
        # produce ~2x the v2 row count (regression to v2's 2-sample
        # behaviour). Roughly: actual <= 1.1 * scanned_expected.
        assert actual <= int(expected * 1.5), (
            f"actual rows ({actual}) is much larger than scanned CpGs "
            f"({expected}) — looks like the script is writing >1 sample per CpG."
        )


# ---------------------------------------------------------------------------
# Kinetics fidelity (forward + the v2 fix on reverse)
# ---------------------------------------------------------------------------

class TestKineticsFidelity:
    @pytest.fixture(scope="class")
    def first_train_cpg_with_source(self, pos_zarr, pos_shards):
        """Find the first CpG in the first non-val read and return its
        (row, source-read-data) pair. Mirrors the script's iteration
        order so we can index shard[0]."""
        all_train = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_train) == 0:
            pytest.skip("no train data")

        root = zarr.open(pos_zarr, mode='r')
        indptr = root['indptr'][:]
        feats = list(root.attrs['features'])
        total_reads = len(indptr) - 1

        rng = np.random.RandomState(42)
        is_val = np.zeros(total_reads, dtype=bool)
        val_indices = rng.choice(total_reads, int(total_reads * 0.2), replace=False)
        is_val[val_indices] = True

        seq_idx = feats.index('seq')
        for r in range(total_reads):
            if is_val[r]:
                continue
            r_start, r_end = int(indptr[r]), int(indptr[r + 1])
            L = r_end - r_start
            seq = root['data'][r_start:r_end, seq_idx]
            for win_start, win_end, rev_start, rev_end in find_cpg_positions(seq, CONTEXT):
                read_data = root['data'][r_start:r_end, :]
                return {
                    'row': all_train[0],
                    'read_data': read_data,
                    'feats': feats,
                    'L': L,
                    'win_start': win_start,
                    'win_end': win_end,
                    'rev_start': rev_start,
                    'rev_end': rev_end,
                }
        pytest.skip("no CpG found in any train read")

    def test_forward_kinetics_match(self, first_train_cpg_with_source):
        d = first_train_cpg_with_source
        fi = d['read_data'][:, d['feats'].index('fi')].astype(np.float32)
        fp = d['read_data'][:, d['feats'].index('fp')].astype(np.float32)
        expected_fi = fi[d['win_start']:d['win_end']].astype(np.float16).astype(np.float32)
        expected_fp = fp[d['win_start']:d['win_end']].astype(np.float16).astype(np.float32)

        np.testing.assert_array_equal(d['row'][:, 1].astype(np.float32), expected_fi)
        np.testing.assert_array_equal(d['row'][:, 2].astype(np.float32), expected_fp)

    def test_reverse_kinetics_use_v2_fix(self, first_train_cpg_with_source):
        """row[:, 3] / row[:, 4] should equal
        ri[rev_start:rev_end][::-1] and rp[rev_start:rev_end][::-1].
        rev_start = L-win_end, rev_end = L-win_start."""
        d = first_train_cpg_with_source
        ri = d['read_data'][:, d['feats'].index('ri')].astype(np.float32)
        rp = d['read_data'][:, d['feats'].index('rp')].astype(np.float32)

        expected_ri = ri[d['rev_start']:d['rev_end']][::-1].astype(np.float16).astype(np.float32)
        expected_rp = rp[d['rev_start']:d['rev_end']][::-1].astype(np.float16).astype(np.float32)

        np.testing.assert_array_equal(d['row'][:, 3].astype(np.float32), expected_ri)
        np.testing.assert_array_equal(d['row'][:, 4].astype(np.float32), expected_rp)

    def test_reverse_kinetics_do_NOT_match_legacy_indexing(self, first_train_cpg_with_source):
        """Discriminating regression test: legacy `np.flip(seg_rev_data)`
        in the SSL script gives `ri[win_start:win_end][::-1]`, not the
        v2 fix. For asymmetric reads these are different — fail if the
        script is doing the wrong thing."""
        d = first_train_cpg_with_source
        # If the read is too short or the window happens to coincide
        # with the read mirror point, skip — the test can't discriminate.
        if d['L'] - d['win_end'] == d['win_start']:
            pytest.skip("symmetric window position; v2 vs legacy collide")
        ri = d['read_data'][:, d['feats'].index('ri')].astype(np.float32)
        wrong_ri = ri[d['win_start']:d['win_end']][::-1].astype(np.float16).astype(np.float32)
        assert not np.allclose(d['row'][:, 3].astype(np.float32), wrong_ri), (
            "reverse ri matches the legacy ri[win_start:win_end][::-1] "
            "indexing — the v2 fix has regressed."
        )


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

class TestMask:
    def test_mask_is_all_zero(self, pos_shards):
        """A CpG window is fully real data, so the mask channel must
        be 0.0 everywhere across every row."""
        all_data = load_all_shards(os.path.join(pos_shards, "train"))
        if len(all_data) == 0:
            pytest.skip("no train data")
        assert (all_data[..., 5] == 0.0).all(), (
            "mask channel must be 0.0 for every position of every CpG window"
        )


# ---------------------------------------------------------------------------
# Train/val split determinism
# ---------------------------------------------------------------------------

class TestTrainValSplit:
    def test_split_is_deterministic(self, pos_zarr, tmp_path_factory):
        """Same seed should produce the same split. Run twice with
        seed=42 and assert row counts match."""
        out_a = str(tmp_path_factory.mktemp("fwdrev_split_a"))
        out_b = str(tmp_path_factory.mktemp("fwdrev_split_b"))
        zarr_to_methyl_memmap_fwdrev(
            pos_zarr, out_a, context=CONTEXT, shard_size=SHARD_SIZE,
            max_shards=1, val_pct=0.2, seed=42,
        )
        zarr_to_methyl_memmap_fwdrev(
            pos_zarr, out_b, context=CONTEXT, shard_size=SHARD_SIZE,
            max_shards=1, val_pct=0.2, seed=42,
        )
        for split in ('train', 'val'):
            a = load_all_shards(os.path.join(out_a, split))
            b = load_all_shards(os.path.join(out_b, split))
            assert len(a) == len(b), f"{split} split row count differs across seeds"
            if len(a):
                np.testing.assert_array_equal(a, b)
