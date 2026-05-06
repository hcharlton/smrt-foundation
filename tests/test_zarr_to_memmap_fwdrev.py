"""
Tests for scripts.zarr_to_memmap_fwdrev (6-channel paired-kinetics SSL
memmap creation).

The most important assertion is the v2-fix regression check: at every
forward window [start, end) of a read of length L, the reverse kinetics
in the output must equal the source `ri[L-end:L-start][::-1]`. The
legacy `np.flip(read_rev[start:end])` indexing in
`zarr_to_memmap_instanceNorm.py` is wrong once fwd and rev are paired
channels; this test would catch a regression to that pattern.

Run:
    python -m pytest tests/test_zarr_to_memmap_fwdrev.py -v -s
"""

import glob
import json
import os

import numpy as np
import pytest
import yaml
import zarr

from scripts.zarr_to_memmap_fwdrev import zarr_to_memmap_fwdrev


# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data.yaml')
ZARR_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '01_processed', 'ssl_sets')
# Any zarr with seq/fi/fp/ri/rp works; ob007_500k is small and present locally.
ZARR_CANDIDATES = ['ob007_500k.zarr', 'ob007.zarr', 'yoran.zarr']
CONTEXT = 4096
SHARD_SIZE = 256
MAX_SHARDS = 1


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
def src_zarr_path():
    path = _find_zarr(ZARR_CANDIDATES)
    if path is None:
        pytest.skip(f"No suitable zarr in {ZARR_DIR}")
    return path


@pytest.fixture(scope="session")
def shards_dir(src_zarr_path, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("fwdrev_ssl"))
    zarr_to_memmap_fwdrev(
        zarr_path=src_zarr_path,
        output_dir=out,
        context=CONTEXT,
        shard_size=SHARD_SIZE,
        max_shards=MAX_SHARDS,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_first_shard(directory):
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    assert paths, f"no shards found in {directory}"
    return np.load(paths[0])


# ---------------------------------------------------------------------------
# Schema + shape
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_sidecar(self, shards_dir):
        with open(os.path.join(shards_dir, "schema.json")) as f:
            schema = json.load(f)
        assert schema['features'] == ['seq', 'fi', 'fp', 'ri', 'rp', 'mask']
        assert schema['pad_idx'] == 5
        assert schema['source'] == 'zarr_to_memmap_fwdrev'
        # The schema records the indexing convention for downstream readers.
        assert 'L-end:L-start' in schema['rev_kinetics_indexing']

    def test_shard_shape(self, shards_dir):
        shard = _load_first_shard(shards_dir)
        assert shard.shape == (SHARD_SIZE, CONTEXT, 6)
        assert shard.dtype == np.float16

    def test_seq_tokens_in_range(self, shards_dir):
        shard = _load_first_shard(shards_dir)
        # Tokens A=0, C=1, G=2, T=3, N=4 per data.yaml; shard padding is 0.
        assert set(np.unique(shard[..., 0]).astype(int)).issubset({0, 1, 2, 3, 4})

    def test_mask_is_binary(self, shards_dir):
        shard = _load_first_shard(shards_dir)
        assert set(np.unique(shard[..., 5]).astype(int)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Kinetics fidelity (forward + the v2 fix on reverse)
# ---------------------------------------------------------------------------

class TestKineticsFidelity:
    """Cross-check the first row of the first shard against the source zarr."""

    @pytest.fixture(scope="class")
    def first_row_with_source(self, src_zarr_path, shards_dir):
        shard = _load_first_shard(shards_dir)
        root = zarr.open(src_zarr_path, mode='r')
        indptr = root['indptr'][:]
        feats = list(root.attrs['features'])

        r = 0
        r_start = int(indptr[r])
        r_end = int(indptr[r + 1])
        L = r_end - r_start
        read_data = root['data'][r_start:r_end, :]
        return {
            'row': shard[0],
            'seq': read_data[:, feats.index('seq')].astype(np.float32),
            'fi': read_data[:, feats.index('fi')].astype(np.float32),
            'fp': read_data[:, feats.index('fp')].astype(np.float32),
            'ri': read_data[:, feats.index('ri')].astype(np.float32),
            'rp': read_data[:, feats.index('rp')].astype(np.float32),
            'L': L,
        }

    def test_forward_kinetics_match(self, first_row_with_source):
        d = first_row_with_source
        seg_len = min(CONTEXT, d['L'])
        np.testing.assert_array_equal(
            d['row'][:seg_len, 1].astype(np.float32),
            d['fi'][:seg_len].astype(np.float16).astype(np.float32),
            err_msg="forward fi at output position t should equal source fi[t]",
        )
        np.testing.assert_array_equal(
            d['row'][:seg_len, 2].astype(np.float32),
            d['fp'][:seg_len].astype(np.float16).astype(np.float32),
            err_msg="forward fp at output position t should equal source fp[t]",
        )

    def test_reverse_kinetics_use_v2_fix(self, first_row_with_source):
        """ri channel at position t should equal ri_source[L-1-t] for t in
        [0, seg_len). This is the v2 alignment: the reverse-strand
        kinetic at forward position t lives at source index L-1-t.
        Equivalently for a window [start, end): row[:, 3] = ri[L-end:
        L-start][::-1]."""
        d = first_row_with_source
        seg_len = min(CONTEXT, d['L'])
        L = d['L']

        expected_ri = d['ri'][L - seg_len:L][::-1].astype(np.float16).astype(np.float32)
        expected_rp = d['rp'][L - seg_len:L][::-1].astype(np.float16).astype(np.float32)

        np.testing.assert_array_equal(
            d['row'][:seg_len, 3].astype(np.float32), expected_ri,
            err_msg="reverse ri must use v2 indexing: ri[L-end:L-start][::-1]",
        )
        np.testing.assert_array_equal(
            d['row'][:seg_len, 4].astype(np.float32), expected_rp,
            err_msg="reverse rp must use v2 indexing: rp[L-end:L-start][::-1]",
        )

    def test_reverse_kinetics_do_NOT_match_legacy_indexing(self, first_row_with_source):
        """Discriminating regression test: the legacy
        `np.flip(read_rev[start:end])` indexing would produce
        `ri[start:end][::-1]` instead of `ri[L-end:L-start][::-1]`. For
        a non-symmetric read these differ; assert the script is NOT
        doing the legacy thing."""
        d = first_row_with_source
        seg_len = min(CONTEXT, d['L'])
        if d['L'] <= seg_len:
            pytest.skip("read shorter than segment, can't distinguish indexings")

        wrong_ri = d['ri'][:seg_len][::-1].astype(np.float16).astype(np.float32)
        actual_ri = d['row'][:seg_len, 3].astype(np.float32)
        # If these match, the script regressed to the legacy bug.
        assert not np.allclose(actual_ri, wrong_ri), (
            "reverse ri matches the legacy ri[start:end][::-1] indexing — "
            "the v2 fix has regressed."
        )


# ---------------------------------------------------------------------------
# Pad mask placement
# ---------------------------------------------------------------------------

class TestPadMask:
    def test_pad_mask_layout(self, src_zarr_path, shards_dir):
        """For each row, mask[0:seg_len] == 0 (real) and
        mask[seg_len:] == 1 (pad), where seg_len is the actual data
        length. We check this by finding the first 1.0 in mask and
        asserting all values before it are 0.0 and all values at/after
        it are 1.0."""
        shard = _load_first_shard(shards_dir)
        for r in range(min(20, shard.shape[0])):
            mask = shard[r, :, 5]
            ones = np.where(mask == 1.0)[0]
            if len(ones) == 0:
                continue  # full row is real data
            first_pad = int(ones[0])
            assert (mask[:first_pad] == 0.0).all(), (
                f"row {r}: mask before first pad must be all 0.0"
            )
            assert (mask[first_pad:] == 1.0).all(), (
                f"row {r}: mask from first pad onwards must be all 1.0"
            )


# ---------------------------------------------------------------------------
# No RC duplication (one row per segment, not two)
# ---------------------------------------------------------------------------

class TestNoRcDuplication:
    def test_consecutive_rows_are_different_reads(self, shards_dir):
        """The legacy SSL pipeline writes 2 rows per read (forward +
        RC reverse). The fwdrev pipeline writes 1 row per segment, no
        RC duplication. So consecutive rows in the shard should
        generally come from different reads (different seq channels)."""
        shard = _load_first_shard(shards_dir)
        if shard.shape[0] < 2:
            pytest.skip("need at least 2 rows")
        # Compare the seq channel of row 0 and row 1. With the legacy
        # script these would be RC of each other (matching after RC);
        # with fwdrev they should be different reads / different seqs.
        seq0 = shard[0, :, 0]
        seq1 = shard[1, :, 0]
        # Tokens are in {0..4}, padding is 0. Probability that two
        # independent reads produce identical 4096-base seq channels is
        # vanishingly small.
        assert not np.array_equal(seq0, seq1), (
            "row 0 and row 1 have identical seq — looks like the script "
            "is duplicating each read (regressed to RC-augment behaviour)."
        )
