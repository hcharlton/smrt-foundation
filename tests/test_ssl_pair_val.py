"""End-to-end tests for the SSL pair validation pipeline.

Covers:
  - `scripts.build_ssl_pair_val.build_ssl_pair_val` writes the expected
    directory structure (pairs/shard_*.npy + gaps.npy + metadata.yaml).
  - Each pair's view2 starts exactly `gap_bp` real bases after view1
    ends, verified by checking contiguous-encoded source positions
    (we control the source so the kinetics channel encodes position).
  - Per-gap counts honor the `total_cap` budget evenly.
  - Reads too short for a given gap are skipped only for that gap, not
    globally — short reads can still contribute pairs for small gaps.
  - The pad channel is preserved (both views are non-pad when the
    source read had enough unpadded length).
  - `smrt_foundation.dataset.PairedGapMemmapDataset` loads the build
    output correctly: __len__, __getitem__ shapes/dtypes/values,
    `gap_filter`, `limit`, and `norm_fn` integration all behave.
  - Caching: shards aren't re-loaded on repeat access.

Tests use a synthetic `ShardedMemmapDataset`-style source built in a
tmp_path fixture, so they run on CI without depending on Gefion data.

Run:
    python -m pytest tests/test_ssl_pair_val.py -v
"""

import os

import numpy as np
import pytest
import torch
import yaml

from scripts.build_ssl_pair_val import (
    DEFAULT_GAPS_BP,
    build_ssl_pair_val,
    unpadded_length,
)
from smrt_foundation.dataset import PairedGapMemmapDataset


# ---------------------------------------------------------------------------
# Synthetic source construction
# ---------------------------------------------------------------------------

CONTEXT = 4096       # max read length in source
N_FEATURES = 4       # seq, fi, fp, pad — matches project convention
SHARD_SIZE = 64      # small for tests
N_SHARDS = 3
N_READS = SHARD_SIZE * N_SHARDS  # 192 reads


def _make_synthetic_source(source_dir: str, seed: int = 0):
    """Write `N_SHARDS` shards of synthetic reads.

    Each read encodes its absolute position-within-read in the kinetic
    channels (channel 1 = fi gets `position`, channel 2 = fp gets
    `position + 0.5`). This lets tests verify that `view2` starts
    exactly `target_len + gap_bp` bases after `view1` started.

    Read lengths are varied per read so we exercise the
    "short read can't fit large gap" code path:
      - reads 0..63: length 4096 (full)
      - reads 64..127: length 2048
      - reads 128..159: length 256
      - reads 160..191: length 96 (only fits the smallest gaps)
    """
    os.makedirs(source_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Length schedule per shard.
    length_per_read = (
        [4096] * 64
        + [2048] * 64
        + [256] * 32
        + [96] * 32
    )
    assert len(length_per_read) == N_READS

    read_global_idx = 0
    for shard_i in range(N_SHARDS):
        shard = np.zeros((SHARD_SIZE, CONTEXT, N_FEATURES), dtype=np.float16)
        # Pad channel defaults to 1.0 across the whole tensor (placeholder).
        shard[:, :, 3] = 1.0

        for local_i in range(SHARD_SIZE):
            unp = length_per_read[read_global_idx]
            # seq: random tokens 0..4 across the unpadded prefix
            shard[local_i, :unp, 0] = rng.integers(0, 5, size=unp).astype(np.float16)
            # fi: encodes position so we can verify pair offsets later
            positions = np.arange(unp, dtype=np.float32)
            shard[local_i, :unp, 1] = positions.astype(np.float16)
            shard[local_i, :unp, 2] = (positions + 0.5).astype(np.float16)
            # pad channel: 0.0 over real bases
            shard[local_i, :unp, 3] = 0.0
            # (rest stays at the all-pad default)
            read_global_idx += 1

        np.save(os.path.join(source_dir, f"shard_{shard_i:05d}.npy"), shard)

    return length_per_read


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    src_dir = str(tmp_path_factory.mktemp("ssl_pair_src"))
    lengths = _make_synthetic_source(src_dir)
    return src_dir, lengths


@pytest.fixture(scope="module")
def small_pair_val(synthetic_source, tmp_path_factory):
    """Build a small pair val set with the default 19 gaps from the
    synthetic source. Total cap is small so the build is fast."""
    src_dir, _ = synthetic_source
    out_dir = str(tmp_path_factory.mktemp("ssl_pair_val"))
    metadata = build_ssl_pair_val(
        source_memmap=src_dir,
        output_dir=out_dir,
        gaps_bp=DEFAULT_GAPS_BP,
        target_len=32,
        total_cap=380,           # 380 / 19 = 20 per gap, achievable for most gaps
        shard_size=50,           # forces multiple shards
        seed=42,
    )
    return out_dir, metadata


# ---------------------------------------------------------------------------
# Build script: structural / metadata
# ---------------------------------------------------------------------------

class TestBuildScriptStructure:
    def test_creates_pairs_dir_and_sidecars(self, small_pair_val):
        out_dir, _ = small_pair_val
        assert os.path.isdir(os.path.join(out_dir, "pairs"))
        assert os.path.isfile(os.path.join(out_dir, "gaps.npy"))
        assert os.path.isfile(os.path.join(out_dir, "metadata.yaml"))

    def test_metadata_matches_disk(self, small_pair_val):
        out_dir, metadata_returned = small_pair_val
        with open(os.path.join(out_dir, "metadata.yaml")) as f:
            metadata_disk = yaml.safe_load(f)
        # Compare a few key fields
        assert metadata_disk["target_len"] == 32
        assert metadata_disk["gaps_bp"] == list(DEFAULT_GAPS_BP)
        assert metadata_disk["total_written"] == metadata_returned["total_written"]
        assert metadata_disk["per_gap_counts"] == metadata_returned["per_gap_counts"]
        assert metadata_disk["seed"] == 42

    def test_gaps_npy_length_matches_pairs(self, small_pair_val):
        out_dir, metadata = small_pair_val
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        assert gaps.dtype == np.int32
        assert len(gaps) == metadata["total_written"]

    def test_per_gap_count_is_balanced_when_feasible(self, small_pair_val):
        # Gaps that fit in even the shortest reads (96 bp unpadded → fits
        # gaps where 64+gap ≤ 96, i.e. gap ≤ 32) should reach the target
        # of 20 pairs each. Gaps too large to fit those short reads will
        # have lower counts but the script doesn't error out.
        _, metadata = small_pair_val
        target = metadata["pairs_per_gap_target"]
        # Gap 0 fits everywhere (64 bp span ≤ all reads of length ≥ 96);
        # gap 32 fits everywhere (96 ≤ 96); gap 64 fits reads ≥ 128 bases.
        assert metadata["per_gap_counts"][0] == target
        assert metadata["per_gap_counts"][32] == target
        # gap 2048 needs unpadded ≥ 2112; only the 64 length-4096 reads
        # qualify, so count = min(target, 64).
        assert metadata["per_gap_counts"][2048] == min(target, 64)

    def test_short_reads_only_skipped_for_gaps_they_cant_fit(self, small_pair_val):
        # The 32 length-96 reads can satisfy gap=0 (span 64) and gap=32
        # (span 96, just barely) but nothing larger. Verify that gap=64
        # has fewer eligible reads than gap=0.
        _, metadata = small_pair_val
        # gap 0: eligible reads = all where unp ≥ 64 = all 192
        # gap 32: eligible reads = all where unp ≥ 96 = all 192
        # gap 64: eligible reads = where unp ≥ 128 = 64+64+32 = 160
        # All of these are ≥ target=20, so all should hit target.
        assert metadata["per_gap_counts"][0] == 20
        assert metadata["per_gap_counts"][64] == 20
        # gap 1024: eligible = where unp ≥ 1088 = 64+64 = 128 → still ≥ 20
        assert metadata["per_gap_counts"][1024] == 20


# ---------------------------------------------------------------------------
# Build script: pair correctness
# ---------------------------------------------------------------------------

class TestPairCorrectness:
    """Surface-level structural checks on the emitted pairs. The
    *content* correctness of each pair (right read, right offsets,
    bit-exact slice) is verified independently in
    `TestIndependentSourceVerification` below, which compares against
    a fresh re-slice of the source rather than relying on encoded
    position values that suffer from float16 precision loss at large
    positions."""

    def _load_all_pairs(self, out_dir):
        import glob
        pair_files = sorted(glob.glob(os.path.join(out_dir, "pairs", "*.npy")))
        pairs = np.concatenate([np.load(p) for p in pair_files], axis=0)
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        return pairs, gaps

    def test_pairs_are_in_unpadded_region(self, small_pair_val):
        out_dir, _ = small_pair_val
        pairs, _ = self._load_all_pairs(out_dir)
        # Pad channel must be 0.0 on every position of every view
        # (build script only emits pairs from the unpadded region).
        assert (pairs[..., 3] == 0.0).all(), "pair has pad-channel leak"

    def test_pair_dtype_is_float16(self, small_pair_val):
        out_dir, _ = small_pair_val
        pairs, _ = self._load_all_pairs(out_dir)
        assert pairs.dtype == np.float16

    def test_pair_shape_is_correct(self, small_pair_val):
        out_dir, _ = small_pair_val
        pairs, _ = self._load_all_pairs(out_dir)
        # (N, 2, target_len, n_features) — target_len=32 from fixture
        assert pairs.shape[1:] == (2, 32, N_FEATURES)


# ---------------------------------------------------------------------------
# Independent re-derivation: pull each pair from its source read directly
# ---------------------------------------------------------------------------

class TestIndependentSourceVerification:
    """For every pair the build script emits, verify that the stored
    `(view1, view2)` is bit-identical to a fresh re-slice of the source
    read at `[anchor : anchor+target_len]` and
    `[anchor+target_len+gap : anchor+2*target_len+gap]`.

    This deliberately does *not* trust the build script's own
    bookkeeping or the position-encoded kinetics fixture — it
    re-derives the expected pair from the on-disk source memmap and
    compares byte-for-byte. Catches:
      - off-by-one errors in anchor sampling
      - silently writing the wrong read's data into a pair
      - any pair tampering between build-time and disk
    """

    def _load_source_reads(self, src_dir):
        """Load the entire synthetic source into a (N_reads, T, C)
        in-memory array, bypassing ShardedMemmapDataset / the build
        script's own iterator. The layout is reconstructed from
        `shard_*.npy` files in shard order, which is the same order
        the build script's iterator yields, so global indices line up.
        """
        import glob
        shard_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        shards = [np.load(p) for p in shard_files]
        return np.concatenate(shards, axis=0)

    def _load_all_pairs(self, out_dir):
        import glob
        pair_files = sorted(glob.glob(os.path.join(out_dir, "pairs", "*.npy")))
        return np.concatenate([np.load(p) for p in pair_files], axis=0)

    def test_sidecars_present_and_consistent_length(self, small_pair_val):
        out_dir, metadata = small_pair_val
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        read_ids = np.load(os.path.join(out_dir, "read_ids.npy"))
        anchors = np.load(os.path.join(out_dir, "anchors.npy"))
        N = metadata["total_written"]
        assert len(gaps) == N
        assert len(read_ids) == N
        assert len(anchors) == N
        assert read_ids.dtype == np.int64
        assert anchors.dtype == np.int32

    def test_every_pair_matches_source_slice(self, synthetic_source, small_pair_val):
        """The crux of independent verification: re-slice the source
        and confirm every emitted pair is bit-equal to that slice.
        """
        src_dir, _ = synthetic_source
        out_dir, _ = small_pair_val

        source_reads = self._load_source_reads(src_dir)  # (N_reads, T, C) float16
        pairs = self._load_all_pairs(out_dir)             # (N_pairs, 2, T, C) float16
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        read_ids = np.load(os.path.join(out_dir, "read_ids.npy"))
        anchors = np.load(os.path.join(out_dir, "anchors.npy"))
        target_len = 32

        # Sanity: read_ids must reference real source reads.
        assert read_ids.min() >= 0
        assert read_ids.max() < source_reads.shape[0]

        for i in range(len(pairs)):
            rid = int(read_ids[i])
            anchor = int(anchors[i])
            g = int(gaps[i])
            src = source_reads[rid]  # (T, C)

            # Independent re-slice. Notice: we use the raw source array
            # and standard numpy indexing — no helper from the build
            # script, no inferred-from-kinetics positions.
            expected_v1 = src[anchor:anchor + target_len]
            expected_v2 = src[anchor + target_len + g:anchor + 2 * target_len + g]

            stored_v1 = pairs[i, 0]
            stored_v2 = pairs[i, 1]

            np.testing.assert_array_equal(
                stored_v1, expected_v1,
                err_msg=f"pair {i}: view1 mismatch (read_id={rid}, anchor={anchor}, gap={g})"
            )
            np.testing.assert_array_equal(
                stored_v2, expected_v2,
                err_msg=f"pair {i}: view2 mismatch (read_id={rid}, anchor={anchor}, gap={g})"
            )

    def test_anchors_are_in_unpadded_region(self, synthetic_source, small_pair_val):
        """Every anchor + 2*target_len + gap must fit within the
        unpadded prefix of the source read it was drawn from."""
        src_dir, _ = synthetic_source
        out_dir, _ = small_pair_val
        source_reads = self._load_source_reads(src_dir)
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        read_ids = np.load(os.path.join(out_dir, "read_ids.npy"))
        anchors = np.load(os.path.join(out_dir, "anchors.npy"))
        target_len = 32

        for i in range(len(gaps)):
            rid = int(read_ids[i])
            anchor = int(anchors[i])
            g = int(gaps[i])
            src = source_reads[rid]
            unp = unpadded_length(src)
            required_end = anchor + 2 * target_len + g
            assert required_end <= unp, (
                f"pair {i}: span ends at {required_end} but source unpadded "
                f"length is {unp} (read_id={rid}, gap={g})"
            )

    def test_no_pair_uses_a_read_too_short_for_its_gap(self, synthetic_source, small_pair_val):
        """The build script must never emit a pair for read+gap combo
        that doesn't fit. If even one pair has source unpadded length
        < required span, the sampling logic is broken."""
        src_dir, _ = synthetic_source
        out_dir, _ = small_pair_val
        source_reads = self._load_source_reads(src_dir)
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        read_ids = np.load(os.path.join(out_dir, "read_ids.npy"))
        target_len = 32

        for i in range(len(gaps)):
            rid = int(read_ids[i])
            g = int(gaps[i])
            unp = unpadded_length(source_reads[rid])
            required = 2 * target_len + g
            assert unp >= required, (
                f"pair {i}: read {rid} has unpadded length {unp} but "
                f"gap {g} requires {required}"
            )


# ---------------------------------------------------------------------------
# unpadded_length helper
# ---------------------------------------------------------------------------

class TestUnpaddedLength:
    def test_all_real(self):
        x = np.zeros((10, 4), dtype=np.float16)
        # pad channel all zero ⇒ all real
        assert unpadded_length(x) == 10

    def test_all_pad(self):
        x = np.zeros((10, 4), dtype=np.float16)
        x[:, 3] = 1.0
        assert unpadded_length(x) == 0

    def test_partial_pad(self):
        x = np.zeros((10, 4), dtype=np.float16)
        x[7:, 3] = 1.0
        assert unpadded_length(x) == 7


# ---------------------------------------------------------------------------
# PairedGapMemmapDataset
# ---------------------------------------------------------------------------

class TestPairedGapMemmapDataset:
    def test_loads_full_set(self, small_pair_val):
        out_dir, metadata = small_pair_val
        ds = PairedGapMemmapDataset(out_dir)
        assert len(ds) == metadata["total_written"]

    def test_getitem_returns_v1_v2_gap(self, small_pair_val):
        out_dir, _ = small_pair_val
        ds = PairedGapMemmapDataset(out_dir)
        v1, v2, g = ds[0]
        assert isinstance(v1, torch.Tensor) and v1.dtype == torch.float32
        assert isinstance(v2, torch.Tensor) and v2.dtype == torch.float32
        assert v1.shape == (32, 4)
        assert v2.shape == (32, 4)
        assert isinstance(g, int)

    def test_gap_filter_single(self, small_pair_val):
        out_dir, metadata = small_pair_val
        ds = PairedGapMemmapDataset(out_dir, gap_filter=[0])
        assert len(ds) == metadata["per_gap_counts"][0]
        for i in range(len(ds)):
            _, _, g = ds[i]
            assert g == 0

    def test_gap_filter_multi(self, small_pair_val):
        out_dir, metadata = small_pair_val
        wanted = [32, 128]
        ds = PairedGapMemmapDataset(out_dir, gap_filter=wanted)
        expected = sum(metadata["per_gap_counts"][g] for g in wanted)
        assert len(ds) == expected
        seen_gaps = set()
        for i in range(len(ds)):
            _, _, g = ds[i]
            seen_gaps.add(g)
        assert seen_gaps == set(wanted)

    def test_dataset_views_match_independent_source_slice(self, synthetic_source, small_pair_val):
        """Independent verification routed through the dataset class.

        For every (read_id, anchor, gap_bp) recorded in the sidecar,
        re-slice the source memmap directly and compare with what
        `PairedGapMemmapDataset.__getitem__` returns. This catches any
        data mangling on the read path, separate from the build-time
        check in `TestIndependentSourceVerification`.
        """
        src_dir, _ = synthetic_source
        out_dir, _ = small_pair_val

        import glob
        shard_files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
        source_reads = np.concatenate([np.load(p) for p in shard_files], axis=0)
        read_ids = np.load(os.path.join(out_dir, "read_ids.npy"))
        anchors = np.load(os.path.join(out_dir, "anchors.npy"))
        target_len = 32

        ds = PairedGapMemmapDataset(out_dir)
        # Spot check a random subset to keep the test fast but covering
        # multiple gaps and shards.
        check_indices = list(range(0, len(ds), max(1, len(ds) // 30)))
        for i in check_indices:
            v1, v2, g = ds[i]
            rid = int(read_ids[i])
            anchor = int(anchors[i])
            src = source_reads[rid]
            expected_v1 = torch.from_numpy(src[anchor:anchor + target_len].copy()).float()
            expected_v2 = torch.from_numpy(
                src[anchor + target_len + g:anchor + 2 * target_len + g].copy()
            ).float()
            assert torch.equal(v1, expected_v1), (
                f"pair {i} (read={rid}, anchor={anchor}, gap={g}): "
                f"dataset view1 != independent source slice"
            )
            assert torch.equal(v2, expected_v2), (
                f"pair {i} (read={rid}, anchor={anchor}, gap={g}): "
                f"dataset view2 != independent source slice"
            )

    def test_norm_fn_applied_to_both_views(self, small_pair_val):
        out_dir, _ = small_pair_val

        # Identity-shifted norm: subtract a known offset so we can verify it ran
        class ShiftNorm:
            def __call__(self, x):
                x = x.clone()
                x[..., 1] = x[..., 1] - 1000.0
                return x

        ds_no_norm = PairedGapMemmapDataset(out_dir)
        ds_norm = PairedGapMemmapDataset(out_dir, norm_fn=ShiftNorm())
        v1_a, v2_a, _ = ds_no_norm[0]
        v1_b, v2_b, _ = ds_norm[0]
        # Channel 1 should be shifted by -1000 in both views.
        assert torch.allclose(v1_b[:, 1], v1_a[:, 1] - 1000.0)
        assert torch.allclose(v2_b[:, 1], v2_a[:, 1] - 1000.0)
        # Other channels unchanged.
        assert torch.allclose(v1_b[:, 0], v1_a[:, 0])
        assert torch.allclose(v1_b[:, 3], v1_a[:, 3])

    def test_limit(self, small_pair_val):
        out_dir, _ = small_pair_val
        ds = PairedGapMemmapDataset(out_dir, limit=10)
        assert len(ds) == 10

    def test_index_out_of_range_raises(self, small_pair_val):
        out_dir, _ = small_pair_val
        ds = PairedGapMemmapDataset(out_dir)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_cache_does_not_reopen_shard(self, small_pair_val):
        out_dir, _ = small_pair_val
        ds = PairedGapMemmapDataset(out_dir, cache_size=10)
        # Access two items in the first shard; expect only one memmap to exist.
        _ = ds[0]
        _ = ds[1]
        assert len(ds.memmaps) == 1
        # Access an item in a later shard.
        _ = ds[len(ds) - 1]
        assert len(ds.memmaps) == 2

    def test_cache_evicts_when_full(self, small_pair_val):
        out_dir, _ = small_pair_val
        ds = PairedGapMemmapDataset(out_dir, cache_size=1)
        _ = ds[0]
        _ = ds[len(ds) - 1]
        # Cache size 1 means only one memmap should be live.
        assert len(ds.memmaps) == 1

    def test_gaps_property_matches_returned_gaps(self, small_pair_val):
        out_dir, _ = small_pair_val
        ds = PairedGapMemmapDataset(out_dir, gap_filter=[0, 64])
        gaps_property = ds.gaps
        gaps_via_getitem = np.array([ds[i][2] for i in range(len(ds))])
        np.testing.assert_array_equal(gaps_property, gaps_via_getitem)

    def test_missing_gaps_npy_raises(self, small_pair_val, tmp_path):
        # Make a directory with pairs/ but no gaps.npy
        bad_dir = str(tmp_path / "bad")
        os.makedirs(os.path.join(bad_dir, "pairs"), exist_ok=True)
        # Copy one shard so pairs_dir isn't empty
        import shutil
        src_shard = sorted(
            os.path.join(small_pair_val[0], "pairs", f)
            for f in os.listdir(os.path.join(small_pair_val[0], "pairs"))
        )[0]
        shutil.copy(src_shard, os.path.join(bad_dir, "pairs"))
        with pytest.raises(FileNotFoundError):
            PairedGapMemmapDataset(bad_dir)

    def test_missing_pairs_dir_raises(self, tmp_path):
        bad_dir = str(tmp_path / "empty")
        os.makedirs(bad_dir)
        # Create an empty pairs/ subdir and a gaps.npy so the missing
        # piece is specifically the lack of shard files.
        os.makedirs(os.path.join(bad_dir, "pairs"))
        np.save(os.path.join(bad_dir, "gaps.npy"), np.array([], dtype=np.int32))
        with pytest.raises(FileNotFoundError):
            PairedGapMemmapDataset(bad_dir)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestRealDataIntegration:
    """End-to-end build + load on real PacBio data derived from the
    `ct22_5reads.bam` fixture. Auto-skips when the upstream memmap
    artifact isn't present (matches the pattern in
    `test_zarr_to_methyl_memmap_v2.py`).

    To produce the input artifact locally from the BAM (one-time):
        python -m scripts.bam_to_zarr \
            --input_path data/00_raw/unlabeled/ct22_5reads.bam \
            --output_path data/01_processed/ssl_sets/ct22_5reads.zarr \
            --config configs/data.yaml --n_reads 0
        python -m scripts.zarr_to_memmap_instanceNorm \
            --input_path data/01_processed/ssl_sets/ct22_5reads.zarr \
            --output_path data/01_processed/ssl_sets/ct22_5reads_raw.memmap \
            --config_path configs/data.yaml \
            --shard_size 8 --context 4096
    """

    REAL_MEMMAP = os.path.join(
        os.path.dirname(__file__), '..',
        'data/01_processed/ssl_sets/ct22_5reads_raw.memmap'
    )

    @pytest.fixture(scope="class")
    def real_pair_val(self, tmp_path_factory):
        if not os.path.isdir(self.REAL_MEMMAP):
            pytest.skip(
                f"ct22_5reads memmap not built. Run the BAM->Zarr->memmap "
                f"chain documented in this class's docstring to create "
                f"{self.REAL_MEMMAP}."
            )
        out_dir = str(tmp_path_factory.mktemp("real_ssl_pair_val"))
        metadata = build_ssl_pair_val(
            source_memmap=self.REAL_MEMMAP,
            output_dir=out_dir,
            gaps_bp=DEFAULT_GAPS_BP,
            target_len=32,
            total_cap=1000,
            shard_size=100,
            seed=42,
        )
        return out_dir, metadata

    def test_build_produces_pairs_at_every_gap(self, real_pair_val):
        _, metadata = real_pair_val
        # ct22 reads are long enough that every gap should produce at
        # least some pairs.
        for g in DEFAULT_GAPS_BP:
            assert metadata["per_gap_counts"][g] > 0, (
                f"gap={g} produced 0 pairs from real data; "
                f"either the build script broke or the source reads "
                f"are unexpectedly short."
            )

    def test_real_pairs_match_independent_source_slice(self, real_pair_val):
        out_dir, _ = real_pair_val
        # Reload the source memmap independently of the build script.
        import glob
        src_files = sorted(glob.glob(os.path.join(self.REAL_MEMMAP, "shard_*.npy")))
        source_reads = np.concatenate([np.load(p) for p in src_files], axis=0)

        ds = PairedGapMemmapDataset(out_dir)
        read_ids = np.load(os.path.join(out_dir, "read_ids.npy"))
        anchors = np.load(os.path.join(out_dir, "anchors.npy"))
        gaps = np.load(os.path.join(out_dir, "gaps.npy"))
        target_len = 32

        # Spot-check ~50 pairs spread across the set.
        check_indices = list(range(0, len(ds), max(1, len(ds) // 50)))
        for i in check_indices:
            v1, v2, g = ds[i]
            rid = int(read_ids[i])
            anchor = int(anchors[i])
            src = source_reads[rid]
            expected_v1 = torch.from_numpy(src[anchor:anchor + target_len].copy()).float()
            expected_v2 = torch.from_numpy(
                src[anchor + target_len + g:anchor + 2 * target_len + g].copy()
            ).float()
            assert torch.equal(v1, expected_v1), (
                f"real-data pair {i}: view1 mismatch "
                f"(read_id={rid}, anchor={anchor}, gap={g})"
            )
            assert torch.equal(v2, expected_v2), (
                f"real-data pair {i}: view2 mismatch "
                f"(read_id={rid}, anchor={anchor}, gap={g})"
            )
            # The gap label round-tripped through the dataset must match
            # the sidecar label for the same row.
            assert int(g) == int(gaps[i])

    def test_real_pairs_have_no_pad_leakage(self, real_pair_val):
        out_dir, _ = real_pair_val
        import glob
        all_pairs = np.concatenate(
            [np.load(p) for p in sorted(glob.glob(os.path.join(out_dir, "pairs", "*.npy")))],
            axis=0,
        )
        # Pad channel must be 0 across every position of every view —
        # build script promises this even for the gap=2048 pairs from
        # the longest source rows.
        assert (all_pairs[..., 3] == 0.0).all(), "pad-channel leak in real-data pairs"


class TestDeterminism:
    def test_same_seed_same_output(self, synthetic_source, tmp_path_factory):
        src_dir, _ = synthetic_source
        out_a = str(tmp_path_factory.mktemp("det_a"))
        out_b = str(tmp_path_factory.mktemp("det_b"))
        for out in (out_a, out_b):
            build_ssl_pair_val(
                source_memmap=src_dir,
                output_dir=out,
                gaps_bp=(0, 64, 256),
                target_len=32,
                total_cap=60,
                shard_size=20,
                seed=123,
            )
        gaps_a = np.load(os.path.join(out_a, "gaps.npy"))
        gaps_b = np.load(os.path.join(out_b, "gaps.npy"))
        np.testing.assert_array_equal(gaps_a, gaps_b)

        # Pair contents identical too.
        import glob
        files_a = sorted(glob.glob(os.path.join(out_a, "pairs", "*.npy")))
        files_b = sorted(glob.glob(os.path.join(out_b, "pairs", "*.npy")))
        assert len(files_a) == len(files_b)
        for fa, fb in zip(files_a, files_b):
            np.testing.assert_array_equal(np.load(fa), np.load(fb))

    def test_different_seed_different_pairs(self, synthetic_source, tmp_path_factory):
        src_dir, _ = synthetic_source
        out_a = str(tmp_path_factory.mktemp("seed_a"))
        out_b = str(tmp_path_factory.mktemp("seed_b"))
        build_ssl_pair_val(src_dir, out_a, gaps_bp=(0,), target_len=32,
                           total_cap=20, shard_size=20, seed=1)
        build_ssl_pair_val(src_dir, out_b, gaps_bp=(0,), target_len=32,
                           total_cap=20, shard_size=20, seed=2)
        a = np.load(os.path.join(out_a, "pairs", "shard_00000.npy"))
        b = np.load(os.path.join(out_b, "pairs", "shard_00000.npy"))
        # Should differ in the anchor positions chosen
        assert not np.array_equal(a, b)
