"""
End-to-end fidelity tests for the BAM -> Zarr -> CpG memmap pipeline.

Verifies that the kinetics features surrounding every CpG site in the
original BAM reads are faithfully transferred through the intermediate
Zarr representation and into the final sharded numpy output.

All pipeline parameters (context, features, etc.) are read from
configs/data.yaml so the tests stay correct when configuration changes.
"""

import os
import glob
import json
import tempfile

import numpy as np
import pysam
import torch
import yaml
import zarr
import pytest

from scripts.bam_to_zarr import bam_to_zarr, _process_read
from scripts.zarr_to_methyl_memmap import (
    zarr_to_sharded_memmap,
    extract_pattern_windows_2d,
)
from smrt_foundation.normalization import build_rc_lookup
from smrt_foundation.dataset import LabeledMemmapDataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data.yaml')
LABELED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '00_raw', 'labeled')

# Accepted BAM filenames in order of preference (subset first for speed)
POS_BAM_CANDIDATES = ['methylated_subset.bam', 'methylated_hifi_reads.bam']
NEG_BAM_CANDIDATES = ['unmethylated_subset.bam', 'unmethylated_hifi_reads.bam']

N_TEST_READS = 50

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _find_bam(candidates):
    """Return the first existing BAM path from a list of candidates."""
    for name in candidates:
        path = os.path.join(LABELED_DIR, name)
        if os.path.exists(path):
            return path
    pytest.skip(
        f"No BAM file found in {LABELED_DIR}. "
        f"Tried: {candidates}"
    )


@pytest.fixture(scope="session")
def config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def seq_map(config):
    return config['data']['token_map']


@pytest.fixture(scope="session")
def cpg_params(config):
    """Pipeline parameters from configs/data.yaml cpg_pipeline section."""
    return config['cpg_pipeline']


@pytest.fixture(scope="session")
def context(cpg_params):
    return cpg_params['context']


@pytest.fixture(scope="session")
def fwd_features(cpg_params):
    return cpg_params['fwd_features']


@pytest.fixture(scope="session")
def rev_features(cpg_params):
    return cpg_params['rev_features']


@pytest.fixture(scope="session")
def n_output_features(fwd_features):
    """Number of output feature columns (fwd features + mask channel)."""
    return len(fwd_features) + 1


@pytest.fixture(scope="session")
def sample_bam():
    """Path to methylated (positive) BAM."""
    return _find_bam(POS_BAM_CANDIDATES)


@pytest.fixture(scope="session")
def sample_bam_neg():
    """Path to unmethylated (negative) BAM."""
    return _find_bam(NEG_BAM_CANDIDATES)


@pytest.fixture(scope="session")
def zarr_dir(sample_bam, config, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("zarr"))
    zarr_path = os.path.join(out, "test.zarr")
    bam_to_zarr(
        bam_path=sample_bam,
        zarr_path=zarr_path,
        n_reads=N_TEST_READS,
        optional_tags=[],
        config=config,
    )
    return zarr_path


@pytest.fixture(scope="session")
def memmap_dir_raw(zarr_dir, config, context, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("memmap_raw"))
    np.random.seed(0)
    cpg = config['cpg_pipeline']
    zarr_to_sharded_memmap(
        zarr_path=zarr_dir,
        output_dir=out,
        config=config,
        context=context,
        shard_size=4096,
        fwd_features=cpg['fwd_features'],
        rev_features=cpg['rev_features'],
        normalize=False,
        use_rc=False,
    )
    return out


@pytest.fixture(scope="session")
def memmap_dir_norm(zarr_dir, config, context, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("memmap_norm"))
    np.random.seed(0)
    cpg = config['cpg_pipeline']
    zarr_to_sharded_memmap(
        zarr_path=zarr_dir,
        output_dir=out,
        config=config,
        context=context,
        shard_size=4096,
        fwd_features=cpg['fwd_features'],
        rev_features=cpg['rev_features'],
        normalize=True,
        use_rc=False,
    )
    return out


# --- Negative (unmethylated) pipeline for full dataloader test ---

@pytest.fixture(scope="session")
def zarr_dir_neg(sample_bam_neg, config, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("zarr_neg"))
    zarr_path = os.path.join(out, "test_neg.zarr")
    bam_to_zarr(
        bam_path=sample_bam_neg,
        zarr_path=zarr_path,
        n_reads=N_TEST_READS,
        optional_tags=[],
        config=config,
    )
    return zarr_path


@pytest.fixture(scope="session")
def memmap_dir_neg_raw(zarr_dir_neg, config, context, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("memmap_neg_raw"))
    np.random.seed(0)
    cpg = config['cpg_pipeline']
    zarr_to_sharded_memmap(
        zarr_path=zarr_dir_neg,
        output_dir=out,
        config=config,
        context=context,
        shard_size=4096,
        fwd_features=cpg['fwd_features'],
        rev_features=cpg['rev_features'],
        normalize=False,
        use_rc=False,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_shards(directory):
    """Load and concatenate all shard .npy files from a directory."""
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    if not paths:
        return np.empty((0,))
    return np.concatenate([np.load(p) for p in paths], axis=0)


def bam_reads_to_list(bam_path, n_reads, tags):
    """Extract the first n_reads valid reads from a BAM as dicts."""
    reads = []
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if n_reads and i >= n_reads:
                break
            rd = _process_read(read, tags)
            if rd is not None:
                reads.append(rd)
    return reads


def _collect_expected_windows_from_bam(bam_path, n_reads, config):
    """Extract all CpG windows (both strands) directly from a BAM file.

    Returns an (N, context, n_fwd_features) float32 array.
    Forward windows use fwd_features; reverse windows use rev_features
    on the flipped read.
    """
    seq_map = config['data']['token_map']
    cpg = config['cpg_pipeline']
    context = cpg['context']
    fwd_features = cpg['fwd_features']
    rev_features = cpg['rev_features']
    cpg_pattern = np.array([seq_map['C'], seq_map['G']])
    tags = ['seq', 'qual'] + sorted(config['data']['kinetics_features'])

    lookup = np.zeros(128, dtype=np.uint8)
    for base, val in seq_map.items():
        if len(base) == 1:
            lookup[ord(base)] = val

    n_feat = len(fwd_features)
    windows = []
    bam_reads = bam_reads_to_list(bam_path, n_reads, tags)
    for rd in bam_reads:
        seq_str = rd['data']['seq']
        seq_tokens = lookup[np.frombuffer(seq_str.upper().encode('ascii'), dtype=np.uint8)]

        # Forward
        fwd_arrays = []
        for f in fwd_features:
            if f == 'seq':
                fwd_arrays.append(seq_tokens)
            else:
                fwd_arrays.append(np.array(rd['data'][f], dtype=np.uint8))
        read_fwd = np.stack(fwd_arrays, axis=1).astype(np.float32)
        for w in extract_pattern_windows_2d(read_fwd, 0, context, cpg_pattern):
            windows.append(w)

        # Reverse (flipped)
        rev_arrays = []
        for f in rev_features:
            if f == 'seq':
                rev_arrays.append(seq_tokens)
            else:
                rev_arrays.append(np.array(rd['data'][f], dtype=np.uint8))
        read_rev = np.flip(np.stack(rev_arrays, axis=1).astype(np.float32), axis=0)
        for w in extract_pattern_windows_2d(read_rev, 0, context, cpg_pattern):
            windows.append(w)

    if not windows:
        return np.empty((0, context, n_feat), dtype=np.float32)
    return np.stack(windows)


def _sort_samples(arr):
    """Lexicographic sort of an (N, ...) array by flattened rows."""
    flat = arr.reshape(arr.shape[0], -1)
    return arr[np.lexsort(flat[:, ::-1].T)]


# ---------------------------------------------------------------------------
# Stage 1: BAM -> Zarr fidelity
# ---------------------------------------------------------------------------

class TestBamToZarr:
    """Verify that the Zarr store faithfully captures BAM read data."""

    def test_zarr_has_expected_arrays(self, zarr_dir):
        root = zarr.open(zarr_dir, mode='r')
        assert 'data' in root
        assert 'indptr' in root
        assert 'features' in root.attrs

    def test_read_count_matches(self, zarr_dir, sample_bam, config):
        root = zarr.open(zarr_dir, mode='r')
        indptr = root['indptr'][:]
        n_zarr_reads = len(indptr) - 1

        tags = ['seq', 'qual'] + sorted(config['data']['kinetics_features'])
        bam_reads = bam_reads_to_list(sample_bam, N_TEST_READS, tags)

        assert n_zarr_reads == len(bam_reads)

    def test_sequence_fidelity(self, zarr_dir, sample_bam, config):
        """Every base in the Zarr seq column matches the BAM sequence."""
        root = zarr.open(zarr_dir, mode='r')
        data = root['data'][:]
        indptr = root['indptr'][:]
        feats = root.attrs['features']
        seq_idx = feats.index('seq')
        seq_map = config['data']['token_map']
        inv_map = {v: k for k, v in seq_map.items()}

        tags = ['seq', 'qual'] + sorted(config['data']['kinetics_features'])
        bam_reads = bam_reads_to_list(sample_bam, N_TEST_READS, tags)

        for i, rd in enumerate(bam_reads):
            start, end = indptr[i], indptr[i + 1]
            zarr_seq = "".join(inv_map.get(int(x), 'N') for x in data[start:end, seq_idx])
            assert zarr_seq == rd['data']['seq'], f"Sequence mismatch in read {i}"

    def test_kinetics_fidelity(self, zarr_dir, sample_bam, config):
        """Kinetics tag values in the Zarr match the BAM exactly."""
        root = zarr.open(zarr_dir, mode='r')
        data = root['data'][:]
        indptr = root['indptr'][:]
        feats = root.attrs['features']

        kin_tags = sorted(config['data']['kinetics_features'])
        tags = ['seq', 'qual'] + kin_tags
        bam_reads = bam_reads_to_list(sample_bam, N_TEST_READS, tags)

        for tag in kin_tags:
            col = feats.index(tag)
            for i, rd in enumerate(bam_reads):
                start, end = indptr[i], indptr[i + 1]
                zarr_vals = data[start:end, col].astype(np.uint8)
                bam_vals = np.array(rd['data'][tag], dtype=np.uint8)
                np.testing.assert_array_equal(
                    zarr_vals, bam_vals,
                    err_msg=f"Tag '{tag}' mismatch in read {i}",
                )


# ---------------------------------------------------------------------------
# Stage 2: Zarr -> CpG memmap fidelity (raw, no normalization)
# ---------------------------------------------------------------------------

class TestZarrToMemmapRaw:
    """With normalization off, the CpG windows in the shards must contain
    the exact same uint8 feature values as the Zarr (cast to float16)."""

    def test_shards_exist(self, memmap_dir_raw):
        train_shards = glob.glob(os.path.join(memmap_dir_raw, "train", "shard_*.npy"))
        val_shards = glob.glob(os.path.join(memmap_dir_raw, "val", "shard_*.npy"))
        assert len(train_shards) + len(val_shards) > 0, "No shards produced"

    def test_schema_written(self, memmap_dir_raw):
        for split in ("train", "val"):
            schema_path = os.path.join(memmap_dir_raw, split, "schema.json")
            assert os.path.exists(schema_path)
            with open(schema_path) as f:
                schema = json.load(f)
            assert "features" in schema
            assert schema["features"][-1] == "mask"

    def test_shard_shape(self, memmap_dir_raw, context, n_output_features):
        for split in ("train", "val"):
            shards = sorted(glob.glob(os.path.join(memmap_dir_raw, split, "shard_*.npy")))
            for sp in shards:
                arr = np.load(sp)
                assert arr.ndim == 3
                assert arr.shape[1] == context, (
                    f"context dimension should be {context}, got {arr.shape[1]}"
                )
                assert arr.shape[2] == n_output_features, (
                    f"feature dimension should be {n_output_features}, got {arr.shape[2]}"
                )

    def test_mask_channel_is_zero_for_data(self, memmap_dir_raw):
        """The mask channel should be 0.0 for all positions with real data."""
        for split in ("train", "val"):
            samples = load_all_shards(os.path.join(memmap_dir_raw, split))
            if samples.size == 0:
                continue
            mask = samples[:, :, -1]
            centre = samples.shape[1] // 2
            assert np.all(mask[:, centre] == 0.0)

    def test_cpg_at_window_centre(self, memmap_dir_raw, seq_map, context):
        """Forward-strand windows should have C,G at the centre."""
        c_token, g_token = float(seq_map['C']), float(seq_map['G'])
        pad = (context - 2) // 2
        for split in ("train", "val"):
            samples = load_all_shards(os.path.join(memmap_dir_raw, split))
            if samples.size == 0:
                continue
            seq_col = 0
            centre_pairs = np.stack([samples[:, pad, seq_col],
                                     samples[:, pad + 1, seq_col]], axis=1)
            cg_mask = (centre_pairs[:, 0] == c_token) & (centre_pairs[:, 1] == g_token)
            assert cg_mask.sum() > 0, "No forward CpG windows found at centre"

    def test_raw_values_match_zarr(self, memmap_dir_raw, zarr_dir, config):
        """For every CpG window in the output, verify that the feature values
        around the CpG site match the original Zarr data (both strands)."""
        root = zarr.open(zarr_dir, mode='r')
        z_data = root['data'][:]
        indptr = root['indptr'][:]
        feats = root.attrs['features']
        seq_map = config['data']['token_map']
        cpg = config['cpg_pipeline']
        context = cpg['context']
        fwd_features = cpg['fwd_features']
        rev_features = cpg['rev_features']
        cpg_pattern = np.array([seq_map['C'], seq_map['G']])

        fwd_cols = [feats.index(f) for f in fwd_features]
        rev_cols = [feats.index(f) for f in rev_features]
        seq_col_fwd = fwd_features.index('seq')
        seq_col_rev = rev_features.index('seq')

        expected_windows = []
        total_reads = len(indptr) - 1
        for r in range(total_reads):
            start, end = indptr[r], indptr[r + 1]
            read_raw = z_data[start:end]

            read_fwd = read_raw[:, fwd_cols].astype(np.float32)
            fwd_wins = extract_pattern_windows_2d(read_fwd, seq_col_fwd, context, cpg_pattern)
            for w in fwd_wins:
                expected_windows.append(w.astype(np.float16))

            read_rev = np.flip(read_raw[:, rev_cols].astype(np.float32), axis=0)
            rev_wins = extract_pattern_windows_2d(read_rev, seq_col_rev, context, cpg_pattern)
            for w in rev_wins:
                expected_windows.append(w.astype(np.float16))

        if not expected_windows:
            pytest.skip("No CpG sites found in test data")

        expected = np.stack(expected_windows)

        n_fwd_feat = len(fwd_features)
        all_samples = np.concatenate([
            load_all_shards(os.path.join(memmap_dir_raw, "train")),
            load_all_shards(os.path.join(memmap_dir_raw, "val")),
        ], axis=0)[:, :, :n_fwd_feat]

        assert len(all_samples) == len(expected), (
            f"CpG window count mismatch: got {len(all_samples)}, expected {len(expected)}"
        )

        np.testing.assert_array_equal(
            _sort_samples(all_samples), _sort_samples(expected),
            err_msg="CpG windows differ between Zarr and memmap output",
        )


# ---------------------------------------------------------------------------
# Stage 2b: Zarr -> CpG memmap with normalization
# ---------------------------------------------------------------------------

class TestZarrToMemmapNormalized:
    """With normalization on, values should be transformed but structure preserved."""

    def test_shards_exist(self, memmap_dir_norm):
        train = glob.glob(os.path.join(memmap_dir_norm, "train", "shard_*.npy"))
        val = glob.glob(os.path.join(memmap_dir_norm, "val", "shard_*.npy"))
        assert len(train) + len(val) > 0

    def test_normalized_kinetics_differ_from_raw(self, memmap_dir_raw, memmap_dir_norm):
        """Normalization should actually change the kinetics values."""
        raw = np.concatenate([
            load_all_shards(os.path.join(memmap_dir_raw, "train")),
            load_all_shards(os.path.join(memmap_dir_raw, "val")),
        ])
        norm = np.concatenate([
            load_all_shards(os.path.join(memmap_dir_norm, "train")),
            load_all_shards(os.path.join(memmap_dir_norm, "val")),
        ])
        if raw.size == 0 or norm.size == 0:
            pytest.skip("No data to compare")

        assert not np.allclose(raw[:, :, 1], norm[:, :, 1], atol=1e-3), \
            "IPD values unchanged after normalization"

    def test_sequence_tokens_unchanged_by_normalization(self, memmap_dir_raw, memmap_dir_norm):
        """Sequence tokens (categorical) must not be altered by normalization."""
        raw = np.concatenate([
            load_all_shards(os.path.join(memmap_dir_raw, "train")),
            load_all_shards(os.path.join(memmap_dir_raw, "val")),
        ])
        norm = np.concatenate([
            load_all_shards(os.path.join(memmap_dir_norm, "train")),
            load_all_shards(os.path.join(memmap_dir_norm, "val")),
        ])
        if raw.size == 0 or norm.size == 0:
            pytest.skip("No data to compare")

        np.testing.assert_array_equal(
            raw[:, :, 0], norm[:, :, 0],
            err_msg="Normalization altered the sequence channel",
        )


# ---------------------------------------------------------------------------
# Stage 3: End-to-end BAM -> memmap CpG feature fidelity
# ---------------------------------------------------------------------------

class TestEndToEndFidelity:
    """Go directly from BAM reads to memmap output and verify that the
    kinetics features at CpG sites are faithfully preserved (raw mode)."""

    def test_bam_cpg_features_in_shards(self, sample_bam, memmap_dir_raw, config):
        """For each CpG site found in the BAM (both strands), confirm that
        the kinetics window written to the shard matches the original BAM
        tag data."""
        cpg = config['cpg_pipeline']
        n_fwd_feat = len(cpg['fwd_features'])

        expected = _collect_expected_windows_from_bam(sample_bam, N_TEST_READS, config)
        if len(expected) == 0:
            pytest.skip("No CpG sites in test BAM subset")

        all_samples = np.concatenate([
            load_all_shards(os.path.join(memmap_dir_raw, "train")),
            load_all_shards(os.path.join(memmap_dir_raw, "val")),
        ])[:, :, :n_fwd_feat]

        assert len(all_samples) == len(expected), (
            f"Window count mismatch: shard has {len(all_samples)}, BAM has {len(expected)}"
        )

        np.testing.assert_array_equal(
            _sort_samples(all_samples),
            _sort_samples(expected.astype(np.float16)),
            err_msg="End-to-end: CpG windows in shards do not match BAM features",
        )


# ---------------------------------------------------------------------------
# Unit tests for extract_pattern_windows_2d
# ---------------------------------------------------------------------------

class TestExtractPatternWindows:

    def test_basic_extraction(self):
        read = np.array([
            [0, 10, 20],
            [1, 11, 21],
            [2, 12, 22],
            [3, 13, 23],
        ], dtype=np.float32)
        pattern = [1, 2]  # CG
        windows = extract_pattern_windows_2d(read, 0, 4, pattern)
        assert windows.shape == (1, 4, 3)
        np.testing.assert_array_equal(windows[0], read)

    def test_no_match(self):
        read = np.array([[0, 10], [3, 13], [0, 10], [3, 13]], dtype=np.float32)
        windows = extract_pattern_windows_2d(read, 0, 4, [1, 2])
        assert windows.shape[0] == 0

    def test_boundary_exclusion(self):
        read = np.array([[1, 0], [2, 0], [3, 0], [0, 0]], dtype=np.float32)
        windows = extract_pattern_windows_2d(read, 0, 4, [1, 2])
        assert windows.shape[0] == 0

    def test_multiple_matches(self):
        read = np.zeros((20, 2), dtype=np.float32)
        read[5, 0] = 1; read[6, 0] = 2
        read[12, 0] = 1; read[13, 0] = 2
        windows = extract_pattern_windows_2d(read, 0, 4, [1, 2])
        assert windows.shape[0] == 2
        for w in windows:
            assert w[1, 0] == 1 and w[2, 0] == 2


# ---------------------------------------------------------------------------
# Stage 4: Full pipeline BAM -> Zarr -> memmap -> LabeledMemmapDataset -> DataLoader
# ---------------------------------------------------------------------------

class TestFullPipelineDataLoader:
    """BAM -> Zarr -> memmap -> LabeledMemmapDataset -> DataLoader.

    Verifies that every sample retrieved through the DataLoader (the exact
    interface used by training) matches the original BAM feature data.
    """

    def test_dataloader_labels(self, memmap_dir_raw, memmap_dir_neg_raw):
        """Positive samples should have label 1.0, negatives 0.0."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_dir_raw, "train"),
            neg_dir=os.path.join(memmap_dir_neg_raw, "train"),
        )
        assert len(ds) > 0

        pos_data, pos_label = ds[0]
        assert pos_label.item() == 1.0

        neg_data, neg_label = ds[ds.pos_len]
        assert neg_label.item() == 0.0

    def test_dataloader_shape(
        self, memmap_dir_raw, memmap_dir_neg_raw, context, n_output_features
    ):
        """Each sample from the DataLoader should have the expected shape."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_dir_raw, "train"),
            neg_dir=os.path.join(memmap_dir_neg_raw, "train"),
        )
        batch_data, batch_labels = next(iter(DataLoader(ds, batch_size=64)))
        assert batch_data.shape == (64, context, n_output_features)
        assert batch_labels.shape == (64,)
        assert batch_data.dtype == torch.float32

    def test_positive_samples_match_bam(
        self, sample_bam, memmap_dir_raw, memmap_dir_neg_raw, config
    ):
        """Every positive sample from the DataLoader matches a CpG window
        extracted directly from the methylated BAM."""
        n_fwd_feat = len(config['cpg_pipeline']['fwd_features'])
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_dir_raw, "train"),
            neg_dir=os.path.join(memmap_dir_neg_raw, "train"),
        )
        loader = DataLoader(ds, batch_size=ds.pos_len, shuffle=False)
        batch_data, batch_labels = next(iter(loader))

        pos_data = batch_data[:ds.pos_len]
        pos_labels = batch_labels[:ds.pos_len]
        assert torch.all(pos_labels == 1.0)

        pos_features = pos_data[:, :, :n_fwd_feat].numpy()

        expected_all = _collect_expected_windows_from_bam(
            sample_bam, N_TEST_READS, config
        )
        assert len(expected_all) > 0, "No CpG windows in methylated BAM"

        def to_row_set(arr):
            return set(map(bytes, arr.reshape(arr.shape[0], -1).astype(np.float32)))

        expected_set = to_row_set(expected_all)
        actual_set = to_row_set(pos_features)

        missing = actual_set - expected_set
        assert len(missing) == 0, (
            f"{len(missing)} positive DataLoader samples have no match in the BAM"
        )

    def test_negative_samples_match_bam(
        self, sample_bam_neg, memmap_dir_raw, memmap_dir_neg_raw, config
    ):
        """Every negative sample from the DataLoader matches a CpG window
        extracted directly from the unmethylated BAM."""
        n_fwd_feat = len(config['cpg_pipeline']['fwd_features'])
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_dir_raw, "train"),
            neg_dir=os.path.join(memmap_dir_neg_raw, "train"),
        )
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        batch_data, batch_labels = next(iter(loader))

        neg_data = batch_data[ds.pos_len:]
        neg_labels = batch_labels[ds.pos_len:]
        assert torch.all(neg_labels == 0.0)

        neg_features = neg_data[:, :, :n_fwd_feat].numpy()

        expected_all = _collect_expected_windows_from_bam(
            sample_bam_neg, N_TEST_READS, config
        )
        assert len(expected_all) > 0, "No CpG windows in unmethylated BAM"

        def to_row_set(arr):
            return set(map(bytes, arr.reshape(arr.shape[0], -1).astype(np.float32)))

        expected_set = to_row_set(expected_all)
        actual_set = to_row_set(neg_features)

        missing = actual_set - expected_set
        assert len(missing) == 0, (
            f"{len(missing)} negative DataLoader samples have no match in the BAM"
        )

    def test_full_batch_combined_counts(
        self, sample_bam, sample_bam_neg, memmap_dir_raw, memmap_dir_neg_raw, config
    ):
        """The total number of samples across both classes in the DataLoader
        should equal the number of CpG windows (train split) from both BAMs."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_dir_raw, "train"),
            neg_dir=os.path.join(memmap_dir_neg_raw, "train"),
        )
        ds_val = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_dir_raw, "val"),
            neg_dir=os.path.join(memmap_dir_neg_raw, "val"),
        )

        total_from_ds = len(ds) + len(ds_val)

        expected_pos = len(_collect_expected_windows_from_bam(
            sample_bam, N_TEST_READS, config
        ))
        expected_neg = len(_collect_expected_windows_from_bam(
            sample_bam_neg, N_TEST_READS, config
        ))

        assert total_from_ds == expected_pos + expected_neg, (
            f"Total samples across train+val ({total_from_ds}) != "
            f"expected CpG windows from BAMs ({expected_pos + expected_neg})"
        )
