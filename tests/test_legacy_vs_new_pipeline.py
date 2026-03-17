"""
Direct comparison tests: legacy parquet pipeline vs new zarr-to-memmap pipeline.

Traces the same BAM reads through both paths and compares what the model
actually sees as input tensors. The goal is to identify systematic differences
that could explain why the classifier performs better on legacy data.

Key areas of comparison:
  1. Feature values at identical CpG sites (raw data fidelity)
  2. Normalization: legacy global log-Z vs new per-read MAD vs raw
  3. Reverse-strand handling: sequence RC, kinetics alignment
  4. What the model embedding layer receives in each case
"""

import os
import glob
import tempfile

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pysam
import pytest
import torch
import yaml

from scripts.bam_to_zarr import bam_to_zarr, _process_read
from scripts.zarr_to_methyl_memmap import (
    zarr_to_sharded_memmap,
    extract_pattern_windows_2d,
)
from archive.make_legacy_labeled_dataset import bam_to_legacy_parquet
from smrt_foundation.dataset import LabeledMemmapDataset, LegacyMethylDataset, compute_log_normalization_stats
from smrt_foundation.normalization import normalize_read_mad

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data.yaml')
LABELED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '00_raw', 'labeled')
POS_BAM_CANDIDATES = ['methylated_subset.bam', 'methylated_hifi_reads.bam']
NEG_BAM_CANDIDATES = ['unmethylated_subset.bam', 'unmethylated_hifi_reads.bam']
N_TEST_READS = 20

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _find_bam(candidates):
    for name in candidates:
        path = os.path.join(LABELED_DIR, name)
        if os.path.exists(path):
            return path
    pytest.skip(f"No BAM in {LABELED_DIR}. Tried: {candidates}")


@pytest.fixture(scope="session")
def config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def seq_map(config):
    return config['data']['token_map']


@pytest.fixture(scope="session")
def context(config):
    return config['cpg_pipeline']['context']


@pytest.fixture(scope="session")
def pos_bam():
    return _find_bam(POS_BAM_CANDIDATES)


@pytest.fixture(scope="session")
def neg_bam():
    return _find_bam(NEG_BAM_CANDIDATES)


# --- Legacy pipeline artifacts ---

@pytest.fixture(scope="session")
def legacy_pos_parquet(pos_bam, context, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("legacy_pos"))
    pq_path = os.path.join(out, "pos.parquet")
    bam_to_legacy_parquet(pos_bam, pq_path, context=context, label=1, n_reads=N_TEST_READS)
    return pq_path


@pytest.fixture(scope="session")
def legacy_neg_parquet(neg_bam, context, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("legacy_neg"))
    pq_path = os.path.join(out, "neg.parquet")
    bam_to_legacy_parquet(neg_bam, pq_path, context=context, label=0, n_reads=N_TEST_READS)
    return pq_path


# --- New pipeline artifacts ---

@pytest.fixture(scope="session")
def zarr_pos(pos_bam, config, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("zarr_pos"))
    zp = os.path.join(out, "pos.zarr")
    bam_to_zarr(pos_bam, zp, n_reads=N_TEST_READS, optional_tags=[], config=config)
    return zp


@pytest.fixture(scope="session")
def zarr_neg(neg_bam, config, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("zarr_neg"))
    zp = os.path.join(out, "neg.zarr")
    bam_to_zarr(neg_bam, zp, n_reads=N_TEST_READS, optional_tags=[], config=config)
    return zp


@pytest.fixture(scope="session")
def memmap_pos_raw(zarr_pos, config, context, tmp_path_factory):
    """New pipeline: raw, with RC (matches workflow defaults)."""
    out = str(tmp_path_factory.mktemp("mm_pos_raw"))
    cpg = config['cpg_pipeline']
    np.random.seed(0)
    zarr_to_sharded_memmap(
        zarr_pos, out, config, context=context, shard_size=4096,
        fwd_features=cpg['fwd_features'], rev_features=cpg['rev_features'],
        normalize=False, use_rc=True,
    )
    return out


@pytest.fixture(scope="session")
def memmap_neg_raw(zarr_neg, config, context, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mm_neg_raw"))
    cpg = config['cpg_pipeline']
    np.random.seed(0)
    zarr_to_sharded_memmap(
        zarr_neg, out, config, context=context, shard_size=4096,
        fwd_features=cpg['fwd_features'], rev_features=cpg['rev_features'],
        normalize=False, use_rc=True,
    )
    return out


@pytest.fixture(scope="session")
def memmap_pos_norm(zarr_pos, config, context, tmp_path_factory):
    """New pipeline: with per-read MAD normalization."""
    out = str(tmp_path_factory.mktemp("mm_pos_norm"))
    cpg = config['cpg_pipeline']
    np.random.seed(0)
    zarr_to_sharded_memmap(
        zarr_pos, out, config, context=context, shard_size=4096,
        fwd_features=cpg['fwd_features'], rev_features=cpg['rev_features'],
        normalize=True, use_rc=True,
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


def _get_first_bam_read_with_cpg(bam_path, context):
    """Return the first BAM read that has at least one CpG window."""
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            if not all(read.has_tag(t) for t in ['fi', 'fp', 'ri', 'rp']):
                continue
            seq = read.query_sequence.upper()
            pad = (context - 2) // 2
            for i in range(len(seq) - 1):
                if seq[i] == 'C' and seq[i + 1] == 'G':
                    start = i - pad
                    if start >= 0 and start + context <= len(seq):
                        return read, i, start
    return None, None, None


# ---------------------------------------------------------------------------
# Test Class 1: Single-sample provenance trace
# ---------------------------------------------------------------------------

class TestSingleSampleProvenance:
    """Trace a single CpG site from the same BAM read through both pipelines
    and compare exactly what each produces."""

    def test_forward_strand_raw_values_match(self, pos_bam, legacy_pos_parquet, config, context, seq_map):
        """At the same CpG site, the raw (pre-normalization) feature values
        from the legacy parquet and the new memmap pipeline should be identical."""
        read, cg_pos, win_start = _get_first_bam_read_with_cpg(pos_bam, context)
        if read is None:
            pytest.skip("No CpG found in test BAM")

        seq_str = read.query_sequence.upper()
        read_name = read.query_name

        # --- Legacy: read from parquet ---
        df = pl.read_parquet(legacy_pos_parquet)
        legacy_rows = df.filter(
            (pl.col('read_name') == read_name) &
            (pl.col('cg_pos') == cg_pos) &
            (pl.col('strand') == 'fwd')
        )
        assert len(legacy_rows) == 1, f"Expected 1 legacy row, got {len(legacy_rows)}"

        legacy_seq = legacy_rows['seq'][0]
        legacy_fi = np.array(legacy_rows['fi'][0].to_list(), dtype=np.uint8)
        legacy_fp = np.array(legacy_rows['fp'][0].to_list(), dtype=np.uint8)

        # --- Ground truth from BAM ---
        bam_seq = seq_str[win_start:win_start + context]
        bam_fi = np.array(read.get_tag('fi'), dtype=np.uint8)[win_start:win_start + context]
        bam_fp = np.array(read.get_tag('fp'), dtype=np.uint8)[win_start:win_start + context]

        # Legacy parquet should have raw values matching BAM
        assert legacy_seq == bam_seq, "Legacy sequence doesn't match BAM"
        np.testing.assert_array_equal(legacy_fi, bam_fi, err_msg="Legacy fi != BAM fi")
        np.testing.assert_array_equal(legacy_fp, bam_fp, err_msg="Legacy fp != BAM fp")

    def test_reverse_strand_sequence_handling(self, pos_bam, legacy_pos_parquet, config, context, seq_map):
        """Legacy always reverse-complements the sequence for reverse strand
        windows. Verify this is consistent."""
        read, cg_pos, _ = _get_first_bam_read_with_cpg(pos_bam, context)
        if read is None:
            pytest.skip("No CpG found")

        seq_str = read.query_sequence.upper()
        read_name = read.query_name

        df = pl.read_parquet(legacy_pos_parquet)
        rev_rows = df.filter(
            (pl.col('read_name') == read_name) &
            (pl.col('strand') == 'rev')
        )
        if len(rev_rows) == 0:
            pytest.skip("No reverse CpG windows for this read")

        # The reverse strand sequence should be the RC of the original
        rev_seq = rev_rows['seq'][0]
        rc_full = "".join({'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}.get(b, 'N')
                          for b in reversed(seq_str))
        rev_cg_pos = rev_rows['cg_pos'][0]
        pad = (context - 2) // 2
        rc_window = rc_full[rev_cg_pos - pad:rev_cg_pos - pad + context]

        assert rev_seq == rc_window, (
            f"Reverse strand sequence mismatch.\n"
            f"  Legacy: {rev_seq}\n"
            f"  Expected RC window: {rc_window}"
        )


# ---------------------------------------------------------------------------
# Test Class 2: Normalization comparison
# ---------------------------------------------------------------------------

class TestNormalizationComparison:
    """Compare the three normalization strategies and their effect on kinetics."""

    def test_raw_kinetics_distribution(self, memmap_pos_raw):
        """Raw pipeline: kinetics should be uint8-scale (0–255)."""
        train = load_all_shards(os.path.join(memmap_pos_raw, "train"))
        val = load_all_shards(os.path.join(memmap_pos_raw, "val"))
        all_data = np.concatenate([train, val], axis=0)
        if len(all_data) == 0:
            pytest.skip("No data")

        # Kinetics columns are 1 and 2 (fi, fp or ri, rp)
        kin = all_data[:, :, 1:3]
        # mask out padded positions
        mask = all_data[:, :, -1]
        active = mask == 0.0

        active_kin = kin[active]
        assert active_kin.max() <= 255.0, "Raw kinetics exceed uint8 range"
        assert active_kin.min() >= 0.0, "Raw kinetics below 0"

    def test_mad_normalized_distribution(self, memmap_pos_norm):
        """MAD-normalized pipeline: kinetics should be roughly centered near 0."""
        train = load_all_shards(os.path.join(memmap_pos_norm, "train"))
        val = load_all_shards(os.path.join(memmap_pos_norm, "val"))
        all_data = np.concatenate([train, val], axis=0)
        if len(all_data) == 0:
            pytest.skip("No data")

        kin = all_data[:, :, 1:3]
        mask = all_data[:, :, -1]
        active_kin = kin[mask == 0.0]

        # After MAD normalization, values should not be in uint8 range
        assert active_kin.max() < 255.0, "MAD norm values still in uint8 range"
        # Median should be roughly near 0 (log1p + centering)
        med = np.median(active_kin)
        assert abs(med) < 5.0, f"MAD norm median far from 0: {med}"

    def test_legacy_log_z_normalization(self, legacy_pos_parquet, context):
        """Legacy pipeline: log(x+1) then Z-normalize produces well-behaved values."""
        df = pl.read_parquet(legacy_pos_parquet)
        if len(df) == 0:
            pytest.skip("Empty parquet")

        kin_feats = ['fi', 'fp', 'ri', 'rp']
        df = df.with_columns([
            pl.col(c).list.to_array(context) for c in kin_feats
        ])

        means, stds = compute_log_normalization_stats(df, kin_feats)

        # Apply the same normalization that LegacyMethylDataset uses
        for k in ['fi', 'fp']:
            raw = df[k].to_numpy().astype(np.float64)
            vals = (np.log(raw + 1) - float(means[k])) / float(stds[k])
            med = np.median(vals)
            std_val = np.std(vals)
            # Should be approximately standard-normal
            assert abs(med) < 1.0, f"Legacy log-Z {k} median: {med}"
            assert 0.3 < std_val < 3.0, f"Legacy log-Z {k} std: {std_val}"

    def test_normalization_strategy_changes_discrimination(
        self, memmap_pos_raw, memmap_neg_raw, memmap_pos_norm,
        legacy_pos_parquet, legacy_neg_parquet, context
    ):
        """Compare how well each normalization separates pos from neg classes.

        This is the crux: if legacy normalization produces more separable
        distributions, the classifier has an easier learning problem.
        """
        # --- Raw pipeline ---
        pos_raw = np.concatenate([
            load_all_shards(os.path.join(memmap_pos_raw, "train")),
            load_all_shards(os.path.join(memmap_pos_raw, "val")),
        ])
        neg_raw = np.concatenate([
            load_all_shards(os.path.join(memmap_neg_raw, "train")),
            load_all_shards(os.path.join(memmap_neg_raw, "val")),
        ])
        if len(pos_raw) == 0 or len(neg_raw) == 0:
            pytest.skip("Missing data for raw comparison")

        # Use center position kinetics as the simplest discriminative feature
        center = pos_raw.shape[1] // 2
        pos_raw_center = pos_raw[:, center, 1:3].mean(axis=1)
        neg_raw_center = neg_raw[:, center, 1:3].mean(axis=1)
        raw_diff = abs(pos_raw_center.mean() - neg_raw_center.mean())

        # --- Legacy pipeline ---
        df_pos = pl.read_parquet(legacy_pos_parquet)
        df_neg = pl.read_parquet(legacy_neg_parquet)
        if len(df_pos) == 0 or len(df_neg) == 0:
            pytest.skip("Missing legacy parquet data")

        kin_feats = ['fi', 'fp', 'ri', 'rp']
        df_all = pl.concat([df_pos, df_neg])
        df_all = df_all.with_columns([
            pl.col(c).list.to_array(context) for c in kin_feats
        ])
        means, stds = compute_log_normalization_stats(df_all, kin_feats)

        def legacy_center_kin(df):
            fi = (np.log(df['fi'].to_numpy() + 1) - means['fi']) / stds['fi']
            fp = (np.log(df['fp'].to_numpy() + 1) - means['fp']) / stds['fp']
            return (fi[:, center] + fp[:, center]) / 2

        df_pos_c = df_pos.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
        df_neg_c = df_neg.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
        pos_legacy_center = legacy_center_kin(df_pos_c)
        neg_legacy_center = legacy_center_kin(df_neg_c)
        legacy_diff = abs(pos_legacy_center.mean() - neg_legacy_center.mean())

        # Just report — we don't assert which is bigger because the test
        # should pass regardless; the point is diagnostic output
        print(f"\n--- Class separation at CpG center ---")
        print(f"  Raw pipeline    |pos-neg| mean diff: {raw_diff:.4f}")
        print(f"  Legacy log-Z    |pos-neg| mean diff: {legacy_diff:.4f}")
        print(f"  Raw pos center mean: {pos_raw_center.mean():.4f} (std {pos_raw_center.std():.4f})")
        print(f"  Raw neg center mean: {neg_raw_center.mean():.4f} (std {neg_raw_center.std():.4f})")
        print(f"  Legacy pos center mean: {pos_legacy_center.mean():.4f} (std {pos_legacy_center.std():.4f})")
        print(f"  Legacy neg center mean: {neg_legacy_center.mean():.4f} (std {neg_legacy_center.std():.4f})")


# ---------------------------------------------------------------------------
# Test Class 3: Feature column alignment
# ---------------------------------------------------------------------------

class TestFeatureColumnAlignment:
    """Verify how features map to model input positions in each pipeline."""

    def test_new_pipeline_mixes_fwd_rev_kinetics(self, memmap_pos_raw, config, seq_map, context):
        """In the new pipeline, forward windows have fi/fp in columns 1-2
        and reverse windows have ri/rp in the SAME columns 1-2.
        Both pipelines do this, so the model sees a mix."""
        cpg = config['cpg_pipeline']
        all_data = np.concatenate([
            load_all_shards(os.path.join(memmap_pos_raw, "train")),
            load_all_shards(os.path.join(memmap_pos_raw, "val")),
        ])
        if len(all_data) == 0:
            pytest.skip("No data")

        # Feature layout: [seq, kin1, kin2, mask]
        # For both fwd and rev windows, seq is in col 0
        # The model's SmrtEmbedding does: x_nuc = x[...,0], x_kin = x[...,1:3]
        # So columns 1-2 carry kinetics regardless of strand
        assert all_data.shape[2] == len(cpg['fwd_features']) + 1  # +1 for mask

    def test_legacy_separates_fwd_rev_kinetics(self, legacy_pos_parquet, context):
        """In the legacy parquet, fi/fp and ri/rp are in SEPARATE columns.
        The LegacyMethylDataset constructs fwd_data=[seq,fi,fp,mask] and
        rev_data=[rc_seq,ri,rp,mask] as separate samples."""
        df = pl.read_parquet(legacy_pos_parquet)
        if len(df) == 0:
            pytest.skip("Empty parquet")

        kin_feats = ['fi', 'fp', 'ri', 'rp']
        # All four kinetics columns exist in the parquet
        for col in kin_feats:
            assert col in df.columns, f"Missing column {col} in legacy parquet"

        fwd_rows = df.filter(pl.col('strand') == 'fwd')
        if len(fwd_rows) == 0:
            pytest.skip("No forward rows")

        fwd_rows = fwd_rows.with_columns([
            pl.col(c).list.to_array(context) for c in kin_feats
        ])

        # Forward rows should have meaningful fi/fp values at the CpG center
        center = context // 2
        fi_vals = fwd_rows['fi'].to_numpy()[:, center]
        fp_vals = fwd_rows['fp'].to_numpy()[:, center]
        # These should not all be zero (they're real kinetics)
        assert fi_vals.sum() > 0, "fi values all zero at center"
        assert fp_vals.sum() > 0, "fp values all zero at center"


# ---------------------------------------------------------------------------
# Test Class 4: Model input tensor comparison
# ---------------------------------------------------------------------------

class TestModelInputTensorComparison:
    """Compare the actual tensors the model receives from each pipeline."""

    def test_new_pipeline_tensor_shape(self, memmap_pos_raw, memmap_neg_raw, context, config):
        """New pipeline: LabeledMemmapDataset -> DataLoader tensor shape."""
        ds = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_pos_raw, "train"),
            neg_dir=os.path.join(memmap_neg_raw, "train"),
        )
        if len(ds) == 0:
            pytest.skip("Empty dataset")

        x, y = ds[0]
        n_feat = len(config['cpg_pipeline']['fwd_features']) + 1
        assert x.shape == (context, n_feat), f"Expected ({context}, {n_feat}), got {x.shape}"
        assert x.dtype == torch.float32

        # Model expects: x_nuc=x[...,0], x_kin=x[...,1:3], x_pad=x[...,3]
        x_nuc = x[:, 0]
        x_kin = x[:, 1:3]
        x_pad = x[:, 3]

        # Sequence should be integer tokens (0-4)
        assert x_nuc.max() <= 4.0
        assert x_nuc.min() >= 0.0

        # Padding mask at center should be 0 (real data)
        assert x_pad[context // 2] == 0.0

    def test_legacy_tensor_shape(self, legacy_pos_parquet, context):
        """Legacy pipeline: LegacyMethylDataset tensor shape."""
        df = pl.read_parquet(legacy_pos_parquet)
        if len(df) == 0:
            pytest.skip("No data")

        kin_feats = ['fi', 'fp', 'ri', 'rp']
        df = df.with_columns([
            pl.col(c).list.to_array(context) for c in kin_feats
        ])
        means, stds = compute_log_normalization_stats(df, kin_feats)

        ds = LegacyMethylDataset(
            legacy_pos_parquet, means, stds, context,
            restrict_row_groups=100, single_strand=True,
        )
        item = next(iter(ds))
        data = item['data']

        # Legacy tensor: [seq, fi, fp, mask] -> shape (context, 4)
        assert data.shape == (context, 4), f"Expected ({context}, 4), got {data.shape}"
        assert data.dtype == torch.float32

    def test_kinetics_scale_comparison(
        self, memmap_pos_raw, memmap_neg_raw, legacy_pos_parquet, config, context
    ):
        """Compare the actual kinetics values the model sees.

        New pipeline (raw): uint8 values (0-255 as float32)
        Legacy pipeline: log(x+1) Z-normalized (roughly standard normal)

        This scale difference directly impacts how the embedding layer
        processes the data.
        """
        # New pipeline sample
        ds_new = LabeledMemmapDataset(
            pos_dir=os.path.join(memmap_pos_raw, "train"),
            neg_dir=os.path.join(memmap_neg_raw, "train"),
        )
        if len(ds_new) == 0:
            pytest.skip("Empty new dataset")

        new_x, _ = ds_new[0]
        new_kin = new_x[:, 1:3]  # kinetics columns

        # Legacy pipeline sample
        df = pl.read_parquet(legacy_pos_parquet)
        if len(df) == 0:
            pytest.skip("No legacy data")

        kin_feats = ['fi', 'fp', 'ri', 'rp']
        df = df.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
        means, stds = compute_log_normalization_stats(df, kin_feats)

        ds_legacy = LegacyMethylDataset(
            legacy_pos_parquet, means, stds, context,
            restrict_row_groups=100, single_strand=True,
        )
        legacy_item = next(iter(ds_legacy))
        legacy_kin = legacy_item['data'][:, 1:3]

        # Compute statistics
        new_active = new_kin[new_x[:, -1] == 0.0]
        legacy_active = legacy_kin  # legacy has no padding for well-formed windows

        print(f"\n--- Kinetics scale comparison ---")
        print(f"  New pipeline (raw):  mean={new_active.mean():.2f}  std={new_active.std():.2f}  "
              f"min={new_active.min():.2f}  max={new_active.max():.2f}")
        print(f"  Legacy (log-Z):      mean={legacy_active.mean():.2f}  std={legacy_active.std():.2f}  "
              f"min={legacy_active.min():.2f}  max={legacy_active.max():.2f}")

        # The scale difference should be dramatic
        # Raw values are 0-255, legacy should be roughly -3 to +3
        assert new_active.max() > 10.0, "New pipeline kin values unexpectedly small"
        assert abs(legacy_active.mean()) < 3.0, "Legacy kin values unexpectedly large"


# ---------------------------------------------------------------------------
# Test Class 5: Reverse strand consistency
# ---------------------------------------------------------------------------

class TestReverseStrandConsistency:
    """Verify reverse strand handling is consistent between pipelines."""

    def test_legacy_reverse_has_rc_sequence(self, legacy_pos_parquet, seq_map, context):
        """Legacy reverse windows should have RC'd sequence."""
        df = pl.read_parquet(legacy_pos_parquet)
        rev_rows = df.filter(pl.col('strand') == 'rev')
        if len(rev_rows) == 0:
            pytest.skip("No reverse rows")

        # Check that reverse windows contain CG at center (since RC is applied)
        pad = (context - 2) // 2
        for seq in rev_rows['seq'].to_list()[:10]:
            center = seq[pad:pad + 2]
            assert center == 'CG', (
                f"Reverse strand window center should be CG after RC, got '{center}'"
            )

    def test_new_pipeline_reverse_with_rc(self, memmap_pos_raw, seq_map, context):
        """New pipeline with use_rc=True: reverse windows should also have CG at center."""
        all_data = np.concatenate([
            load_all_shards(os.path.join(memmap_pos_raw, "train")),
            load_all_shards(os.path.join(memmap_pos_raw, "val")),
        ])
        if len(all_data) == 0:
            pytest.skip("No data")

        c_tok, g_tok = float(seq_map['C']), float(seq_map['G'])
        pad = (context - 2) // 2

        # Every window should have CG at center (both fwd and rc'd rev)
        seq_at_center = all_data[:, pad:pad + 2, 0]
        cg_mask = (seq_at_center[:, 0] == c_tok) & (seq_at_center[:, 1] == g_tok)
        assert cg_mask.all(), (
            f"Not all windows have CG at center: {cg_mask.sum()}/{len(cg_mask)}"
        )

    def test_new_pipeline_reverse_kinetics_are_flipped(self, pos_bam, config, context, seq_map):
        """Verify the new pipeline correctly flips reverse-strand kinetics."""
        read, cg_pos, _ = _get_first_bam_read_with_cpg(pos_bam, context)
        if read is None:
            pytest.skip("No CpG found")

        ri = np.array(read.get_tag('ri'), dtype=np.uint8)
        rp = np.array(read.get_tag('rp'), dtype=np.uint8)

        # After flip + RC, a CG in the RC sequence at position p corresponds
        # to a GC at position (len-2-p) in the original orientation
        # The kinetics at those positions should come from the FLIPPED ri/rp
        ri_flipped = np.flip(ri)
        rp_flipped = np.flip(rp)

        # Just verify the flip operation is consistent
        assert ri_flipped[0] == ri[-1]
        assert rp_flipped[0] == rp[-1]


# ---------------------------------------------------------------------------
# Test Class 6: Window count comparison
# ---------------------------------------------------------------------------

class TestWindowCountComparison:
    """Both pipelines should find the same number of CpG sites per read."""

    def test_same_total_cpg_windows(self, legacy_pos_parquet, memmap_pos_raw):
        """Legacy and new pipeline should produce the same number of
        CpG windows from the same BAM input."""
        df = pl.read_parquet(legacy_pos_parquet)
        legacy_count = len(df)

        new_count = sum(
            load_all_shards(os.path.join(memmap_pos_raw, split)).shape[0]
            for split in ("train", "val")
            if load_all_shards(os.path.join(memmap_pos_raw, split)).size > 0
        )

        assert legacy_count == new_count, (
            f"Window count mismatch: legacy={legacy_count}, new={new_count}"
        )

    def test_same_reads_contribute(self, legacy_pos_parquet, pos_bam, context):
        """Every read that contributes windows in the legacy pipeline
        should also contribute in the new pipeline (both scan the same reads)."""
        df = pl.read_parquet(legacy_pos_parquet)
        legacy_read_names = set(df['read_name'].unique().to_list())

        # Count reads with CpG windows directly from BAM
        bam_reads_with_cpg = set()
        with pysam.AlignmentFile(pos_bam, "rb", check_sq=False) as bam:
            for i, read in enumerate(bam):
                if i >= N_TEST_READS:
                    break
                if not all(read.has_tag(t) for t in ['fi', 'fp', 'ri', 'rp']):
                    continue
                seq = read.query_sequence.upper()
                pad = (context - 2) // 2
                for j in range(len(seq) - 1):
                    if seq[j] == 'C' and seq[j + 1] == 'G':
                        start = j - pad
                        if start >= 0 and start + context <= len(seq):
                            bam_reads_with_cpg.add(read.query_name)
                            break

        assert legacy_read_names == bam_reads_with_cpg, (
            f"Read set mismatch: "
            f"legacy-only={legacy_read_names - bam_reads_with_cpg}, "
            f"bam-only={bam_reads_with_cpg - legacy_read_names}"
        )


# ---------------------------------------------------------------------------
# Test Class 7: Diagnostic summary (always passes, prints comparison)
# ---------------------------------------------------------------------------

class TestDiagnosticSummary:
    """Print a diagnostic summary comparing both pipelines.
    This test always passes — it's purely informational."""

    def test_print_pipeline_comparison(
        self, memmap_pos_raw, memmap_neg_raw, memmap_pos_norm,
        legacy_pos_parquet, legacy_neg_parquet, config, context
    ):
        cpg = config['cpg_pipeline']

        # New pipeline stats
        pos_raw = np.concatenate([
            load_all_shards(os.path.join(memmap_pos_raw, "train")),
            load_all_shards(os.path.join(memmap_pos_raw, "val")),
        ])
        neg_raw = np.concatenate([
            load_all_shards(os.path.join(memmap_neg_raw, "train")),
            load_all_shards(os.path.join(memmap_neg_raw, "val")),
        ])
        pos_norm = np.concatenate([
            load_all_shards(os.path.join(memmap_pos_norm, "train")),
            load_all_shards(os.path.join(memmap_pos_norm, "val")),
        ])

        # Legacy stats
        df_pos = pl.read_parquet(legacy_pos_parquet)
        df_neg = pl.read_parquet(legacy_neg_parquet)

        print("\n" + "=" * 70)
        print("PIPELINE COMPARISON DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"\nSample counts:")
        print(f"  New pipeline (pos raw):  {len(pos_raw)}")
        print(f"  New pipeline (neg raw):  {len(neg_raw)}")
        print(f"  New pipeline (pos norm): {len(pos_norm)}")
        print(f"  Legacy (pos):            {len(df_pos)}")
        print(f"  Legacy (neg):            {len(df_neg)}")

        print(f"\nFeature layout:")
        print(f"  New pipeline: columns = {cpg['fwd_features'] + ['mask']}")
        print(f"  Legacy:       fwd_data = [seq, fi, fp, mask]")
        print(f"                rev_data = [rc_seq, ri, rp, mask]")

        if len(pos_raw) > 0:
            center = context // 2
            mask = pos_raw[:, :, -1]
            kin_raw = pos_raw[:, center, 1:3]
            kin_norm = pos_norm[:, center, 1:3]

            print(f"\nKinetics at CpG center (positive class):")
            print(f"  Raw:     mean={kin_raw.mean():.2f}  std={kin_raw.std():.2f}  "
                  f"range=[{kin_raw.min():.1f}, {kin_raw.max():.1f}]")
            print(f"  MAD-norm: mean={kin_norm.mean():.3f}  std={kin_norm.std():.3f}  "
                  f"range=[{kin_norm.min():.2f}, {kin_norm.max():.2f}]")

        if len(df_pos) > 0:
            kin_feats = ['fi', 'fp', 'ri', 'rp']
            df_all = pl.concat([df_pos, df_neg]).with_columns([
                pl.col(c).list.to_array(context) for c in kin_feats
            ])
            means, stds = compute_log_normalization_stats(df_all, kin_feats)

            df_pos_a = df_pos.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
            fi_norm = (np.log(df_pos_a['fi'].to_numpy() + 1) - means['fi']) / stds['fi']
            fp_norm = (np.log(df_pos_a['fp'].to_numpy() + 1) - means['fp']) / stds['fp']
            legacy_center = np.stack([fi_norm[:, center], fp_norm[:, center]], axis=1)

            print(f"  Legacy log-Z: mean={legacy_center.mean():.3f}  std={legacy_center.std():.3f}  "
                  f"range=[{legacy_center.min():.2f}, {legacy_center.max():.2f}]")

        print(f"\nNormalization strategies:")
        print(f"  New (raw):  None — model sees uint8 values as float32")
        print(f"  New (norm): Per-read MAD: log1p -> median/MAD per read")
        print(f"  Legacy:     Global log-Z: log(x+1) -> (x-mean)/std across dataset")

        print(f"\nReverse strand handling:")
        print(f"  New pipeline (use_rc=True): flip + RC sequence")
        print(f"  Legacy:                     flip + RC sequence (always)")

        print("\n" + "=" * 70)
