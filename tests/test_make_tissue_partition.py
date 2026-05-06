"""
Tests for scripts/make_tissue_partition.py.

The partition is the ground truth for every later evaluation. Any leakage
across splits silently inflates val accuracy. These tests prove:

  1. No read_name appears in more than one split (no leakage).
  2. train and val_<train_cell_short> reads come exclusively from train_cell.
  3. val_<heldout_cell_short> reads come exclusively from heldout_cell.
  4. The three split sets are pairwise disjoint.
  5. Each (split, tissue) bucket has exactly the requested count.
  6. Same seed -> bit-identical CSV output.
  7. Different seed -> different assignment (sanity).
  8. Capacity errors and config errors raise before any file is written.
  9. Existing partition.csv is not overwritten without --overwrite.
 10. (Smoke) Real yoran_ctx4096 data passes the same invariants.

Split names are derived from the trailing token of each cell ID
(`cell_str.split('_')[-1]`). For yoran cells `m84108_..._s1` and
`m84108_..._s2`, the splits are `train`, `val_s1`, `val_s2`.

The synthetic-manifest harness lets the rigorous unit tests run in
milliseconds; the real-data smoke test catches assumptions that only break
on actual data (cell_str format, per-tissue capacity).
"""

import json
import os

import polars as pl
import pytest

from scripts.make_tissue_partition import make_partition, write_partition


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REAL_YORAN_DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..',
    'data', '01_processed', 'tissue_sets', 'yoran_ctx4096',
)
REAL_TRAIN_CELL = 'm84108_251007_115244_s1'
REAL_HELDOUT_CELL = 'm84108_250930_153107_s2'
REAL_VAL_TRAIN_SPLIT = 'val_s1'
REAL_VAL_HELDOUT_SPLIT = 'val_s2'

SYNTH_TISSUES = ('t0', 't1', 't2')
SYNTH_TRAIN_CELL = 'synth_A'
SYNTH_HELDOUT_CELL = 'synth_B'
SYNTH_CELLS = (SYNTH_TRAIN_CELL, SYNTH_HELDOUT_CELL)
SYNTH_VAL_TRAIN_SPLIT = 'val_A'   # 'synth_A'.split('_')[-1] -> 'A'
SYNTH_VAL_HELDOUT_SPLIT = 'val_B'
SYNTH_ROWS_PER_BUCKET = 100   # rows per (tissue, cell)


# ---------------------------------------------------------------------------
# Synthetic manifest fixture
# ---------------------------------------------------------------------------


def _build_synthetic_manifest(out_dir):
    """Write a small synthetic manifest.parquet + schema.json under out_dir.

    Schema mirrors what scripts/bam_to_labeled_memmap.py produces, using
    only the columns make_tissue_partition reads (read_name, tissue_str,
    cell_str). Other columns are populated with placeholder values for
    schema fidelity but unused by the partition logic.
    """
    rows = []
    counter = 0
    for tissue in SYNTH_TISSUES:
        for cell in SYNTH_CELLS:
            for _ in range(SYNTH_ROWS_PER_BUCKET):
                rows.append({
                    'shard_idx': 0,
                    'row_idx': counter,
                    'read_name': f'{cell}/{counter}/ccs',
                    'tissue_str': tissue,
                    'cell_str': cell,
                    'tissue_id': SYNTH_TISSUES.index(tissue),
                    'cell_id': SYNTH_CELLS.index(cell),
                    'crop_start': 0,
                    'read_length': 4096,
                })
                counter += 1
    manifest = pl.DataFrame(rows, schema={
        'shard_idx': pl.UInt32,
        'row_idx': pl.UInt32,
        'read_name': pl.String,
        'tissue_str': pl.String,
        'cell_str': pl.String,
        'tissue_id': pl.Int32,
        'cell_id': pl.Int32,
        'crop_start': pl.UInt32,
        'read_length': pl.UInt32,
    })
    manifest.write_parquet(os.path.join(out_dir, 'manifest.parquet'))

    schema = {
        'tissue_to_id': {t: i for i, t in enumerate(SYNTH_TISSUES)},
        'cell_to_id': {c: i for i, c in enumerate(SYNTH_CELLS)},
        'context': 4096,
        'features': ['seq', 'fi', 'fp', 'ri', 'rp', 'mask'],
        'dtype': 'uint8',
    }
    with open(os.path.join(out_dir, 'schema.json'), 'w') as f:
        json.dump(schema, f)
    return out_dir


@pytest.fixture(scope='session')
def synthetic_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp('synthetic_tissue_dataset')
    return _build_synthetic_manifest(str(out))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_default(synthetic_data_dir, **overrides):
    """Run make_partition with conservative defaults that fit within the
    synthetic capacity (3 tissues x 2 cells x 100 rows/bucket).

    Defaults:
      train_cell=synth_A, heldout_cell=synth_B
      -> split names 'train', 'val_A', 'val_B'.
      train_per_tissue=10, val_train_cell_per_tissue=5,
      val_heldout_cell_per_tissue=5, seed=0.
    """
    kwargs = dict(
        data_dir=synthetic_data_dir,
        train_cell=SYNTH_TRAIN_CELL,
        heldout_cell=SYNTH_HELDOUT_CELL,
        train_per_tissue=10,
        val_train_cell_per_tissue=5,
        val_heldout_cell_per_tissue=5,
        seed=0,
    )
    kwargs.update(overrides)
    return make_partition(**kwargs)


# ---------------------------------------------------------------------------
# Leakage tests (the user's explicit requirement)
# ---------------------------------------------------------------------------


def test_no_read_leakage_across_splits(synthetic_data_dir):
    """Every read_name appears in at most one split."""
    p = _run_default(synthetic_data_dir)
    assert p['read_name'].n_unique() == p.height, (
        f"duplicate read_name across splits: "
        f"{p.height - p['read_name'].n_unique()} duplicates"
    )


def test_train_and_val_train_cell_only_from_train_cell(synthetic_data_dir):
    """No read in 'train' or val_<train_cell_short> originates from heldout_cell."""
    p = _run_default(synthetic_data_dir)
    from_train_cell = p.filter(
        pl.col('split').is_in(['train', SYNTH_VAL_TRAIN_SPLIT])
    )['read_name'].to_list()
    for name in from_train_cell:
        assert name.startswith(f'{SYNTH_TRAIN_CELL}/'), (
            f"name {name!r} in train/{SYNTH_VAL_TRAIN_SPLIT} must come from "
            f"train_cell={SYNTH_TRAIN_CELL}"
        )


def test_val_heldout_cell_only_from_heldout_cell(synthetic_data_dir):
    """val_<heldout_cell_short> reads come exclusively from heldout_cell."""
    p = _run_default(synthetic_data_dir)
    for name in p.filter(
        pl.col('split') == SYNTH_VAL_HELDOUT_SPLIT
    )['read_name'].to_list():
        assert name.startswith(f'{SYNTH_HELDOUT_CELL}/'), (
            f"name {name!r} in {SYNTH_VAL_HELDOUT_SPLIT} must come from "
            f"heldout_cell={SYNTH_HELDOUT_CELL}"
        )


def test_pairwise_split_disjoint(synthetic_data_dir):
    """All three splits are pairwise disjoint sets."""
    p = _run_default(synthetic_data_dir)
    splits = ('train', SYNTH_VAL_TRAIN_SPLIT, SYNTH_VAL_HELDOUT_SPLIT)
    sets = {
        s: set(p.filter(pl.col('split') == s)['read_name'].to_list())
        for s in splits
    }
    assert sets['train'].isdisjoint(sets[SYNTH_VAL_TRAIN_SPLIT])
    assert sets['train'].isdisjoint(sets[SYNTH_VAL_HELDOUT_SPLIT])
    assert sets[SYNTH_VAL_TRAIN_SPLIT].isdisjoint(sets[SYNTH_VAL_HELDOUT_SPLIT])


def test_split_names_match_cell_suffixes(synthetic_data_dir):
    """The val split names are derived from cell_str trailing tokens."""
    p = _run_default(synthetic_data_dir)
    expected = {'train', SYNTH_VAL_TRAIN_SPLIT, SYNTH_VAL_HELDOUT_SPLIT}
    assert set(p['split'].unique().to_list()) == expected


def test_every_partition_read_exists_in_manifest(synthetic_data_dir):
    """Every read_name in partition.csv corresponds to a real manifest row.
    Catches accidental string mutation."""
    p = _run_default(synthetic_data_dir)
    manifest = pl.read_parquet(
        os.path.join(synthetic_data_dir, 'manifest.parquet')
    )
    manifest_names = set(manifest['read_name'].to_list())
    partition_names = set(p['read_name'].to_list())
    assert partition_names.issubset(manifest_names)


# ---------------------------------------------------------------------------
# Stratification correctness
# ---------------------------------------------------------------------------


def test_per_split_per_tissue_counts_exact(synthetic_data_dir):
    """Each (split, tissue) bucket contains exactly the requested count."""
    p = _run_default(
        synthetic_data_dir,
        train_per_tissue=10,
        val_train_cell_per_tissue=5,
        val_heldout_cell_per_tissue=5,
    )
    manifest = pl.read_parquet(
        os.path.join(synthetic_data_dir, 'manifest.parquet')
    ).select(['read_name', 'tissue_str'])
    joined = p.join(manifest, on='read_name', how='left')
    counts = (
        joined.group_by(['split', 'tissue_str']).len()
        .sort(['split', 'tissue_str'])
    )
    expected = {('train', t): 10 for t in SYNTH_TISSUES}
    expected.update({(SYNTH_VAL_TRAIN_SPLIT, t): 5 for t in SYNTH_TISSUES})
    expected.update({(SYNTH_VAL_HELDOUT_SPLIT, t): 5 for t in SYNTH_TISSUES})
    for row in counts.iter_rows(named=True):
        key = (row['split'], row['tissue_str'])
        assert row['len'] == expected[key], (
            f"({row['split']}, {row['tissue_str']}): "
            f"got {row['len']} expected {expected[key]}"
        )
    seen = {(row['split'], row['tissue_str']) for row in counts.iter_rows(named=True)}
    assert seen == set(expected.keys())


def test_total_count_equals_sum_of_requested(synthetic_data_dir):
    """Total partition rows = n_tissues * (train + val_train_cell + val_heldout_cell)."""
    p = _run_default(
        synthetic_data_dir,
        train_per_tissue=7,
        val_train_cell_per_tissue=3,
        val_heldout_cell_per_tissue=4,
    )
    assert p.height == len(SYNTH_TISSUES) * (7 + 3 + 4)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seed_determinism_dataframe(synthetic_data_dir):
    """Same seed -> identical DataFrame rows in the same order."""
    p1 = _run_default(synthetic_data_dir, seed=42)
    p2 = _run_default(synthetic_data_dir, seed=42)
    assert p1.equals(p2)


def test_seed_determinism_csv_bytes(synthetic_data_dir, tmp_path):
    """Same seed -> bit-identical CSV file bytes."""
    p1 = _run_default(synthetic_data_dir, seed=42)
    p2 = _run_default(synthetic_data_dir, seed=42)
    out1 = tmp_path / 'a.csv'
    out2 = tmp_path / 'b.csv'
    write_partition(p1, str(out1))
    write_partition(p2, str(out2))
    assert out1.read_bytes() == out2.read_bytes()


def test_different_seed_changes_assignment(synthetic_data_dir):
    """Sanity: changing the seed actually changes the assignment of reads
    to splits (the set of read_names per split should differ)."""
    p1 = _run_default(synthetic_data_dir, seed=0)
    p2 = _run_default(synthetic_data_dir, seed=1)
    s1 = set(p1.filter(pl.col('split') == 'train')['read_name'].to_list())
    s2 = set(p2.filter(pl.col('split') == 'train')['read_name'].to_list())
    assert s1 != s2, 'train set is identical across seeds (RNG not seeded?)'


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_insufficient_train_cell_rows_raises(synthetic_data_dir):
    """train_per_tissue + val_train_cell_per_tissue exceeds capacity in train_cell."""
    with pytest.raises(ValueError, match='need .* have'):
        _run_default(
            synthetic_data_dir,
            train_per_tissue=80,
            val_train_cell_per_tissue=80,
            val_heldout_cell_per_tissue=5,
        )


def test_insufficient_heldout_cell_rows_raises(synthetic_data_dir):
    """val_heldout_cell_per_tissue exceeds capacity in heldout_cell."""
    with pytest.raises(ValueError, match='need .* have'):
        _run_default(
            synthetic_data_dir,
            val_heldout_cell_per_tissue=SYNTH_ROWS_PER_BUCKET + 1,
        )


def test_train_cell_eq_heldout_cell_raises(synthetic_data_dir):
    with pytest.raises(AssertionError, match='must differ'):
        _run_default(
            synthetic_data_dir,
            train_cell=SYNTH_TRAIN_CELL,
            heldout_cell=SYNTH_TRAIN_CELL,
        )


def test_cell_suffix_collision_raises(tmp_path):
    """Two distinct cell IDs that share the same trailing token would
    produce colliding split names. The script must refuse."""
    # Build a synthetic dataset with cells that share a suffix.
    out = tmp_path / 'collision_dataset'
    out.mkdir()
    rows = []
    counter = 0
    for tissue in SYNTH_TISSUES:
        for cell in ('runA_s1', 'runB_s1'):  # same suffix 's1'
            for _ in range(SYNTH_ROWS_PER_BUCKET):
                rows.append({
                    'shard_idx': 0, 'row_idx': counter,
                    'read_name': f'{cell}/{counter}/ccs',
                    'tissue_str': tissue, 'cell_str': cell,
                    'tissue_id': 0, 'cell_id': 0,
                    'crop_start': 0, 'read_length': 4096,
                })
                counter += 1
    pl.DataFrame(rows).write_parquet(str(out / 'manifest.parquet'))
    with open(str(out / 'schema.json'), 'w') as f:
        json.dump({'tissue_to_id': {t: i for i, t in enumerate(SYNTH_TISSUES)}}, f)
    with pytest.raises(AssertionError, match='trailing token'):
        make_partition(
            data_dir=str(out),
            train_cell='runA_s1', heldout_cell='runB_s1',
            train_per_tissue=10,
            val_train_cell_per_tissue=5,
            val_heldout_cell_per_tissue=5,
            seed=0,
        )


def test_unknown_train_cell_raises(synthetic_data_dir):
    with pytest.raises(AssertionError, match='not in manifest'):
        _run_default(synthetic_data_dir, train_cell='nonexistent_cell')


def test_unknown_heldout_cell_raises(synthetic_data_dir):
    with pytest.raises(AssertionError, match='not in manifest'):
        _run_default(synthetic_data_dir, heldout_cell='nonexistent_cell')


def test_capacity_failure_writes_no_file(synthetic_data_dir, tmp_path):
    """Capacity error must surface before any file is written."""
    out = tmp_path / 'should_not_exist.csv'
    with pytest.raises(ValueError):
        p = _run_default(
            synthetic_data_dir,
            train_per_tissue=80,
            val_train_cell_per_tissue=80,
        )
        write_partition(p, str(out))
    assert not out.exists()


def test_overwrite_protection(synthetic_data_dir, tmp_path):
    """write_partition refuses to overwrite an existing file by default."""
    out = tmp_path / 'p.csv'
    out.write_text('preexisting\n')
    p = _run_default(synthetic_data_dir)
    with pytest.raises(FileExistsError):
        write_partition(p, str(out))
    write_partition(p, str(out), overwrite=True)
    new = pl.read_csv(str(out))
    assert new.height == p.height


# ---------------------------------------------------------------------------
# Duplicate read_name handling
# ---------------------------------------------------------------------------


def _build_manifest_with_duplicates(out_dir):
    """Synthetic manifest where some read_names appear twice (same read,
    different crop_start). Mirrors what bam_to_labeled_memmap.py can
    produce on real BAMs that contain duplicate reads."""
    rows = []
    counter = 0
    for tissue in SYNTH_TISSUES:
        for cell in SYNTH_CELLS:
            for _ in range(SYNTH_ROWS_PER_BUCKET):
                rows.append({
                    'shard_idx': 0, 'row_idx': counter,
                    'read_name': f'{cell}/{counter}/ccs',
                    'tissue_str': tissue, 'cell_str': cell,
                    'tissue_id': SYNTH_TISSUES.index(tissue),
                    'cell_id': SYNTH_CELLS.index(cell),
                    'crop_start': 0, 'read_length': 4096,
                })
                counter += 1
            base_start = counter - SYNTH_ROWS_PER_BUCKET
            # Duplicate the first 5 reads of this (tissue, cell) bucket
            # with a different crop_start.
            for j in range(5):
                rows.append({
                    'shard_idx': 0, 'row_idx': counter,
                    'read_name': f'{cell}/{base_start + j}/ccs',  # SAME read_name
                    'tissue_str': tissue, 'cell_str': cell,
                    'tissue_id': SYNTH_TISSUES.index(tissue),
                    'cell_id': SYNTH_CELLS.index(cell),
                    'crop_start': 1024, 'read_length': 4096,
                })
                counter += 1

    manifest = pl.DataFrame(rows, schema={
        'shard_idx': pl.UInt32,
        'row_idx': pl.UInt32,
        'read_name': pl.String,
        'tissue_str': pl.String,
        'cell_str': pl.String,
        'tissue_id': pl.Int32,
        'cell_id': pl.Int32,
        'crop_start': pl.UInt32,
        'read_length': pl.UInt32,
    })
    manifest.write_parquet(os.path.join(out_dir, 'manifest.parquet'))

    schema = {
        'tissue_to_id': {t: i for i, t in enumerate(SYNTH_TISSUES)},
        'cell_to_id': {c: i for i, c in enumerate(SYNTH_CELLS)},
        'context': 4096,
        'features': ['seq', 'fi', 'fp', 'ri', 'rp', 'mask'],
        'dtype': 'uint8',
    }
    with open(os.path.join(out_dir, 'schema.json'), 'w') as f:
        json.dump(schema, f)
    return out_dir


@pytest.fixture(scope='session')
def synthetic_data_dir_with_duplicates(tmp_path_factory):
    out = tmp_path_factory.mktemp('synthetic_tissue_dataset_dups')
    return _build_manifest_with_duplicates(str(out))


def test_duplicate_read_names_yield_unique_partition(
    synthetic_data_dir_with_duplicates,
):
    """When the manifest has duplicate read_names, the partition still
    contains each read_name at most once (the partition function dedupes
    by read_name before sampling)."""
    p = _run_default(synthetic_data_dir_with_duplicates)
    assert p['read_name'].n_unique() == p.height, (
        'partition still has duplicate read_names despite dedupe'
    )
    splits = ('train', SYNTH_VAL_TRAIN_SPLIT, SYNTH_VAL_HELDOUT_SPLIT)
    sets = {
        s: set(p.filter(pl.col('split') == s)['read_name'].to_list())
        for s in splits
    }
    assert sets['train'].isdisjoint(sets[SYNTH_VAL_TRAIN_SPLIT])
    assert sets['train'].isdisjoint(sets[SYNTH_VAL_HELDOUT_SPLIT])
    assert sets[SYNTH_VAL_TRAIN_SPLIT].isdisjoint(sets[SYNTH_VAL_HELDOUT_SPLIT])


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isdir(REAL_YORAN_DATA_DIR),
    reason=f'{REAL_YORAN_DATA_DIR} not present',
)
def test_real_yoran_partition_invariants():
    """End-to-end on the real yoran_ctx4096 manifest with the supervised_52
    config. Verifies all leakage / stratification invariants on actual data.
    Catches:
      - cell_str format assumptions (we filter by literal string equality).
      - per-(tissue, cell) capacity assumptions.
      - off-by-one bugs that synthetic data with uniform bucket sizes would
        miss.
    """
    p = make_partition(
        data_dir=REAL_YORAN_DATA_DIR,
        train_cell=REAL_TRAIN_CELL,
        heldout_cell=REAL_HELDOUT_CELL,
        train_per_tissue=6250,
        val_train_cell_per_tissue=1250,
        val_heldout_cell_per_tissue=1250,
        seed=42,
    )

    # Total rows.
    expected_total = 8 * (6250 + 1250 + 1250)
    assert p.height == expected_total

    # Split names match the cell suffixes.
    assert set(p['split'].unique().to_list()) == {
        'train', REAL_VAL_TRAIN_SPLIT, REAL_VAL_HELDOUT_SPLIT,
    }

    # No leakage.
    assert p['read_name'].n_unique() == p.height

    # Pairwise disjoint sets.
    train = set(p.filter(pl.col('split') == 'train')['read_name'].to_list())
    val_t = set(p.filter(pl.col('split') == REAL_VAL_TRAIN_SPLIT)['read_name'].to_list())
    val_h = set(p.filter(pl.col('split') == REAL_VAL_HELDOUT_SPLIT)['read_name'].to_list())
    assert train.isdisjoint(val_t)
    assert train.isdisjoint(val_h)
    assert val_t.isdisjoint(val_h)

    # Cell membership.
    train_prefix = REAL_TRAIN_CELL + '/'
    heldout_prefix = REAL_HELDOUT_CELL + '/'
    assert all(n.startswith(train_prefix) for n in train), (
        'train set contains a read not from train_cell'
    )
    assert all(n.startswith(train_prefix) for n in val_t), (
        f'{REAL_VAL_TRAIN_SPLIT} set contains a read not from train_cell'
    )
    assert all(n.startswith(heldout_prefix) for n in val_h), (
        f'{REAL_VAL_HELDOUT_SPLIT} set contains a read not from heldout_cell'
    )

    # Stratification: each (split, tissue) bucket has the requested count.
    # Dedupe the manifest by read_name first to mirror the partition
    # function's policy. The yoran manifest contains 588 duplicate read_name
    # rows (same read, different crop_start); the partition keeps each
    # read_name exactly once, so the join needs the same dedupe semantics.
    manifest = (
        pl.read_parquet(os.path.join(REAL_YORAN_DATA_DIR, 'manifest.parquet'))
        .select(['read_name', 'tissue_str'])
        .unique(subset=['read_name'], keep='first', maintain_order=False)
    )
    joined = p.join(manifest, on='read_name', how='left')
    assert joined['tissue_str'].null_count() == 0, (
        'some partition reads not found in manifest'
    )
    counts = joined.group_by(['split', 'tissue_str']).len()
    expected_per_tissue = {
        'train': 6250,
        REAL_VAL_TRAIN_SPLIT: 1250,
        REAL_VAL_HELDOUT_SPLIT: 1250,
    }
    for row in counts.iter_rows(named=True):
        assert row['len'] == expected_per_tissue[row['split']], (
            f"({row['split']}, {row['tissue_str']}): got {row['len']}, "
            f"expected {expected_per_tissue[row['split']]}"
        )
