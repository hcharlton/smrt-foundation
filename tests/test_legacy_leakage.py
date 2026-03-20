"""
Legacy train/val leakage check.

Checks whether the legacy parquet train/test split leaks windows from
the same read across splits. If read_names overlap, the model partially
sees test reads during training — inflating eval metrics.

Run standalone on HPC:
    python tests/test_legacy_leakage.py

Or with pytest:
    python -m pytest tests/test_legacy_leakage.py -v -s
"""

import os
import sys
import polars as pl

TRAIN_PATH = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
TEST_PATH = 'data/01_processed/val_sets/pacbio_standard_test.parquet'


def check_leakage(train_path, test_path):
    print(f"Train: {train_path}")
    print(f"Test:  {test_path}")

    if not os.path.exists(train_path):
        print(f"SKIP: {train_path} not found")
        return None
    if not os.path.exists(test_path):
        print(f"SKIP: {test_path} not found")
        return None

    train_df = pl.read_parquet(train_path, columns=['read_name', 'strand', 'cg_pos'])
    test_df = pl.read_parquet(test_path, columns=['read_name', 'strand', 'cg_pos'])

    train_reads = set(train_df['read_name'].unique().to_list())
    test_reads = set(test_df['read_name'].unique().to_list())
    overlap = train_reads & test_reads

    print(f"\n--- Read-level analysis ---")
    print(f"  Train reads:      {len(train_reads)}")
    print(f"  Test reads:       {len(test_reads)}")
    print(f"  Overlapping reads: {len(overlap)}")

    if overlap:
        pct = 100 * len(overlap) / len(test_reads)
        print(f"  Overlap %% of test: {pct:.1f}%")
        print(f"  LEAKAGE DETECTED: {len(overlap)} reads appear in both train and test")

        # Count how many windows come from leaked reads
        train_leaked = train_df.filter(pl.col('read_name').is_in(list(overlap)))
        test_leaked = test_df.filter(pl.col('read_name').is_in(list(overlap)))
        print(f"\n--- Window-level impact ---")
        print(f"  Train windows from leaked reads: {len(train_leaked)} / {len(train_df)} "
              f"({100*len(train_leaked)/len(train_df):.1f}%)")
        print(f"  Test windows from leaked reads:  {len(test_leaked)} / {len(test_df)} "
              f"({100*len(test_leaked)/len(test_df):.1f}%)")
    else:
        print(f"  NO LEAKAGE: train and test have disjoint read sets")

    print(f"\n--- Window counts ---")
    print(f"  Train windows: {len(train_df)}")
    print(f"  Test windows:  {len(test_df)}")

    train_fwd = train_df.filter(pl.col('strand') == 'fwd')
    train_rev = train_df.filter(pl.col('strand') == 'rev')
    test_fwd = test_df.filter(pl.col('strand') == 'fwd')
    test_rev = test_df.filter(pl.col('strand') == 'rev')
    print(f"  Train fwd/rev:  {len(train_fwd)} / {len(train_rev)}")
    print(f"  Test fwd/rev:   {len(test_fwd)} / {len(test_rev)}")

    return len(overlap)


def test_no_leakage():
    """Pytest-compatible test."""
    result = check_leakage(TRAIN_PATH, TEST_PATH)
    if result is None:
        import pytest
        pytest.skip("Parquet files not found")
    assert result == 0, f"LEAKAGE: {result} reads in both train and test"


if __name__ == '__main__':
    train = sys.argv[1] if len(sys.argv) > 1 else TRAIN_PATH
    test = sys.argv[2] if len(sys.argv) > 2 else TEST_PATH
    overlap = check_leakage(train, test)
    if overlap is None:
        sys.exit(0)
    sys.exit(1 if overlap > 0 else 0)
