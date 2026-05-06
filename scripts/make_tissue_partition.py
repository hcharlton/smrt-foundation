"""
Build a deterministic, verifiable train / val_<train_cell_suffix> /
val_<heldout_cell_suffix> partition over a tissue dataset's manifest.

Inputs:
  <data_dir>/manifest.parquet  - one row per labeled window (produced by
                                 scripts/bam_to_labeled_memmap.py).
  <data_dir>/schema.json       - read for the canonical tissue list.

Output:
  <data_dir>/partition.csv (or --output_path) with two columns:
    read_name (str), split (str).

  Split names:
    'train'                 - from `train_cell` only.
    'val_<train_short>'     - from `train_cell` only, disjoint with train.
    'val_<heldout_short>'   - from `heldout_cell` (a cell never seen by the
                              train split).

  `train_short` and `heldout_short` are the trailing tokens of the cell IDs
  (i.e. `cell_str.split('_')[-1]`). For yoran this gives `val_s1` and
  `val_s2`, so the val split names tell you exactly which sequencing cell
  each row came from.

Policy:
  - `train` and `val_<train_short>` come from `train_cell` only and are
    disjoint.
  - `val_<heldout_short>` comes from `heldout_cell` only.
  - Each split is stratified by tissue at exactly the requested per-tissue
    count.

The partition is the auditable record of the splits for downstream training.
It is read (never written) by training scripts; train.py asserts the file
exists and refuses to start without it.

Usage:
    python -m scripts.make_tissue_partition \\
        --data_dir data/01_processed/tissue_sets/yoran_ctx4096 \\
        --train_cell m84108_251007_115244_s1 \\
        --heldout_cell m84108_250930_153107_s2 \\
        --train_per_tissue 6250 \\
        --val_train_cell_per_tissue 1250 \\
        --val_heldout_cell_per_tissue 1250 \\
        --seed 42
"""

import argparse
import json
import os
import sys

import numpy as np
import polars as pl


def _cell_short(cell_str):
    """Trailing token of a PacBio cell ID. e.g. 'm84108_251007_115244_s1' -> 's1'."""
    return cell_str.split('_')[-1]


def make_partition(
    data_dir,
    train_cell,
    heldout_cell,
    train_per_tissue,
    val_train_cell_per_tissue,
    val_heldout_cell_per_tissue,
    seed,
):
    """
    Build the partition DataFrame. Pure function: no IO writes.

    Returns a polars.DataFrame with columns ('read_name', 'split'), sorted by
    'read_name'. Split names are 'train', 'val_<train_short>', and
    'val_<heldout_short>', where the suffixes are the trailing tokens of
    the cell IDs (e.g. 's1', 's2' for yoran cells).

    Determinism is anchored by:
      - manifest.sort('read_name') before any sampling (canonical pool order
        regardless of how the manifest was originally written).
      - Tissues iterated in sorted(schema['tissue_to_id'].keys()) order.
      - Single np.random.default_rng(seed) instance advanced consistently
        across the loop.
      - Output sorted by read_name.

    Raises:
      AssertionError if train_cell == heldout_cell, if their suffixes
      collide, or if either cell is absent from the manifest.
      ValueError if any (tissue, cell) bucket has fewer rows than requested.
    """
    assert train_cell != heldout_cell, (
        f"train_cell and heldout_cell must differ; got both = {train_cell!r}"
    )

    train_short = _cell_short(train_cell)
    heldout_short = _cell_short(heldout_cell)
    assert train_short != heldout_short, (
        f"train_cell and heldout_cell share the same trailing token "
        f"{train_short!r}; split names would collide. Pass cells with "
        f"distinct suffixes (e.g. ..._s1 vs ..._s2)."
    )
    val_train_split = f'val_{train_short}'
    val_heldout_split = f'val_{heldout_short}'

    manifest_path = os.path.join(data_dir, 'manifest.parquet')
    schema_path = os.path.join(data_dir, 'schema.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f'missing manifest.parquet at {manifest_path}')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f'missing schema.json at {schema_path}')

    manifest = pl.read_parquet(manifest_path)
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    tissues = sorted(schema['tissue_to_id'].keys())

    # Deduplicate by read_name. The build script can produce more than one
    # manifest row per read_name (different crop_start values for the same
    # read). The partition is keyed by read_name, so we sample over distinct
    # read_names: each one goes to exactly one split, ruling out cross-split
    # leakage even when the manifest contains duplicates. At training time,
    # the dataloader's filter on read_name will pick up every manifest row
    # for the chosen reads, which is the intended behavior.
    n_before = manifest.height
    manifest = manifest.unique(subset=['read_name'], keep='first', maintain_order=False)
    n_after = manifest.height
    if n_before != n_after:
        print(
            f'[make_partition] manifest has {n_before - n_after} duplicate '
            f'read_name rows ({n_before:,} total -> {n_after:,} unique). '
            f'Sampling over unique read_names so each appears in at most '
            f'one split.',
            file=sys.stderr,
        )

    cells_present = set(manifest['cell_str'].unique().to_list())
    for cell in (train_cell, heldout_cell):
        assert cell in cells_present, (
            f"cell {cell!r} not in manifest; present cells: "
            f"{sorted(cells_present)}"
        )

    # Per-(tissue, cell) capacity check before any sampling. Surfaces a clean
    # error rather than silently truncating.
    for tissue in tissues:
        n_train_cell = manifest.filter(
            (pl.col('cell_str') == train_cell) & (pl.col('tissue_str') == tissue)
        ).height
        if n_train_cell < train_per_tissue + val_train_cell_per_tissue:
            raise ValueError(
                f"tissue={tissue} cell={train_cell}: need "
                f"{train_per_tissue + val_train_cell_per_tissue} rows for "
                f"train+{val_train_split}, have {n_train_cell}"
            )
        n_heldout = manifest.filter(
            (pl.col('cell_str') == heldout_cell) & (pl.col('tissue_str') == tissue)
        ).height
        if n_heldout < val_heldout_cell_per_tissue:
            raise ValueError(
                f"tissue={tissue} cell={heldout_cell}: need "
                f"{val_heldout_cell_per_tissue} rows for {val_heldout_split}, "
                f"have {n_heldout}"
            )

    # Canonical pool ordering. Independent of manifest write order.
    manifest = manifest.sort('read_name')

    rng = np.random.default_rng(seed)
    rows = []
    for tissue in tissues:
        train_pool = manifest.filter(
            (pl.col('cell_str') == train_cell) & (pl.col('tissue_str') == tissue)
        )['read_name']
        # train_idx and val_train_cell_idx are contiguous slices of one
        # permutation, so they are disjoint by construction.
        perm = rng.permutation(train_pool.len())
        train_idx = perm[:train_per_tissue]
        val_train_cell_idx = perm[
            train_per_tissue : train_per_tissue + val_train_cell_per_tissue
        ]

        heldout_pool = manifest.filter(
            (pl.col('cell_str') == heldout_cell) & (pl.col('tissue_str') == tissue)
        )['read_name']
        heldout_perm = rng.permutation(heldout_pool.len())
        val_heldout_cell_idx = heldout_perm[:val_heldout_cell_per_tissue]

        train_names = train_pool.gather(train_idx).to_list()
        val_train_cell_names = train_pool.gather(val_train_cell_idx).to_list()
        val_heldout_cell_names = heldout_pool.gather(val_heldout_cell_idx).to_list()

        rows.extend({'read_name': n, 'split': 'train'} for n in train_names)
        rows.extend(
            {'read_name': n, 'split': val_train_split} for n in val_train_cell_names
        )
        rows.extend(
            {'read_name': n, 'split': val_heldout_split} for n in val_heldout_cell_names
        )

    partition = pl.DataFrame(rows, schema={'read_name': pl.String, 'split': pl.String})
    return partition.sort('read_name')


def write_partition(partition, output_path, overwrite=False):
    """Write partition.csv. Refuses to overwrite an existing file unless
    `overwrite=True`."""
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    partition.write_csv(output_path)


def _print_summary(partition, manifest_path, output_path):
    """Stdout summary so leakage is visible at runtime as a sanity check
    alongside the test suite's rigorous proof."""
    # Dedupe the manifest by read_name so the summary join doesn't inflate
    # counts when the manifest contains duplicate read_names (same read,
    # multiple crop_starts). The partition is already deduplicated.
    manifest = (
        pl.read_parquet(manifest_path)
        .select(['read_name', 'tissue_str', 'cell_str'])
        .unique(subset=['read_name'], keep='first', maintain_order=False)
    )
    joined = partition.join(manifest, on='read_name', how='left')

    print('--------------------------------------')
    print(f'Wrote {partition.height} rows -> {output_path}')
    print()
    print('Per-split, per-tissue counts:')
    counts = (
        joined.group_by(['split', 'tissue_str']).len()
        .sort(['split', 'tissue_str'])
    )
    print(counts)
    print()
    print('Per-split, per-cell counts:')
    print(
        joined.group_by(['split', 'cell_str']).len()
        .sort(['split', 'cell_str'])
    )
    print()

    # Disjointness sanity-check at write time. The test suite is the
    # rigorous proof; this is a runtime tripwire.
    split_names = sorted(partition['split'].unique().to_list())
    sets = {
        s: set(partition.filter(pl.col('split') == s)['read_name'].to_list())
        for s in split_names
    }
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                print(
                    f'ERROR: {len(overlap)} read_names appear in both {a} and {b}',
                    file=sys.stderr,
                )
                sys.exit(1)
    print('Disjointness check: OK (no read_name appears in multiple splits)')
    print('--------------------------------------')


def main():
    parser = argparse.ArgumentParser(
        description='Build a deterministic train/val_random/val_heldout partition '
                    'over a tissue dataset manifest.'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing manifest.parquet and schema.json.')
    parser.add_argument('--train_cell', type=str, required=True,
                        help='cell_str whose reads supply train and val_random.')
    parser.add_argument('--heldout_cell', type=str, required=True,
                        help='cell_str whose reads supply val_heldout. '
                             'Must differ from --train_cell.')
    parser.add_argument('--train_per_tissue', type=int, required=True,
                        help='Per-tissue rows for the train split (from train_cell).')
    parser.add_argument('--val_train_cell_per_tissue', type=int, required=True,
                        help='Per-tissue rows for val_<train_cell_short> '
                             '(from train_cell, disjoint with train).')
    parser.add_argument('--val_heldout_cell_per_tissue', type=int, required=True,
                        help='Per-tissue rows for val_<heldout_cell_short> '
                             '(from heldout_cell only).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default=None,
                        help='Default: <data_dir>/partition.csv.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace an existing partition file.')

    args = parser.parse_args()

    if args.train_per_tissue <= 0:
        print('Error: --train_per_tissue must be positive', file=sys.stderr)
        sys.exit(1)
    if args.val_train_cell_per_tissue <= 0:
        print('Error: --val_train_cell_per_tissue must be positive', file=sys.stderr)
        sys.exit(1)
    if args.val_heldout_cell_per_tissue <= 0:
        print('Error: --val_heldout_cell_per_tissue must be positive', file=sys.stderr)
        sys.exit(1)

    data_dir = os.path.expanduser(args.data_dir)
    output_path = (
        os.path.expanduser(args.output_path)
        if args.output_path is not None
        else os.path.join(data_dir, 'partition.csv')
    )

    partition = make_partition(
        data_dir=data_dir,
        train_cell=args.train_cell,
        heldout_cell=args.heldout_cell,
        train_per_tissue=args.train_per_tissue,
        val_train_cell_per_tissue=args.val_train_cell_per_tissue,
        val_heldout_cell_per_tissue=args.val_heldout_cell_per_tissue,
        seed=args.seed,
    )
    write_partition(partition, output_path, overwrite=args.overwrite)
    _print_summary(
        partition,
        manifest_path=os.path.join(data_dir, 'manifest.parquet'),
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
