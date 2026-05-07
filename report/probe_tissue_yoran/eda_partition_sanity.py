"""Verify that partition.csv is disjoint, stratified, and cell-mapped.

Run: `python -m report.probe_tissue_yoran.eda_partition_sanity`

Asserts that train/val_s1/val_s2 read sets are disjoint, that train+val_s1
share one cell while val_s2 is from the other, and that per-tissue counts
match the declared targets within 1%. Writes a summary table to
`results/partition_sanity.csv`.
"""

import os
import polars as pl

from . import _shared


EXPECTED_PER_TISSUE = {
    'train': 6250,
    'val_s1': 1250,
    'val_s2': 1250,
}


def main():
    _shared.ensure_dirs()

    # Anomaly check: read the raw partition without whitespace stripping and
    # flag any deviation from the canonical split labels {train, val_s1, val_s2}.
    raw = _shared.load_partition(strip_whitespace=False)
    raw_unique = sorted(raw['split'].unique().to_list())
    expected_labels = {'train', 'val_s1', 'val_s2'}
    bad = [s for s in raw_unique if s not in expected_labels]
    if bad:
        print("ANOMALY: partition.csv contains non-canonical split labels:")
        for s in bad:
            n = (raw['split'] == s).sum()
            byte_repr = ' '.join(f"{b:#04x}" for b in s.encode('utf-8'))
            print(f"  {s!r}  bytes=[{byte_repr}]  n_rows={n}")
        print("  -> _shared.load_partition() strips whitespace by default, so "
              "downstream probes are unaffected. Fix the source partition file "
              "(scripts/make_tissue_partition.py) to remove this anomaly at the root.\n")

    print("Asserting partition is disjoint, stratified, cell-mapped ...")
    partition, manifest = _shared.assert_partition_sane(
        expected_per_tissue=EXPECTED_PER_TISSUE, tol=0.01,
    )
    print("  OK")

    # Manifest-duplicate audit: the build script can emit multiple windows for
    # the same read_name. Filtering by read_name therefore returns more rows
    # than the partition declares. Quantify the inflation per split.
    print("\nManifest-duplicate audit per split (filter_expr produces N+inflation rows):")
    rows = []
    for sp in ('train', 'val_s1', 'val_s2'):
        names = partition.filter(pl.col('split') == sp)['read_name'].to_list()
        n_part = len(names)
        n_mani = manifest.filter(pl.col('read_name').is_in(names)).height
        inflation = n_mani - n_part
        rows.append({'split': sp, 'partition_rows': n_part,
                     'manifest_rows': n_mani, 'inflation': inflation})
        print(f"  {sp}: partition={n_part}, manifest_rows={n_mani}, inflation=+{inflation}")
    pl.DataFrame(rows).write_csv(os.path.join(_shared.RESULTS_DIR, 'manifest_inflation.csv'))

    summary = (
        partition.join(
            manifest.select(['read_name', 'tissue_id', 'cell_id']).unique(subset='read_name'),
            on='read_name', how='left',
        )
        .group_by(['split', 'tissue_id', 'cell_id'])
        .agg(pl.len().alias('n_reads'))
        .sort(['split', 'tissue_id', 'cell_id'])
    )

    out_path = os.path.join(_shared.RESULTS_DIR, 'partition_sanity.csv')
    summary.write_csv(out_path)
    print(f"\nWrote {out_path}")

    print("\nPer-(split, cell) totals:")
    by_split_cell = (
        partition.join(
            manifest.select(['read_name', 'cell_id']).unique(subset='read_name'),
            on='read_name', how='left',
        )
        .group_by(['split', 'cell_id'])
        .agg(pl.len().alias('n'))
        .sort(['split', 'cell_id'])
    )
    print(by_split_cell)

    print("\nPer-(split, tissue) totals:")
    by_split_tissue = (
        partition.join(
            manifest.select(['read_name', 'tissue_id']).unique(subset='read_name'),
            on='read_name', how='left',
        )
        .group_by(['split', 'tissue_id'])
        .agg(pl.len().alias('n'))
        .sort(['split', 'tissue_id'])
    )
    print(by_split_tissue)


if __name__ == '__main__':
    main()
