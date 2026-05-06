"""
Tissue dataset overview EDA.

Inspects the labeled tissue dataset produced by
`scripts/bam_to_labeled_memmap.py` and surfaces the diagnostics that
gate the supervised_52 training plan:

  1. Per-tissue retained-row counts and per-cell counts.
  2. The tissue x cell crosstab — the load-bearing signal for whether
     held-out-cell validation is feasible. A tissue that lives in only
     one cell can't have its cell held out without losing it entirely.
  3. Per-tissue read-length distribution from the manifest's
     `read_length` column. Surfaces tissue-correlated length bias
     introduced by the build-time `read_length >= context` filter.
  4. Per-(tissue, cell) retention rate, computed by joining the labels
     file against the manifest. Identifies which (tissue, cell) buckets
     lost the most data to the length filter.
  5. In-window padding fraction across a sample of shards. Should be
     identically zero by construction (the build script drops short
     reads); a non-zero number here would indicate a build-time bug.

Output:
  - <output_path>.html  -- multi-panel altair chart
  - <output_path>.summary.json  -- structured summary for downstream
    pipeline decisions

Usage:
  python -m report.eda.tissue_dataset_overview.plot \\
      --data_dir data/01_processed/tissue_sets/yoran_ctx4096 \\
      --label_path data/01_processed/ssl_sets/yoran_read_labels.txt \\
      --output_path report/eda/tissue_dataset_overview/overview.html
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

alt.data_transformers.enable('vegafusion')


def _read_labels(path):
    """Parse `<read_name> <tissue>` whitespace-separated label file.

    The file in the repo has a trailing space on every line, so polars
    sees three columns. Read with a placeholder third column and drop
    it, rather than special-casing ragged lines.
    """
    df = pl.read_csv(
        path, has_header=False, separator=' ',
        new_columns=['read_name', 'tissue_str', '_pad'],
        truncate_ragged_lines=True,
    ).drop('_pad')
    df = df.with_columns(
        pl.col('read_name').str.split('/').list.get(0).alias('cell_str')
    )
    return df


def _check_padding(data_dir, n_shards_to_check=5, rows_per_shard=100):
    """Spot-check the mask channel across sampled rows. Should be 0
    everywhere if the build dropped short reads correctly."""
    shard_paths = sorted(glob.glob(os.path.join(data_dir, 'shard_*.npy')))
    if not shard_paths:
        return {'shards_checked': 0, 'rows_checked': 0,
                'padding_fraction_max': float('nan'),
                'padding_fraction_mean': float('nan')}
    chosen = shard_paths[::max(1, len(shard_paths) // n_shards_to_check)][:n_shards_to_check]

    pads = []
    rows_checked = 0
    for p in chosen:
        arr = np.load(p, mmap_mode='r')
        n_rows = min(rows_per_shard, arr.shape[0])
        block = np.array(arr[:n_rows])
        pad_per_row = (block[..., -1] == 1).mean(axis=1)
        pads.append(pad_per_row)
        rows_checked += n_rows
    pads = np.concatenate(pads)
    return {
        'shards_checked': len(chosen),
        'rows_checked': rows_checked,
        'padding_fraction_max': float(pads.max()),
        'padding_fraction_mean': float(pads.mean()),
    }


def _tissue_count_chart(manifest):
    counts = manifest.group_by('tissue_str').len().sort('tissue_str')
    return alt.Chart(counts).mark_bar().encode(
        alt.X('tissue_str:N').title('Tissue').sort(None),
        alt.Y('len:Q').title('Retained rows'),
        alt.Tooltip(['tissue_str:N', 'len:Q']),
    ).properties(
        width=420, height=240,
        title=alt.TitleParams(
            text='Rows per tissue',
            subtitle='Number of windows in the manifest, by tissue.',
        ),
    )


def _tissue_cell_heatmap(manifest):
    grid = (
        manifest.group_by(['tissue_str', 'cell_str', 'cell_short']).len()
        .sort(['tissue_str', 'cell_short'])
    )
    return alt.Chart(grid).mark_rect().encode(
        alt.X('cell_short:N').title('Cell'),
        alt.Y('tissue_str:N').title('Tissue').sort(None),
        alt.Color('len:Q')
            .scale(scheme='viridis')
            .legend(alt.Legend(title='Rows')),
        tooltip=[
            alt.Tooltip('tissue_str:N', title='Tissue'),
            alt.Tooltip('cell_short:N', title='Cell'),
            alt.Tooltip('cell_str:N', title='Cell (full)'),
            alt.Tooltip('len:Q', title='Rows', format=','),
        ],
    ).properties(
        width=420, height=240,
        title=alt.TitleParams(
            text='Rows by tissue and cell',
            subtitle='Manifest row counts split by source PacBio cell.',
        ),
    )


def _read_length_chart(manifest_sample):
    """Box-style read length per tissue. Uses Altair's mark_boxplot.
    Operates on a stratified sample (caller-prepared) to keep the
    embedded data small."""
    return alt.Chart(manifest_sample).mark_boxplot(extent='min-max').encode(
        alt.X('tissue_str:N').title('Tissue').sort(None),
        alt.Y('read_length:Q').title('Original read length (bp)')
            .scale(zero=False),
        alt.Color('tissue_str:N').legend(None),
    ).properties(
        width=860, height=260,
        title=alt.TitleParams(
            text='Read length by tissue',
            subtitle=('Distribution of original (uncropped) read length per '
                      'tissue, over a stratified sample of the manifest.'),
        ),
    )


def _waterfall_chart(waterfall_long, max_reads_per_tissue):
    """Per-tissue waterfall: n_in_labels -> n_after_cap -> n_in_manifest.
    Three bars side-by-side per tissue make it visually obvious that the
    per-tissue cap is the dominant filter, not the length filter."""
    stage_order = ['n_in_labels', 'n_after_cap', 'n_in_manifest']
    stage_labels = {
        'n_in_labels': '1. in labels file',
        'n_after_cap': '2. after per-tissue cap',
        'n_in_manifest': '3. in manifest (after length filter)',
    }
    df = waterfall_long.with_columns(
        pl.col('stage').replace_strict(stage_labels).alias('stage_label')
    )
    label_order = [stage_labels[s] for s in stage_order]
    return alt.Chart(df).mark_bar().encode(
        alt.X('tissue_str:N').title('Tissue').sort(None),
        alt.Y('count:Q').title('Reads'),
        alt.Color('stage_label:N').sort(label_order)
            .scale(domain=label_order,
                   range=['#9ecae1', '#4292c6', '#08519c'])
            .legend(alt.Legend(title='Pipeline stage', orient='right')),
        alt.XOffset('stage_label:N').sort(label_order),
        tooltip=[
            alt.Tooltip('tissue_str:N', title='Tissue'),
            alt.Tooltip('stage_label:N', title='Stage'),
            alt.Tooltip('count:Q', title='Reads', format=','),
        ],
    ).properties(
        width=860, height=260,
        title=alt.TitleParams(
            text='Filter cascade: labels file -> per-tissue cap -> length filter',
            subtitle=(f'Reads remaining at each stage of the build pipeline '
                      f'(per-tissue cap = {max_reads_per_tissue:,}). '
                      f'The cap is the dominant filter; the length filter '
                      f'drops only a small remainder.'),
        ),
    )


def _length_retention_chart(retention_df):
    """Per-(tissue, cell) length-filter retention. Computed against the
    EXPECTED post-cap count per (tissue, cell) so the cap's dominant
    effect is factored out and what remains is the length filter."""
    return alt.Chart(retention_df).mark_bar().encode(
        alt.X('tissue_str:N').title('Tissue').sort(None),
        alt.Y('length_retention_pct:Q').title('Length-filter retention (%)')
            .scale(domain=[0, 100]),
        alt.Color('cell_short:N')
            .legend(alt.Legend(title='Cell', orient='right')),
        alt.XOffset('cell_short:N'),
        tooltip=[
            alt.Tooltip('tissue_str:N', title='Tissue'),
            alt.Tooltip('cell_short:N', title='Cell'),
            alt.Tooltip('cell_str:N', title='Cell (full)'),
            alt.Tooltip('n_in_labels_tc:Q', title='Labels (t,c)', format=','),
            alt.Tooltip('expected_post_cap:Q', title='Expected after cap', format=',.0f'),
            alt.Tooltip('n_in_manifest:Q', title='In manifest', format=','),
            alt.Tooltip('length_retention_pct:Q',
                        title='Length retention %', format='.1f'),
        ],
    ).properties(
        width=860, height=260,
        title=alt.TitleParams(
            text='Length-filter retention by tissue and cell',
            subtitle=('Fraction of reads kept by the read_length >= context '
                      'filter, with the per-tissue cap factored out. '
                      'Uniform across (tissue, cell) means no length-driven '
                      'selection bias.'),
        ),
    )


def _length_density_chart(manifest_sample, context):
    """Per-tissue density of read lengths over a stratified sample,
    with a rule at `context` showing where the build-time length
    filter sits."""
    extent_max = int(manifest_sample['read_length'].max() or 1)
    base = alt.Chart(manifest_sample).transform_density(
        'read_length',
        groupby=['tissue_str'],
        as_=['read_length', 'density'],
        extent=[0, extent_max],
    ).mark_area(opacity=0.6).encode(
        alt.X('read_length:Q').title('Original read length (bp)'),
        alt.Y('density:Q').stack(None).title('Density'),
        alt.Color('tissue_str:N').title('Tissue'),
    )
    rule = alt.Chart(pl.DataFrame({'x': [int(context)]})).mark_rule(
        color='black', strokeDash=[4, 4]
    ).encode(alt.X('x:Q'))
    return alt.layer(base, rule).properties(
        width=860, height=240,
        title=alt.TitleParams(
            text='Read-length density by tissue',
            subtitle=(f'Density of original (uncropped) read length per '
                      f'tissue. Dashed vertical line = context cutoff '
                      f'({context} bp); reads shorter than this are dropped '
                      f'at build time.'),
        ),
    )


def main(data_dir, label_path, output_path, n_padding_check=5,
         context_override=None):
    schema_path = os.path.join(data_dir, 'schema.json')
    manifest_path = os.path.join(data_dir, 'manifest.parquet')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f'missing schema.json at {schema_path}')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f'missing manifest.parquet at {manifest_path}')

    print(f'[load] schema {schema_path}')
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    build_context = int(schema.get('context'))
    context = int(context_override) if context_override is not None else build_context
    tissue_to_id = dict(schema.get('tissue_to_id', {}))
    tissues_in_dataset = set(tissue_to_id.keys())
    print(f'  build_context={build_context}  '
          f'plot_context={context}  '
          f'tissues={sorted(tissues_in_dataset)}')

    print(f'[load] manifest {manifest_path}')
    manifest = pl.read_parquet(manifest_path)
    # Trailing token of the cell ID (e.g. m84108_..._s1 -> s1) is the
    # human-readable label used throughout the charts; the full cell_str
    # stays in tooltips for traceability.
    manifest = manifest.with_columns(
        pl.col('cell_str').str.split('_').list.last().alias('cell_short')
    )
    print(f'  {manifest.height:,} rows')

    print('[charts] building tissue-count and tissue x cell views')
    tissue_counts = manifest.group_by('tissue_str').len().sort('tissue_str')
    cell_counts = manifest.group_by('cell_str').len().sort('cell_str')
    tissue_cell = (
        manifest.group_by(['tissue_str', 'cell_str']).len()
        .sort(['tissue_str', 'cell_str'])
    )

    print('[stats] per-tissue read length quartiles')
    length_summary = (
        manifest.group_by('tissue_str')
        .agg(
            pl.col('read_length').min().alias('len_min'),
            pl.col('read_length').quantile(0.25).alias('len_q25'),
            pl.col('read_length').median().alias('len_median'),
            pl.col('read_length').quantile(0.75).alias('len_q75'),
            pl.col('read_length').max().alias('len_max'),
            pl.col('read_length').mean().alias('len_mean'),
        )
        .sort('tissue_str')
    )

    max_reads_per_tissue = int(schema.get('max_reads_per_tissue', 0))

    retention_df = None
    waterfall_long = None
    tissue_waterfall = None
    if label_path and os.path.exists(label_path):
        print(f'[load] labels {label_path}')
        labels = _read_labels(label_path)
        labels = labels.filter(pl.col('tissue_str').is_in(list(tissues_in_dataset)))
        labels = labels.with_columns(
            pl.col('cell_str').str.split('_').list.last().alias('cell_short')
        )

        # Per-tissue counts at each stage of the build pipeline.
        labels_per_t = (
            labels.group_by('tissue_str').len()
            .rename({'len': 'n_in_labels'})
        )
        manifest_per_t = (
            manifest.group_by('tissue_str').len()
            .rename({'len': 'n_in_manifest'})
        )
        cap = max_reads_per_tissue if max_reads_per_tissue > 0 else None
        tissue_waterfall = labels_per_t.join(
            manifest_per_t, on='tissue_str', how='left'
        ).with_columns(
            pl.col('n_in_manifest').fill_null(0),
            (pl.col('n_in_labels').clip(upper_bound=cap) if cap is not None
             else pl.col('n_in_labels')).alias('n_after_cap'),
        ).sort('tissue_str')

        waterfall_long = tissue_waterfall.unpivot(
            index='tissue_str',
            on=['n_in_labels', 'n_after_cap', 'n_in_manifest'],
            variable_name='stage', value_name='count',
        )

        # Per-(tissue, cell): length-filter retention factored out from cap.
        labels_per_tc = (
            labels.group_by(['tissue_str', 'cell_str', 'cell_short']).len()
            .rename({'len': 'n_in_labels_tc'})
        )
        manifest_per_tc = (
            manifest.group_by(['tissue_str', 'cell_str', 'cell_short']).len()
            .rename({'len': 'n_in_manifest'})
        )
        retention_df = (
            labels_per_tc.join(
                manifest_per_tc, on=['tissue_str', 'cell_str', 'cell_short'], how='left'
            ).with_columns(pl.col('n_in_manifest').fill_null(0))
            .join(
                tissue_waterfall.select(['tissue_str', 'n_in_labels', 'n_after_cap'])
                    .rename({'n_in_labels': 'tissue_total_labels'}),
                on='tissue_str',
            )
            .with_columns(
                (pl.col('n_in_labels_tc').cast(pl.Float64)
                 * pl.col('n_after_cap')
                 / pl.col('tissue_total_labels'))
                    .alias('expected_post_cap'),
            )
            .with_columns(
                pl.when(pl.col('expected_post_cap') > 0)
                  .then(100.0 * pl.col('n_in_manifest') / pl.col('expected_post_cap'))
                  .otherwise(None)
                  .alias('length_retention_pct')
            )
            .sort(['tissue_str', 'cell_short'])
        )
        print(f'  per-tissue waterfall:')
        print(tissue_waterfall)
        print(f'  per-(tissue, cell) length-filter retention:')
        print(retention_df.select([
            'tissue_str', 'cell_str', 'n_in_labels_tc',
            'expected_post_cap', 'n_in_manifest', 'length_retention_pct',
        ]))
    else:
        print(f'[load] labels file not found at {label_path!r} — skipping retention panel')

    print('[stats] padding spot-check')
    padding_stats = _check_padding(data_dir, n_shards_to_check=n_padding_check)
    print(f'  {padding_stats}')

    tissues_per_cell = (
        manifest.group_by('cell_str')
        .agg(pl.col('tissue_str').n_unique().alias('n_tissues'))
        .sort('cell_str')
    )
    cells_per_tissue = (
        manifest.group_by('tissue_str')
        .agg(
            pl.col('cell_str').n_unique().alias('n_cells'),
            pl.col('cell_str').unique().sort().alias('cells'),
        )
        .sort('tissue_str')
    )
    single_cell_tissues = cells_per_tissue.filter(pl.col('n_cells') == 1)
    multi_cell_tissues = cells_per_tissue.filter(pl.col('n_cells') >= 2)
    held_out_feasible = multi_cell_tissues.height > 0 and single_cell_tissues.height == 0

    print()
    print('=== summary ===')
    print(f'manifest rows: {manifest.height:,}')
    print(f'tissues: {tissue_counts.height}')
    print(f'cells: {cell_counts.height}')
    print()
    print('per-tissue counts:')
    print(tissue_counts)
    print()
    print('per-cell counts:')
    print(cell_counts)
    print()
    print('tissues per cell:')
    print(tissues_per_cell)
    print()
    print('cells per tissue:')
    print(cells_per_tissue)
    print()
    print('per-tissue read length:')
    print(length_summary)
    print()
    # The detailed per-bucket dump already happened above when retention_df
    # was built; no second print here.
    print(f'padding spot-check: {padding_stats}')
    print()
    print(
        f'held-out-cell feasible for ALL tissues: {held_out_feasible} '
        f'(single-cell tissues: {single_cell_tissues.height}, '
        f'multi-cell tissues: {multi_cell_tissues.height})'
    )
    if single_cell_tissues.height > 0:
        print('  single-cell tissues (held-out-cell would lose them):')
        print(single_cell_tissues)

    summary = {
        'data_dir': os.path.abspath(data_dir),
        'context': context,
        'manifest_rows': int(manifest.height),
        'n_tissues': int(tissue_counts.height),
        'n_cells': int(cell_counts.height),
        'tissue_to_id': tissue_to_id,
        'tissue_counts': tissue_counts.to_dicts(),
        'cell_counts': cell_counts.to_dicts(),
        'tissue_cell_counts': tissue_cell.to_dicts(),
        'tissue_length_summary': length_summary.to_dicts(),
        'tissues_per_cell': tissues_per_cell.to_dicts(),
        'cells_per_tissue': cells_per_tissue.to_dicts(),
        'padding_check': padding_stats,
        'held_out_cell_feasible_for_all_tissues': bool(held_out_feasible),
        'single_cell_tissues': single_cell_tissues['tissue_str'].to_list(),
    }
    if retention_df is not None and tissue_waterfall is not None:
        summary['max_reads_per_tissue'] = max_reads_per_tissue
        summary['tissue_waterfall'] = tissue_waterfall.to_dicts()
        summary['length_retention'] = retention_df.to_dicts()

    base, _ = os.path.splitext(output_path)
    summary_path = base + '.summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'[save] summary -> {summary_path}')

    print('[charts] composing')
    # Stratified sample for the read-length panels: cap each tissue at
    # `per_tissue_sample` rows so the embedded chart data stays small
    # (raw read_length values get serialized into the HTML).
    per_tissue_sample = 5000
    sampled_frames = []
    for tissue in manifest['tissue_str'].unique().to_list():
        sub = manifest.filter(pl.col('tissue_str') == tissue).select(
            ['tissue_str', 'read_length']
        )
        if sub.height > per_tissue_sample:
            sub = sub.sample(n=per_tissue_sample, seed=0)
        sampled_frames.append(sub)
    manifest_sample = pl.concat(sampled_frames)
    print(f'  read-length sample: {manifest_sample.height} rows '
          f'(<= {per_tissue_sample}/tissue)')

    panels = [
        alt.hconcat(
            _tissue_count_chart(manifest),
            _tissue_cell_heatmap(manifest),
        ).resolve_scale(color='independent'),
        _read_length_chart(manifest_sample),
        _length_density_chart(manifest_sample, context),
    ]
    if waterfall_long is not None:
        panels.append(_waterfall_chart(waterfall_long, max_reads_per_tissue))
    if retention_df is not None:
        panels.append(_length_retention_chart(retention_df))

    dataset_name = os.path.basename(os.path.abspath(data_dir))
    pad_max = padding_stats['padding_fraction_max']
    pad_line = (
        f'Padding check: max fraction = {pad_max:.4f} across '
        f'{padding_stats["rows_checked"]} rows from '
        f'{padding_stats["shards_checked"]} shards (expected 0).'
    )
    chart = alt.vconcat(*panels).resolve_scale(
        color='independent', x='independent', y='independent',
    ).properties(
        title=alt.TitleParams(
            text=f'Tissue dataset overview: {dataset_name}',
            subtitle=[
                f'{manifest.height:,} windows; build context = {build_context} bp; '
                f'context line drawn at {context} bp.',
                pad_line,
            ],
            anchor='start',
        )
    )

    chart.save(output_path)
    print(f'[save] chart -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/01_processed/tissue_sets/yoran_ctx4096')
    parser.add_argument('--label_path', type=str,
                        default='data/01_processed/val_sets/yoran_read_labels.txt')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path for the .html chart output')
    parser.add_argument('--n_padding_check', type=int, default=5,
                        help='Number of shards to spot-check for padding')
    parser.add_argument('--context_override', type=int, default=None,
                        help='Plot the rule at this context instead of the '
                             'build-time context from schema.json. Useful '
                             'when planning to crop further at training time.')
    args = parser.parse_args()
    main(args.data_dir, args.label_path, args.output_path,
         args.n_padding_check, args.context_override)

