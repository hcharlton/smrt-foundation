"""Per-(tissue, cell) kinetics distributions and per-channel means.

Run: `python -m report.probe_tissue_yoran.eda_kinetics_distributions`

The headline question for this EDA: are the eight tissue distributions
visibly separated, or are the only visible separations between the two
cells? If all eight tissue curves overlap exactly while the two cell
curves are distinct, the dominant signal is cell-batch, not tissue.

Two outputs per kinetics channel `c ∈ {fi, fp, ri, rp}`:

1. `kinetics_density_by_tissue_<c>.svg` — overlaid density curves of
   normalised channel `c`, one curve per tissue. 8 curves per panel; if
   they collapse onto one another there is no per-tissue mean shift.

2. `kinetics_density_by_cell_<c>.svg` — overlaid density curves split by
   cell_id. 2 curves per panel; if they separate clearly while the
   per-tissue curves do not, the cell-batch effect dominates.

Plus one summary heatmap of the per-(tissue, cell, channel) mean (in
normalised space). Reads 4k randomly sampled rows from each split (12k
total) so the densities have enough resolution without ballooning memory.
"""

import os
import polars as pl
import altair as alt

from . import _shared


SAMPLE_PER_SPLIT = 2_000
CHANNEL_NAMES = ['fi', 'fp', 'ri', 'rp']


def _load_subsampled():
    """Load a subset of each split with normalisation applied."""
    norm = _shared.compute_norm()
    parts = {}
    for sp in ('train', 'val_s1', 'val_s2'):
        parts[sp] = _shared.load_split(
            sp, norm_fn=norm, context=2048, limit=SAMPLE_PER_SPLIT,
        )
    return parts


def _stack(parts):
    """Concatenate splits into one set of per-row arrays."""
    means = []
    for sp in ('train', 'val_s1', 'val_s2'):
        d = parts[sp]
        kin = d['X'][..., _shared.KIN_COLS].mean(axis=1)  # (N, 4) per-read mean
        for ci, ch in enumerate(CHANNEL_NAMES):
            for i in range(kin.shape[0]):
                means.append({
                    'split': sp,
                    'tissue': _shared.TISSUES[int(d['tissue_id'][i])],
                    'cell_id': int(d['cell_id'][i]),
                    'channel': ch,
                    'mean': float(kin[i, ci]),
                })
    return pl.DataFrame(means)


def _density_chart(df, color_field, color_scale, title, out_path):
    """Altair density chart, faceted by channel."""
    chart = (
        alt.Chart(df)
        .transform_density(
            'mean',
            groupby=[color_field],
            extent=[float(df['mean'].min()), float(df['mean'].max())],
            counts=False,
            steps=200,
        )
        .mark_line(opacity=0.8)
        .encode(
            alt.X('value:Q').title('per-read mean (normalised)'),
            alt.Y('density:Q').title('density'),
            alt.Color(f'{color_field}:N', scale=color_scale),
        )
        .properties(width=320, height=200)
        .facet(column=alt.Column('channel:N').title(None), title=title)
        .resolve_scale(x='independent', y='independent')
    )
    chart.save(out_path)
    print(f"  saved {out_path}")


def main():
    _shared.ensure_dirs()
    print("Loading subsampled splits with normalisation ...")
    parts = _load_subsampled()
    df = _stack(parts)
    print(f"  combined frame: {df.height} rows")

    # Cell label as nice string for plotting.
    df = df.with_columns(
        pl.when(pl.col('cell_id') == 0).then(pl.lit('cell_id=0 (s2)'))
          .otherwise(pl.lit('cell_id=1 (s1)'))
          .alias('cell_label')
    )

    print("\n[1/3] Per-tissue density (color = tissue) ...")
    _density_chart(
        df,
        color_field='tissue',
        color_scale=_shared.tissue_color_scale(),
        title='Per-read mean kinetics (normalised), coloured by tissue',
        out_path=os.path.join(_shared.FIGURES_DIR, 'kinetics_density_by_tissue.svg'),
    )

    print("[2/3] Per-cell density (color = cell_id) ...")
    _density_chart(
        df,
        color_field='cell_label',
        color_scale=_shared.cell_color_scale(),
        title='Per-read mean kinetics (normalised), coloured by cell',
        out_path=os.path.join(_shared.FIGURES_DIR, 'kinetics_density_by_cell.svg'),
    )

    print("[3/3] Per-(tissue, cell, channel) mean heatmap ...")
    means = (
        df.group_by(['tissue', 'cell_label', 'channel'])
        .agg(pl.col('mean').mean().alias('avg'))
        .sort(['channel', 'cell_label', 'tissue'])
    )
    means.write_csv(os.path.join(_shared.RESULTS_DIR, 'kinetics_means_by_tissue_cell.csv'))

    heatmap = alt.Chart(means).mark_rect().encode(
        alt.X('tissue:N').title('Tissue'),
        alt.Y('cell_label:N').title('Cell'),
        alt.Color('avg:Q').scale(scheme='redblue', domainMid=0).title('Mean (normalised)'),
    ).properties(width=320, height=80).facet(
        row=alt.Row('channel:N').title(None),
        title='Per-(tissue, cell) mean kinetics in normalised space',
    )
    heatmap.save(os.path.join(_shared.FIGURES_DIR, 'kinetics_mean_heatmap.svg'))
    print(f"  saved {os.path.join(_shared.FIGURES_DIR, 'kinetics_mean_heatmap.svg')}")


if __name__ == '__main__':
    main()
