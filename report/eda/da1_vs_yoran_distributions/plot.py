"""
DA1 vs yoran kinetics distribution comparison.

Compares per-channel kinetics value distributions between the two ssl_61
pretraining sources -- DA1 (Sequel II) and yoran (Revio) -- to decide
the normalization strategy. Heavily-overlapping distributions mean a
combined-stats KineticsNorm is adequate; meaningfully displaced
distributions mean per-source norm is the right call (and the per-source
NormedDataset wiring in `scripts/experiments/ssl_61_dual_sample/_shared_train.py`
is what makes that downstream-safe).

Pipeline:
  1. Stream a chunked-random sample of `n_reads_per_source` reads from
     each source's `*_raw.memmap`. Flatten active (non-padded) positions.
  2. Compute per-channel mean/std/q05/q50/q95 from the sample.
  3. Compute a 2-sample KS statistic per channel between the two sources.
  4. Bin each channel's values into `n_bins` (shared range per channel
     across sources) and ship only the bin counts to altair.

Output:
  - <output_path>              -- altair chart (svg/html by suffix)
  - <output_path>.summary.json -- structured per-channel + KS summary

The bin-counts-only contract is why this fits in the default plot.sh
32 GB budget: peak RAM is bounded by the two source samples (~hundreds
of MB) plus tiny per-bin and per-stat aggregates. The previous version
embedded raw values into a polars DataFrame and blew the budget; do not
revert.

Usage:
  bash plot.sh report/eda/da1_vs_yoran_distributions
  # or directly with a custom output:
  python report/eda/da1_vs_yoran_distributions/plot.py \\
      --output_path report/eda/da1_vs_yoran_distributions/plot.svg \\
      --n_reads_per_source 1024
"""

import os
import sys
import json
import argparse
import numpy as np
import polars as pl
import altair as alt
import torch
from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

alt.data_transformers.enable('vegafusion')

from smrt_foundation.dataset import ShardedMemmapDataset, ChunkedRandomSampler


SOURCES = {
    'da1': 'data/01_processed/ssl_sets/da1_raw.memmap',
    'yoran': 'data/01_processed/ssl_sets/yoran_raw.memmap',
}
KIN_CHANNELS = {1: 'IPD', 2: 'PW'}
SEED = 42


def _sample_kinetics(path, n_reads, chunk_size, batch_size):
    """Stream `n_reads` reads from the memmap in small batches and return a
    (n_active_positions, n_features) float32 array of active positions.

    We never materialize all `n_reads` reads in one tensor: each
    `DataLoader` iteration produces `batch_size` reads, we filter to active
    positions, and append. The padding-channel filter happens immediately
    so post-CNN/transformer assumptions about "active positions only" are
    respected for downstream stats.
    """
    ds = ShardedMemmapDataset(path)
    sampler = ChunkedRandomSampler(ds, chunk_size, shuffle_within=True)
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    collected = []
    seen = 0
    for batch in dl:
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        flat = x.reshape(-1, x.shape[-1]).numpy().astype(np.float32, copy=False)
        active = flat[:, -1] == 0.0
        collected.append(flat[active])
        seen += x.shape[0]
        if seen >= n_reads:
            break
    if not collected:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def _ks_2sample(a, b):
    """2-sample KS statistic + asymptotic p-value approximation.

    Returns (D, p_approx). D is the max abs difference between the two
    empirical CDFs on the sorted union; p uses the standard asymptotic
    `2 * exp(-2 * (sqrt(n*m/(n+m)) * D)^2)` form (correct only in the
    large-sample regime, which we are in).
    """
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    pool = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pool, side='right') / a.size
    cdf_b = np.searchsorted(b, pool, side='right') / b.size
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    p = 2.0 * np.exp(-2.0 * (en * D) ** 2)
    return D, float(min(1.0, p))


def _per_channel_stats(values, source, channel_name):
    return {
        'source': source,
        'channel': channel_name,
        'n': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'q05': float(np.quantile(values, 0.05)),
        'q50': float(np.quantile(values, 0.50)),
        'q95': float(np.quantile(values, 0.95)),
    }


def _histogram_chart(hist_df):
    return alt.Chart(hist_df).mark_bar(opacity=0.55).encode(
        alt.X('bin_lo:Q', bin='binned').title('Value'),
        alt.X2('bin_hi:Q'),
        alt.Y('density:Q').stack(None).title('Density'),
        alt.Color('source:N').title('Source'),
        tooltip=[
            alt.Tooltip('source:N'),
            alt.Tooltip('channel:N'),
            alt.Tooltip('bin_center:Q', format='.3f', title='Center'),
            alt.Tooltip('count:Q', format=',', title='Count'),
            alt.Tooltip('density:Q', format='.4f'),
        ],
    ).properties(
        width=420, height=260,
    ).facet(
        column=alt.Column('channel:N').title(None),
    ).resolve_scale(
        x='independent', y='independent',
    ).properties(
        title=alt.TitleParams(
            text='Per-channel density: DA1 vs yoran',
            subtitle=('Histogram of active (non-padded) positions from a '
                      'chunked-random sample of each source. Bins are shared '
                      'across sources within each channel for direct overlay.'),
        ),
    )


def _quantile_summary_chart(stats_df):
    """Per-(source, channel) quantile summary: q05-q95 range rule, median
    dot, mean diamond. Faceted by channel so x-scales are channel-specific
    and the two sources are directly comparable within each facet.

    Replaces an earlier text-table panel whose long labels overflowed
    between columns; this version conveys the same numerical content
    spatially and makes a between-source shift visually obvious.
    """
    base = alt.Chart(stats_df).encode(
        alt.Y('source:N').title(None).sort(None),
        alt.Color('source:N').legend(None),
    )
    range_layer = base.mark_rule(strokeWidth=6, opacity=0.45).encode(
        alt.X('q05:Q').title('Value'),
        alt.X2('q95:Q'),
        tooltip=[
            alt.Tooltip('source:N'),
            alt.Tooltip('channel:N'),
            alt.Tooltip('q05:Q', format='.3f'),
            alt.Tooltip('q50:Q', format='.3f', title='median'),
            alt.Tooltip('q95:Q', format='.3f'),
            alt.Tooltip('mean:Q', format='.3f'),
            alt.Tooltip('std:Q', format='.3f'),
            alt.Tooltip('n:Q', format=','),
        ],
    )
    median_layer = base.mark_point(size=140, filled=True).encode(
        alt.X('q50:Q'),
        tooltip=[
            alt.Tooltip('source:N'),
            alt.Tooltip('q50:Q', format='.3f', title='median'),
        ],
    )
    mean_layer = base.mark_point(size=120, shape='diamond', filled=False, strokeWidth=2).encode(
        alt.X('mean:Q'),
        tooltip=[
            alt.Tooltip('source:N'),
            alt.Tooltip('mean:Q', format='.3f'),
            alt.Tooltip('std:Q', format='.3f'),
        ],
    )
    layered = alt.layer(range_layer, median_layer, mean_layer).properties(
        width=380, height=110,
    )
    return layered.facet(
        column=alt.Column('channel:N').title(None),
    ).resolve_scale(
        x='independent',
    ).properties(
        title=alt.TitleParams(
            text='Per-channel quantile summary',
            subtitle=('Bar = q05 to q95. Filled circle = median. '
                      'Open diamond = mean. Same sample as the histograms above.'),
        ),
    )


def _ks_chart(ks_df):
    return alt.Chart(ks_df).mark_bar().encode(
        alt.X('channel:N').title('Channel'),
        alt.Y('ks_statistic:Q').title('KS D').scale(domain=[0, 1]),
        alt.Color('channel:N').legend(None),
        tooltip=[
            alt.Tooltip('channel:N'),
            alt.Tooltip('ks_statistic:Q', format='.4f', title='D'),
            alt.Tooltip('p_value_approx:Q', format='.2e', title='p (approx)'),
        ],
    ).properties(
        width=420, height=240,
        title=alt.TitleParams(
            text='2-sample KS statistic per channel',
            subtitle=('D close to 0: distributions overlap (combined norm OK). '
                      'D large: per-source norm is the right call.'),
        ),
    )


def main(output_path, n_reads_per_source, n_bins, chunk_size, batch_size):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    samples = {}
    for name, path in SOURCES.items():
        full_path = os.path.join(module_path, path) if not os.path.isabs(path) else path
        print(f'[load] {name}: streaming {n_reads_per_source} reads from {full_path}')
        flat = _sample_kinetics(full_path, n_reads_per_source,
                                chunk_size=chunk_size, batch_size=batch_size)
        samples[name] = flat
        print(f'  -> {flat.shape[0]:,} active positions ({flat.nbytes / 1e6:.1f} MB)')

    print('[stats] per-channel summary')
    stats_rows = []
    for c, cname in KIN_CHANNELS.items():
        for name, flat in samples.items():
            stats_rows.append(_per_channel_stats(flat[:, c], name, cname))
            r = stats_rows[-1]
            print(f'  [{r["source"]:>6}] {r["channel"]}: '
                  f'mean={r["mean"]:.3f} std={r["std"]:.3f} '
                  f'q05={r["q05"]:.3f} q50={r["q50"]:.3f} q95={r["q95"]:.3f}')

    print('[stats] 2-sample KS')
    ks_rows = []
    src_names = list(samples.keys())
    if len(src_names) == 2:
        a, b = src_names
        for c, cname in KIN_CHANNELS.items():
            D, p = _ks_2sample(samples[a][:, c], samples[b][:, c])
            ks_rows.append({
                'channel': cname,
                'ks_statistic': D,
                'p_value_approx': p,
                'a': a,
                'b': b,
            })
            print(f'  {cname}: D={D:.4f}  p~={p:.2e}')

    print('[charts] computing per-channel histograms')
    hist_rows = []
    for c, cname in KIN_CHANNELS.items():
        pooled = np.concatenate([samples[s][:, c] for s in samples])
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        edges = np.linspace(float(lo), float(hi), n_bins + 1)
        width = float(edges[1] - edges[0])
        for name, flat in samples.items():
            vals = flat[:, c]
            counts, _ = np.histogram(vals, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for cc, lo_e, hi_e, n in zip(centers, edges[:-1], edges[1:], counts):
                hist_rows.append({
                    'source': name,
                    'channel': cname,
                    'bin_center': float(cc),
                    'bin_lo': float(lo_e),
                    'bin_hi': float(hi_e),
                    'count': int(n),
                    'density': (float(n) / max(1, vals.size)) / max(width, 1e-9),
                })

    hist_df = pl.DataFrame(hist_rows)
    stats_df = pl.DataFrame(stats_rows)
    ks_df = pl.DataFrame(ks_rows) if ks_rows else None

    base, _ = os.path.splitext(output_path)
    summary_path = base + '.summary.json'
    summary = {
        'sources': SOURCES,
        'n_reads_per_source': n_reads_per_source,
        'n_bins': n_bins,
        'per_channel_stats': stats_rows,
        'ks_2sample': ks_rows,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f'[save] summary -> {summary_path}')

    print('[charts] composing')
    panels = [_histogram_chart(hist_df), _quantile_summary_chart(stats_df)]
    if ks_df is not None:
        panels.append(_ks_chart(ks_df))

    chart = alt.vconcat(*panels).resolve_scale(color='independent').properties(
        title=alt.TitleParams(
            text='DA1 (Sequel II) vs yoran (Revio): kinetics distributions',
            subtitle=[
                f'Sampled {n_reads_per_source} reads/source from raw SSL memmaps; '
                f'active (non-padded) positions only.',
                'Informs the ssl_61 normalization strategy (per-source vs combined).',
            ],
            anchor='start',
        )
    )
    chart.save(output_path)
    print(f'[save] chart -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True,
                        help='Chart output path. Suffix decides format (.svg / .html).')
    parser.add_argument('--n_reads_per_source', type=int, default=1024,
                        help='Reads per source to stream-sample. 1024 fits in '
                             'plot.sh\'s default 32 GB budget; raise for tighter '
                             'tail estimates if you bump --mem.')
    parser.add_argument('--n_bins', type=int, default=80)
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='ChunkedRandomSampler chunk size during streaming.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='DataLoader batch size during streaming.')
    args = parser.parse_args()
    main(args.output_path, args.n_reads_per_source, args.n_bins,
         args.chunk_size, args.batch_size)
