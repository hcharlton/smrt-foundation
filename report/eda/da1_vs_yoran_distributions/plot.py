"""
DA1 vs yoran kinetics distribution comparison.

Compares per-channel kinetic value distributions between the two ssl_61
pretraining sources (DA1 = Sequel II, yoran = Revio) to inform the
normalization strategy. Heavily-overlapping distributions across sources
mean a combined-stats KineticsNorm is adequate; meaningfully displaced
distributions mean per-source norm is the right call.

Output:
  - <output_path>              -- altair chart (.svg / .html by suffix)
  - <output_path>.summary.json -- per-channel stats + KS for downstream use

Tune behavior by editing the constants block at the top. Chart layout
lives inline at the bottom of `main()` -- one `chart = alt.vconcat(...)`
to edit.

Usage:
  bash plot.sh report/eda/da1_vs_yoran_distributions
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
N_READS_PER_SOURCE = 1024
N_BINS = 80
CHUNK_SIZE = 512
BATCH_SIZE = 128
SEED = 42


def _sample_kinetics(path):
    """Stream-sample N_READS_PER_SOURCE reads from a memmap and return a
    (n_active_positions, n_features) float32 array. Batched so peak RAM
    stays bounded -- never materializes all reads in one tensor."""
    ds = ShardedMemmapDataset(path)
    sampler = ChunkedRandomSampler(ds, CHUNK_SIZE, shuffle_within=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    collected = []
    seen = 0
    for batch in dl:
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        flat = x.reshape(-1, x.shape[-1]).numpy().astype(np.float32, copy=False)
        active = flat[:, -1] == 0.0
        collected.append(flat[active])
        seen += x.shape[0]
        if seen >= N_READS_PER_SOURCE:
            break
    if not collected:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def _ks_2sample(a, b):
    """2-sample KS statistic + asymptotic p-value approximation."""
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    pool = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pool, side='right') / a.size
    cdf_b = np.searchsorted(b, pool, side='right') / b.size
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    p = 2.0 * np.exp(-2.0 * (en * D) ** 2)
    return D, float(min(1.0, p))


def main(output_path):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    samples = {}
    for name, path in SOURCES.items():
        full = os.path.join(module_path, path) if not os.path.isabs(path) else path
        print(f'[load] {name}: streaming {N_READS_PER_SOURCE} reads from {full}')
        samples[name] = _sample_kinetics(full)
        print(f'  -> {samples[name].shape[0]:,} active positions '
              f'({samples[name].nbytes / 1e6:.1f} MB)')

    stats_rows = []
    for c, cname in KIN_CHANNELS.items():
        for name, flat in samples.items():
            v = flat[:, c]
            r = {
                'source': name, 'channel': cname, 'n': int(v.size),
                'mean': float(v.mean()), 'std': float(v.std()),
                'q05': float(np.quantile(v, 0.05)),
                'q50': float(np.quantile(v, 0.50)),
                'q95': float(np.quantile(v, 0.95)),
            }
            stats_rows.append(r)
            print(f'  [{name:>6}] {cname}: '
                  f'mean={r["mean"]:.3f} std={r["std"]:.3f} '
                  f'q05={r["q05"]:.3f} q50={r["q50"]:.3f} q95={r["q95"]:.3f}')

    ks_rows = []
    src_names = list(samples.keys())
    if len(src_names) == 2:
        a, b = src_names
        for c, cname in KIN_CHANNELS.items():
            D, p = _ks_2sample(samples[a][:, c], samples[b][:, c])
            ks_rows.append({'channel': cname, 'ks_statistic': D,
                            'p_value_approx': p, 'a': a, 'b': b})
            print(f'  KS {cname}: D={D:.4f}  p~={p:.2e}')

    hist_rows = []
    for c, cname in KIN_CHANNELS.items():
        pooled = np.concatenate([samples[s][:, c] for s in samples])
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        edges = np.linspace(float(lo), float(hi), N_BINS + 1)
        width = float(edges[1] - edges[0])
        for name, flat in samples.items():
            vals = flat[:, c]
            counts, _ = np.histogram(vals, bins=edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for cc, lo_e, hi_e, n in zip(centers, edges[:-1], edges[1:], counts):
                hist_rows.append({
                    'source': name, 'channel': cname,
                    'bin_center': float(cc),
                    'bin_lo': float(lo_e), 'bin_hi': float(hi_e),
                    'count': int(n),
                    'density': (float(n) / max(1, vals.size)) / max(width, 1e-9),
                })

    hist_df = pl.DataFrame(hist_rows)
    stats_df = pl.DataFrame(stats_rows)
    ks_df = pl.DataFrame(ks_rows) if ks_rows else None

    base, _ = os.path.splitext(output_path)
    summary_path = base + '.summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'sources': SOURCES,
            'n_reads_per_source': N_READS_PER_SOURCE,
            'n_bins': N_BINS,
            'per_channel_stats': stats_rows,
            'ks_2sample': ks_rows,
        }, f, indent=2, default=float)
    print(f'[save] summary -> {summary_path}')

    # --- Chart: edit anything below here ---
    hist_panel = alt.Chart(hist_df).mark_bar(opacity=0.55).encode(
        alt.X('bin_lo:Q', bin='binned').title('Value'),
        alt.X2('bin_hi:Q'),
        alt.Y('density:Q').stack(None).title('Density'),
        alt.Color('source:N').title('Source'),
    ).properties(width=420, height=260).facet(
        column=alt.Column('channel:N').title(None),
    ).resolve_scale(x='independent', y='independent').properties(
        title=alt.TitleParams(
            text='Per-channel density: DA1 vs yoran',
            subtitle='Histogram of active positions; bins shared per channel across sources.',
        ),
    )

    quantile_base = alt.Chart(stats_df).encode(
        alt.Y('source:N').title(None).sort(None),
        alt.Color('source:N').legend(None),
    )
    quantile_panel = alt.layer(
        quantile_base.mark_rule(strokeWidth=6, opacity=0.45).encode(
            alt.X('q05:Q').title('Value'), alt.X2('q95:Q'),
        ),
        quantile_base.mark_point(size=140, filled=True).encode(alt.X('q50:Q')),
        quantile_base.mark_point(size=120, shape='diamond', filled=False,
                                 strokeWidth=2).encode(alt.X('mean:Q')),
    ).properties(width=380, height=110).facet(
        column=alt.Column('channel:N').title(None),
    ).resolve_scale(x='independent').properties(
        title=alt.TitleParams(
            text='Per-channel quantile summary',
            subtitle='Bar = q05 to q95. Filled circle = median. Open diamond = mean.',
        ),
    )

    panels = [hist_panel, quantile_panel]
    if ks_df is not None:
        ks_panel = alt.Chart(ks_df).mark_bar().encode(
            alt.X('channel:N').title('Channel'),
            alt.Y('ks_statistic:Q').title('KS D').scale(domain=[0, 1]),
            alt.Color('channel:N').legend(None),
        ).properties(
            width=420, height=240,
            title=alt.TitleParams(
                text='2-sample KS statistic per channel',
                subtitle='D close to 0 = combined norm OK. D large = per-source norm warranted.',
            ),
        )
        panels.append(ks_panel)

    chart = alt.vconcat(*panels).resolve_scale(color='independent').properties(
        title=alt.TitleParams(
            text='DA1 (Sequel II) vs yoran (Revio): kinetics distributions',
            subtitle=[
                f'Sampled {N_READS_PER_SOURCE} reads/source from raw SSL memmaps; '
                f'active (non-padded) positions only.',
                'Informs the ssl_61 normalization strategy (per-source vs combined).',
            ],
            anchor='start',
        ),
    )
    chart.save(output_path)
    print(f'[save] chart -> {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
