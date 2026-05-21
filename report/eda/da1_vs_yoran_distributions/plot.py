"""
Per-channel kinetics distribution comparison between DA1 (Sequel II) and
yoran (Revio) SSL memmaps.

Informs the ssl_61 normalization decision. Two-column faceted histogram:
  columns = kinetics channel (IPD, PW)
  color   = source (da1, yoran)
Plus a 2-sample KS statistic per channel printed to stdout.

If distributions overlap heavily across sources, a combined-stats norm is
adequate. If they differ materially (different scale or shape), per-source
normalization is the right choice and ssl_61's _shared_train.py picks up
on it automatically via its per-source NormedDataset wrappers.

Run:
    bash plot.sh report/eda/da1_vs_yoran_distributions
"""

import os
import sys
import argparse
import numpy as np
import polars as pl
import altair as alt
import torch
from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import ShardedMemmapDataset, ChunkedRandomSampler


SOURCES = {
    'da1': 'data/01_processed/ssl_sets/da1_raw.memmap',
    'yoran': 'data/01_processed/ssl_sets/yoran_raw.memmap',
}
N_SAMPLES = 16_384
CHUNK_SIZE = 2048
SEED = 42
CHANNEL_NAMES = {1: 'IPD', 2: 'PW'}


def sample_source(path, n_samples):
    ds = ShardedMemmapDataset(path)
    sampler = ChunkedRandomSampler(ds, CHUNK_SIZE, shuffle_within=True)
    batch = next(iter(DataLoader(ds, batch_size=n_samples, sampler=sampler)))
    x = batch if not isinstance(batch, (tuple, list)) else batch[0]
    flat = x.reshape(-1, x.shape[-1])
    active = flat[:, -1] == 0.0
    return flat[active].numpy()


def ks_2sample(a, b):
    """2-sample KS statistic + asymptotic p-value approximation."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    pool = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, pool, side='right') / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, pool, side='right') / len(b_sorted)
    D = float(np.max(np.abs(cdf_a - cdf_b)))
    n, m = len(a_sorted), len(b_sorted)
    en = np.sqrt(n * m / (n + m))
    p = 2.0 * np.exp(-2.0 * (en * D) ** 2)
    return D, float(min(1.0, p))


def main(output_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples = {}
    for name, path in SOURCES.items():
        full_path = os.path.join(module_path, path) if not os.path.isabs(path) else path
        print(f'[{name}] sampling {N_SAMPLES} reads from {full_path}')
        samples[name] = sample_source(full_path, N_SAMPLES)
        print(f'  -> {samples[name].shape[0]} active positions')

    print('\nPer-channel stats (mean +/- std, q05/q50/q95):')
    for c, cname in CHANNEL_NAMES.items():
        for name, arr in samples.items():
            v = arr[:, c]
            print(f'  [{name}] {cname}: {v.mean():.3f} +/- {v.std():.3f}  '
                  f'q05={np.quantile(v, 0.05):.3f} q50={np.quantile(v, 0.5):.3f} q95={np.quantile(v, 0.95):.3f}')

    if len(samples) == 2:
        a_name, b_name = list(samples.keys())
        print(f'\nKS 2-sample ({a_name} vs {b_name}):')
        for c, cname in CHANNEL_NAMES.items():
            D, p = ks_2sample(samples[a_name][:, c], samples[b_name][:, c])
            print(f'  {cname}: D={D:.4f}  p~={p:.2e}')

    rows = []
    for c, cname in CHANNEL_NAMES.items():
        all_vals = np.concatenate([arr[:, c] for arr in samples.values()])
        lo, hi = np.quantile(all_vals, [0.005, 0.995])
        for name, arr in samples.items():
            v = arr[:, c]
            v = v[(v >= lo) & (v <= hi)]
            rows.append(pl.DataFrame({
                'value': v.astype(np.float64),
                'channel': [cname] * len(v),
                'source': [name] * len(v),
            }))
    df = pl.concat(rows)

    chart = alt.Chart(df).mark_bar(opacity=0.55).encode(
        alt.X('value:Q').bin(maxbins=80).title(None),
        alt.Y('count():Q').stack(None).title('Count'),
        alt.Color('source:N'),
    ).properties(
        width=400,
        height=240,
    ).facet(
        column=alt.Column('channel:N').title(None),
    ).resolve_scale(
        x='independent',
        y='independent',
    ).properties(
        title=f'DA1 vs yoran kinetics distributions (n={N_SAMPLES} reads/source)'
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
