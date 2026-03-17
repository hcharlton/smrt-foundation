"""
Per-position kinetics profile across the CpG window.

Plots the mean IPD at each position in the context window for
methylated vs unmethylated classes, under each normalization strategy.
The CpG dinucleotide sits at the center — any methylation-dependent
kinetics signal should appear as a pos/neg divergence near the center.

This directly visualizes what the model's attention mechanism sees
at each position and whether the normalization preserves or destroys
the discriminative signal.
"""

import os
import sys
import yaml
import glob
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.normalization import normalize_read_mad
from smrt_foundation.dataset import compute_log_normalization_stats


def load_all_shards(directory, max_samples=100_000):
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    if not paths:
        return np.empty((0,))
    arrays = []
    n = 0
    for p in paths:
        arr = np.load(p)
        arrays.append(arr)
        n += arr.shape[0]
        if max_samples and n >= max_samples:
            break
    out = np.concatenate(arrays, axis=0)
    return out[:max_samples] if max_samples else out


def main(output_path):
    alt.data_transformers.enable('vegafusion')

    with open('./configs/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    with open('./configs/supervised.yaml', 'r') as f:
        sup_config = yaml.safe_load(f)

    context = data_config['cpg_pipeline']['context']

    # ---- Raw pipeline ----
    pos_raw = load_all_shards(sup_config['pos_data_train'])
    neg_raw = load_all_shards(sup_config['neg_data_train'])

    # Mean IPD per position (col 1), only non-padded
    def positional_mean(data, col=1):
        mask = data[:, :, -1] == 0.0
        vals = data[:, :, col].astype(np.float64)
        vals[~mask] = np.nan
        return np.nanmean(vals, axis=0)

    pos_raw_profile = positional_mean(pos_raw)
    neg_raw_profile = positional_mean(neg_raw)

    # ---- Per-read MAD ----
    is_continuous = np.array([False, True, True, False])
    rng = np.random.default_rng(42)

    def mad_normalize_all(data, n_samples=20_000):
        idx = rng.choice(len(data), min(n_samples, len(data)), replace=False)
        out = np.full((len(idx), context), np.nan, dtype=np.float64)
        for j, i in enumerate(idx):
            window = data[i].astype(np.float32).copy()
            active = window[:, -1] == 0.0
            if active.sum() < 4:
                continue
            normed = normalize_read_mad(window.copy(), is_continuous_mask=is_continuous)
            ipd = normed[:, 1].astype(np.float64)
            ipd[~active] = np.nan
            out[j] = ipd
        return np.nanmean(out, axis=0)

    pos_mad_profile = mad_normalize_all(pos_raw)
    neg_mad_profile = mad_normalize_all(neg_raw)

    # ---- Legacy log-Z ----
    legacy_path = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
    kin_feats = ['fi', 'fp', 'ri', 'rp']
    if os.path.exists(legacy_path):
        q = pl.scan_parquet(legacy_path).head(200_000)
        df = q.collect()
        df = df.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
        means, stds = compute_log_normalization_stats(df, kin_feats)
        fi_all = df['fi'].to_numpy().astype(np.float64)
        labels = df['label'].to_numpy()
        fi_norm = (np.log(fi_all + 1) - float(means['fi'])) / float(stds['fi'])
        pos_legacy_profile = fi_norm[labels == 1].mean(axis=0)
        neg_legacy_profile = fi_norm[labels == 0].mean(axis=0)
    else:
        all_raw_ipd = np.concatenate([
            pos_raw[:, :, 1].astype(np.float64),
            neg_raw[:, :, 1].astype(np.float64)
        ])
        m = np.mean(np.log1p(all_raw_ipd))
        s = np.std(np.log1p(all_raw_ipd))
        pos_legacy_profile = np.mean((np.log1p(pos_raw[:, :, 1].astype(np.float64)) - m) / s, axis=0)
        neg_legacy_profile = np.mean((np.log1p(neg_raw[:, :, 1].astype(np.float64)) - m) / s, axis=0)

    # ---- Build dataframe ----
    positions = np.arange(context)
    rows = []
    for pos_prof, neg_prof, norm_name in [
        (pos_raw_profile, neg_raw_profile, '1. Raw (uint8)'),
        (pos_mad_profile, neg_mad_profile, '2. Per-read MAD'),
        (pos_legacy_profile, neg_legacy_profile, '3. Legacy log-Z'),
    ]:
        for p in range(context):
            rows.append({'position': p, 'value': float(pos_prof[p]),
                         'class': 'methylated', 'normalization': norm_name})
            rows.append({'position': p, 'value': float(neg_prof[p]),
                         'class': 'unmethylated', 'normalization': norm_name})

    df = pl.DataFrame(rows)

    # CpG center annotation
    cpg_center = (context - 2) // 2

    base = alt.Chart(df).mark_line(strokeWidth=2).encode(
        alt.X('position:Q').title('Position in window'),
        alt.Y('value:Q').title('Mean IPD'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8']
        )),
        alt.StrokeDash('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=[[1, 0], [5, 5]]
        )),
    ).properties(
        width=500,
        height=180,
    )

    rule = alt.Chart(pl.DataFrame({'x': [cpg_center]})).mark_rule(
        color='gray', strokeDash=[3, 3], opacity=0.6
    ).encode(x='x:Q')

    chart = alt.layer(base, rule).facet(
        row=alt.Row('normalization:N').title('Normalization'),
    ).resolve_scale(
        y='independent',
    ).properties(
        title='Mean IPD profile across CpG window: methylated vs unmethylated'
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
