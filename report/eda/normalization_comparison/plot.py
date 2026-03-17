"""
Kinetics distribution comparison across three normalization strategies.

Produces a 3x2 faceted histogram grid:
  rows    = normalization method (raw, per-read MAD, legacy log-Z)
  columns = kinetics feature (IPD, pulse width)

Demonstrates that raw uint8 values are heavily right-skewed while
legacy log-Z normalization produces approximately standard-normal
distributions favorable for the model's linear embedding layer.
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
from smrt_foundation.dataset import LabeledMemmapDataset, compute_log_normalization_stats


def load_all_shards(directory, max_samples=200_000):
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
    center = context // 2

    # ---- New pipeline: raw ----
    pos_raw = load_all_shards(sup_config['pos_data_train'])
    neg_raw = load_all_shards(sup_config['neg_data_train'])
    all_raw = np.concatenate([pos_raw, neg_raw], axis=0)
    mask_raw = all_raw[:, :, -1] == 0.0
    ipd_raw = all_raw[:, :, 1][mask_raw].flatten().astype(np.float64)
    pw_raw = all_raw[:, :, 2][mask_raw].flatten().astype(np.float64)

    # subsample for plotting
    rng = np.random.default_rng(42)
    n_plot = min(500_000, len(ipd_raw))
    idx = rng.choice(len(ipd_raw), n_plot, replace=False)
    ipd_raw, pw_raw = ipd_raw[idx], pw_raw[idx]

    # ---- New pipeline: per-read MAD ----
    # Apply MAD normalization to the raw data sample-by-sample
    is_continuous = np.array([False, True, True, False])  # [seq, ipd, pw, mask]
    sample_idx = rng.choice(len(all_raw), min(20_000, len(all_raw)), replace=False)
    ipd_mad_list, pw_mad_list = [], []
    for i in sample_idx:
        window = all_raw[i].astype(np.float32).copy()
        active = window[:, -1] == 0.0
        if active.sum() < 4:
            continue
        normed = normalize_read_mad(window.copy(), is_continuous_mask=is_continuous)
        ipd_mad_list.append(normed[active, 1])
        pw_mad_list.append(normed[active, 2])
    ipd_mad = np.concatenate(ipd_mad_list)
    pw_mad = np.concatenate(pw_mad_list)

    # ---- Legacy pipeline: log-Z ----
    legacy_path = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
    if os.path.exists(legacy_path):
        q = pl.scan_parquet(legacy_path).head(200_000)
        train_df = q.collect()
        kin_feats = ['fi', 'fp', 'ri', 'rp']
        train_df = train_df.with_columns([
            pl.col(c).list.to_array(context) for c in kin_feats
        ])
        means, stds = compute_log_normalization_stats(train_df, kin_feats)
        fi_norm = (np.log(train_df['fi'].to_numpy().astype(np.float64) + 1) - float(means['fi'])) / float(stds['fi'])
        fp_norm = (np.log(train_df['fp'].to_numpy().astype(np.float64) + 1) - float(means['fp'])) / float(stds['fp'])
        ipd_legacy = fi_norm.flatten()
        pw_legacy = fp_norm.flatten()
        n_leg = min(500_000, len(ipd_legacy))
        idx_l = rng.choice(len(ipd_legacy), n_leg, replace=False)
        ipd_legacy, pw_legacy = ipd_legacy[idx_l], pw_legacy[idx_l]
    else:
        # fallback: simulate log-Z from raw
        ipd_legacy = (np.log1p(ipd_raw) - np.mean(np.log1p(ipd_raw))) / np.std(np.log1p(ipd_raw))
        pw_legacy = (np.log1p(pw_raw) - np.mean(np.log1p(pw_raw))) / np.std(np.log1p(pw_raw))

    # ---- Build polars dataframe ----
    # Align lengths
    n_min = min(len(ipd_raw), len(ipd_mad), len(ipd_legacy))
    df = pl.DataFrame({
        'value': np.concatenate([
            ipd_raw[:n_min], pw_raw[:n_min],
            ipd_mad[:n_min], pw_mad[:n_min],
            ipd_legacy[:n_min], pw_legacy[:n_min],
        ]),
        'feature': (
            ['IPD'] * n_min + ['PW'] * n_min +
            ['IPD'] * n_min + ['PW'] * n_min +
            ['IPD'] * n_min + ['PW'] * n_min
        ),
        'normalization': (
            ['1. Raw (uint8)'] * (n_min * 2) +
            ['2. Per-read MAD'] * (n_min * 2) +
            ['3. Legacy log-Z'] * (n_min * 2)
        ),
    })

    chart = alt.Chart(df).mark_bar(opacity=0.8).encode(
        alt.X('value:Q').bin(maxbins=80).title(None),
        alt.Y('count():Q').title('Count'),
        alt.Color('normalization:N', legend=None),
    ).properties(
        width=350,
        height=200,
    ).facet(
        row=alt.Row('normalization:N').title('Normalization'),
        column=alt.Column('feature:N').title('Kinetics Feature'),
    ).resolve_scale(
        x='independent',
        y='independent',
    ).properties(
        title='Kinetics distribution: raw vs MAD vs legacy log-Z'
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
