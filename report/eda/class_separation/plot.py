"""
Class separation comparison: methylated vs unmethylated under each normalization.

Produces overlaid density histograms of the center-position IPD values
for positive (methylated) and negative (unmethylated) classes, faceted
by normalization strategy. Shows how legacy log-Z normalization produces
more separable class distributions than raw uint8 values.
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


def load_all_shards(directory, max_samples=50_000):
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
    with open('./configs/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    with open('./configs/supervised.yaml', 'r') as f:
        sup_config = yaml.safe_load(f)

    # Fallback to subset data if full dataset not available
    if not os.path.isdir(os.path.expandvars(sup_config.get('pos_data_train', ''))):
        sup_config['pos_data_train'] = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
        sup_config['neg_data_train'] = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'

    context = data_config['cpg_pipeline']['context']
    center = context // 2

    # ---- Raw pipeline: center IPD by class ----
    pos_raw = load_all_shards(sup_config['pos_data_train'])
    neg_raw = load_all_shards(sup_config['neg_data_train'])
    pos_ipd_raw = pos_raw[:, center, 1].astype(np.float64)
    neg_ipd_raw = neg_raw[:, center, 1].astype(np.float64)

    # ---- Per-read MAD normalization on center IPD ----
    is_continuous = np.array([False, True, True, False])

    def mad_normalize_centers(data):
        vals = []
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), min(10_000, len(data)), replace=False)
        for i in idx:
            window = data[i].astype(np.float32).copy()
            if window[center, -1] != 0.0:
                continue
            normed = normalize_read_mad(window.copy(), is_continuous_mask=is_continuous)
            vals.append(normed[center, 1])
        return np.array(vals)

    pos_ipd_mad = mad_normalize_centers(pos_raw)
    neg_ipd_mad = mad_normalize_centers(neg_raw)

    # ---- Legacy log-Z ----
    legacy_path = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
    if not os.path.exists(legacy_path):
        legacy_path = 'data/01_processed/val_sets/legacy_subset_train.parquet'
    kin_feats = ['fi', 'fp', 'ri', 'rp']
    if os.path.exists(legacy_path):
        df = pl.read_parquet(legacy_path)
        if len(df) > 50_000:
            df = df.sample(n=50_000, seed=42)
        df = df.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
        df = df.with_columns([pl.col(c).cast(pl.Array(pl.Float64, context)) for c in kin_feats])
        means, stds = compute_log_normalization_stats(df, kin_feats)
        fi_all = df['fi'].to_numpy().astype(np.float64)
        labels = df['label'].to_numpy()
        fi_norm = (np.log(fi_all + 1) - float(means['fi'])) / float(stds['fi'])
        pos_ipd_legacy = fi_norm[labels == 1, center]
        neg_ipd_legacy = fi_norm[labels == 0, center]
    else:
        # fallback
        all_ipd = np.concatenate([pos_ipd_raw, neg_ipd_raw])
        m, s = np.mean(np.log1p(all_ipd)), np.std(np.log1p(all_ipd))
        pos_ipd_legacy = (np.log1p(pos_ipd_raw) - m) / s
        neg_ipd_legacy = (np.log1p(neg_ipd_raw) - m) / s

    # ---- subsample to equal size ----
    rng = np.random.default_rng(42)
    n = min(50_000, len(pos_ipd_raw), len(neg_ipd_raw),
            len(pos_ipd_mad), len(neg_ipd_mad),
            len(pos_ipd_legacy), len(neg_ipd_legacy))

    def sub(arr):
        return arr[rng.choice(len(arr), n, replace=len(arr) < n)]

    df = pl.DataFrame({
        'value': np.concatenate([
            sub(pos_ipd_raw), sub(neg_ipd_raw),
            sub(pos_ipd_mad), sub(neg_ipd_mad),
            sub(pos_ipd_legacy), sub(neg_ipd_legacy),
        ]),
        'class': (
            ['methylated'] * n + ['unmethylated'] * n
        ) * 3,
        'normalization': (
            ['1. Raw (uint8)'] * (n * 2) +
            ['2. Per-read MAD'] * (n * 2) +
            ['3. Legacy log-Z'] * (n * 2)
        ),
    })

    norms = ['1. Raw (uint8)', '2. Per-read MAD', '3. Legacy log-Z']
    rows = []
    for norm in norms:
        row_df = df.filter(pl.col('normalization') == norm)
        row_chart = alt.Chart(row_df).mark_area(
            opacity=0.5,
            interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=60).title('IPD at CpG center'),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('class:N', scale=alt.Scale(
                domain=['methylated', 'unmethylated'],
                range=['#e45756', '#4c78a8']
            )),
        ).properties(
            width=400,
            height=200,
            title=norm,
        )
        rows.append(row_chart)

    chart = alt.vconcat(*rows).properties(
        title='Class separation at CpG center: IPD under each normalization'
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
