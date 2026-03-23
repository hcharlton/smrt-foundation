"""
Side-by-side heatmaps: what the model actually sees from each pipeline.

Visual sanity check across all hypotheses. Shows individual CpG windows
after normalization as the model's embedding layer would receive them.
Reveals alignment issues, unexpected patterns, or blank regions.

Uses legacy parquet + new memmap shards.
"""

import os
import sys
import glob
import yaml
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import compute_log_normalization_stats

LEGACY_PATH = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
LEGACY_FALLBACK = 'data/01_processed/val_sets/legacy_subset_train.parquet'
CONTEXT = 32
N_WINDOWS = 10


def load_all_shards(directory, max_samples=500):
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    if not paths:
        return np.empty((0,))
    arrays = [np.load(p) for p in paths]
    out = np.concatenate(arrays, axis=0)
    return out[:max_samples]


def legacy_windows(parquet_path, label_val, n, context):
    """Load and normalize legacy windows, returning [n, context, 3] (seq, fi_norm, fp_norm)."""
    df = pl.read_parquet(parquet_path)
    kin_feats = ['fi', 'fp', 'ri', 'rp']
    df = df.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])

    means, stds = compute_log_normalization_stats(df, kin_feats)

    subset = df.filter(pl.col('label') == label_val).head(n)
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    windows = []
    for row in subset.iter_rows(named=True):
        seq_tokens = np.array([vocab.get(c, 4) for c in row['seq']], dtype=np.float64)
        fi_norm = (np.log(np.array(row['fi'], dtype=np.float64) + 1) - float(means['fi'])) / float(stds['fi'])
        fp_norm = (np.log(np.array(row['fp'], dtype=np.float64) + 1) - float(means['fp'])) / float(stds['fp'])
        windows.append(np.stack([seq_tokens, fi_norm, fp_norm], axis=-1))

    return np.array(windows)


def new_windows(pos_dir, neg_dir, label_val, n):
    """Load new pipeline windows, apply log1p + z-score, return [n, context, 3]."""
    directory = pos_dir if label_val == 1 else neg_dir
    data = load_all_shards(directory, max_samples=n * 10)
    if data.ndim < 3:
        return np.empty((0,))
    data = data[:n].astype(np.float64)

    # Apply log1p to kinetics columns then z-score
    kin = data[:, :, 1:3].copy()
    kin_log = np.log1p(kin)
    means = kin_log.mean(axis=(0, 1))
    stds = kin_log.std(axis=(0, 1))
    kin_norm = (kin_log - means) / (stds + 1e-8)

    result = np.zeros_like(data[:, :, :3])
    result[:, :, 0] = data[:, :, 0]  # seq tokens
    result[:, :, 1:3] = kin_norm
    return result


def windows_to_df(windows, pipeline, label_name, n_windows):
    """Convert [n, context, 3] array to a long-form DataFrame for heatmap plotting."""
    rows = []
    for win_idx in range(min(n_windows, len(windows))):
        for pos in range(windows.shape[1]):
            for feat_idx, feat_name in enumerate(['seq', 'IPD', 'PW']):
                rows.append({
                    'window': win_idx,
                    'position': pos,
                    'feature': feat_name,
                    'value': float(windows[win_idx, pos, feat_idx]),
                    'pipeline': pipeline,
                    'class': label_name,
                })
    return pl.DataFrame(rows)


def main(output_path):
    with open('./configs/supervised.yaml', 'r') as f:
        sup_config = yaml.safe_load(f)

    legacy_path = LEGACY_PATH if os.path.exists(LEGACY_PATH) else LEGACY_FALLBACK

    pos_train = sup_config.get('pos_data_train', '')
    neg_train = sup_config.get('neg_data_train', '')
    if not os.path.isdir(os.path.expandvars(pos_train)):
        pos_train = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
        neg_train = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'

    dfs = []

    # Legacy windows
    if os.path.exists(legacy_path):
        for label_val, label_name in [(1, 'methylated'), (0, 'unmethylated')]:
            wins = legacy_windows(legacy_path, label_val, N_WINDOWS, CONTEXT)
            if len(wins) > 0:
                dfs.append(windows_to_df(wins, 'legacy', label_name, N_WINDOWS))
    else:
        print(f"WARNING: No legacy parquet at {legacy_path}")

    # New pipeline windows
    for label_val, label_name in [(1, 'methylated'), (0, 'unmethylated')]:
        wins = new_windows(pos_train, neg_train, label_val, N_WINDOWS)
        if len(wins) > 0:
            dfs.append(windows_to_df(wins, 'new', label_name, N_WINDOWS))

    if not dfs:
        print("ERROR: No data loaded")
        sys.exit(1)

    all_df = pl.concat(dfs)

    # Only plot IPD and PW (seq tokens are categorical, not useful as heatmap)
    kin_df = all_df.filter(pl.col('feature') != 'seq')

    chart = alt.Chart(kin_df).mark_rect().encode(
        alt.X('position:O').title('Position in window'),
        alt.Y('window:O').title('Sample'),
        alt.Color('value:Q').scale(scheme='blueorange', domainMid=0).title('Normalized value'),
        alt.Row('class:N'),
        alt.Column('pipeline:N'),
        alt.Facet('feature:N'),
    ).properties(
        width=350, height=120,
    )

    final = chart.properties(
        title='Model input heatmaps: normalized kinetics at CpG windows'
    )
    final.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
