"""
Positional mean comparison: new memmap pipeline vs legacy parquet pipeline.

Line plot of the mean feature value at each position in the CpG context
window, comparing the new and legacy pipelines. One pane per feature
(sequence token, IPD, pulse width). Data is loaded through the actual
dataset classes and DataLoader — the same interface used by training.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import polars as pl
import altair as alt
from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset, LegacyMethylDataset, compute_log_normalization_stats


def main(output_path):
    with open('./configs/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    with open('./configs/supervised.yaml', 'r') as f:
        sup_config = yaml.safe_load(f)

    if not os.path.isdir(os.path.expandvars(sup_config.get('pos_data_train', ''))):
        sup_config['pos_data_train'] = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
        sup_config['neg_data_train'] = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'

    context = data_config['cpg_pipeline']['context']

    # ---- New pipeline: LabeledMemmapDataset -> DataLoader ----
    ds_new = LabeledMemmapDataset(
        sup_config['pos_data_train'],
        sup_config['neg_data_train'],
        limit=100_000,
    )
    batch_new = next(iter(DataLoader(ds_new, batch_size=len(ds_new), shuffle=False)))
    new_x = batch_new[0]  # (B, context, 4): [seq, ipd, pw, mask]
    new_means = new_x.mean(dim=0).numpy()  # (context, 4)

    # ---- Legacy pipeline: LegacyMethylDataset -> DataLoader ----
    legacy_path = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
    if not os.path.exists(legacy_path):
        legacy_path = 'data/01_processed/val_sets/legacy_subset_train.parquet'

    kin_feats = ['fi', 'fp', 'ri', 'rp']
    stats_df = pl.read_parquet(legacy_path)
    if len(stats_df) > 100_000:
        stats_df = stats_df.sample(n=100_000, seed=42)
    stats_df = stats_df.with_columns([pl.col(c).list.to_array(context) for c in kin_feats])
    stats_df = stats_df.with_columns([pl.col(c).cast(pl.Array(pl.Float64, context)) for c in kin_feats])
    means, stds = compute_log_normalization_stats(stats_df, kin_feats)

    ds_legacy = LegacyMethylDataset(
        legacy_path,
        means=means,
        stds=stds,
        context=context,
        single_strand=True,
        restrict_row_groups=100,
    )
    batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=len(ds_legacy))))
    legacy_data = batch_legacy['data']  # (B, context, 4): [seq, ipd, pw, mask]
    legacy_data[legacy_data.isinf()] = float('nan')
    legacy_means = legacy_data.nanmean(dim=0).numpy()  # (context, 4)

    # ---- Build charts ----
    feature_names = ['Sequence token', 'IPD', 'Pulse width']
    feature_cols = [0, 1, 2]

    charts = []
    for feat_name, col in zip(feature_names, feature_cols):
        positions = np.arange(context)
        feat_df = pl.DataFrame({
            'position': np.concatenate([positions, positions]),
            'value': np.concatenate([new_means[:, col], legacy_means[:, col]]),
            'pipeline': ['new'] * context + ['legacy'] * context,
        })

        chart = alt.Chart(feat_df).mark_line(strokeWidth=2).encode(
            alt.X('position:Q').title('Position in window').scale(domain=[0, 31]),
            alt.Y('value:Q').title(f'Mean {feat_name}').scale(zero=False),
            alt.Color('pipeline:N', scale=alt.Scale(
                domain=['new', 'legacy'],
                range=['#4c78a8', '#e45756']
            )),
            alt.StrokeDash('pipeline:N', scale=alt.Scale(
                domain=['new', 'legacy'],
                range=[[1, 0], [5, 5]]
            )),
        ).properties(width=500, height=180, title=feat_name)
        charts.append(chart)

    final = alt.vconcat(*charts).properties(
        title='Positional mean comparison: new pipeline vs legacy'
    )

    final.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
