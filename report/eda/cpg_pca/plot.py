import os
import sys
import yaml
import argparse
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LegacyMethylDataset, compute_log_normalization_stats
from smrt_foundation.normalization import ZNorm

def main(output_path):
    alt.data_transformers.enable('vegafusion')
    

    q = (
        pl.scan_parquet('data/01_processed/val_sets/legacy_subset_train.parquet')
        .head(2_000_000)
        .drop_nans()
        .drop_nulls()
    )

    train_df = q.collect()
    train_means, train_stds = compute_log_normalization_stats(train_df, ['fi', 'fp', 'ri', 'rp'])

    ds_legacy = LegacyMethylDataset(
        'data/01_processed/val_sets/legacy_subset_train.parquet',
        means=train_means,
        stds=train_stds,
        context=32,
        single_strand=True,
        restrict_row_groups=10,
    )

    batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=1000_000)))

    X_legacy = batch_legacy['data'][..., 1:3].flatten(start_dim=1)
    X_legacy_reduced = PCA(n_components=3).fit_transform(X_legacy)

    print(batch_legacy['label'].float().mean())



    df_legacy = pl.DataFrame({
        'pc1': X_legacy_reduced[...,0],
        'pc2': X_legacy_reduced[...,1],
        'pc3': X_legacy_reduced[...,2],
        'y': batch_legacy['label']
    }) 


    chart = alt.Chart(df_legacy).mark_circle(opacity=0.1).encode(
        alt.X('pc1:Q'),
        alt.Y('pc3:Q'),
        alt.Color('y:N')
    ).properties(
        width=600,
        height=600
    )

    chart.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)