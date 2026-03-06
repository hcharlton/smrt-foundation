import os
import sys
import yaml
import argparse
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

module_path = os.path.abspath("/dcai/users/chache/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LegacyMethylDataset, LabeledMemmapDataset, compute_log_normalization_stats, ChunkedRandomSampler
from smrt_foundation.normalization import ZNorm

def main(output_path):
    alt.data_transformers.enable('vegafusion')
    
    with open('./configs/supervised.yaml', 'r') as f:
        config = yaml.safe_load(f)

    q = (
        pl.scan_parquet('data/01_processed/val_sets/pacbio_standard_train.parquet')
        .head(2_000_000)
    )

    train_df = q.collect()
    train_means, train_stds = compute_log_normalization_stats(train_df, ['fi', 'fp', 'ri', 'rp'])

    tmp_ds = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'))

    ds_new = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=2626190)
    ds_legacy = LegacyMethylDataset(
        'data/01_processed/val_sets/pacbio_standard_train.parquet',
        means=train_means,
        stds=train_stds,
        context=32,
        single_strand=True,
        restrict_row_groups=10
    )

    batch_new = next(iter(DataLoader(ds_new, batch_size=10_000, shuffle=False)))
    batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=10_000, shuffle=False)))

    X_new = batch_new[0][...,1:3].flatten(start_dim=1)
    X_legacy = batch_legacy['data'][..., 1:3].flatten(start_dim=1)
    print(X_new.shape)
    X_new_reduced = PCA(n_components=3).fit_transform(X_new)
    X_legacy_reduced = PCA(n_components=3).fit_transform(X_legacy)

    print(X_new_reduced.shape)
    print(batch_legacy['label'].float().mean())


    df_new = pl.DataFrame({
        'pc1': X_new_reduced[...,0],
        'pc2': X_new_reduced[...,1],
        'pc3': X_new_reduced[...,2],
        'y': batch_new[1]
    }) 

    df_legacy = pl.DataFrame({
        'pc1': X_legacy_reduced[...,0],
        'pc2': X_legacy_reduced[...,1],
        'pc3': X_legacy_reduced[...,2],
        'y': batch_legacy['label']
    }) 

    print(df_new.head())

    chart = alt.Chart(df_legacy).mark_circle(opacity=0.3).encode(
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