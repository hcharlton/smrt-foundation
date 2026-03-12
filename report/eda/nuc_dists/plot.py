# import yaml, os, sys
# import polars as pl
# import altair as alt
# from torch.utils.data import DataLoader
# import argparse
# import torch

# module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from smrt_foundation.dataset import LegacyMethylDataset, LabeledMemmapDataset, compute_log_normalization_stats
# alt.data_transformers.enable('vegafusion')

# config_path = './configs/supervised.yaml'
# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)

# KINETICS_FEATURES = ['fi', 'fp', 'ri', 'rp']

# q = (
#     pl.scan_parquet('data/01_processed/val_sets/pacbio_standard_train.parquet')
#     .head(2_000_000)
#     )
# train_df = q.collect()

# train_means, train_stds = compute_log_normalization_stats(train_df, KINETICS_FEATURES)

# ds_new = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=int(2626190))

# ds_legacy = LegacyMethylDataset('data/01_processed/val_sets/pacbio_standard_train.parquet',
#                                 means=train_means,
#                                 stds=train_stds,
#                                 context=32,
#                                 single_strand=True,
#                                 restrict_row_groups=5)
# print('calculating lengths')
# print(len(ds_new))
# print(len(ds_legacy))


# batch_new =  next(iter(DataLoader(ds_new, batch_size=100_000, shuffle=False)))
# batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=100_000)))

# print(batch_new[0].shape)
# print(batch_legacy['data'].shape)

# new_nuc= batch_new[0][...,0].ravel()
# legacy_nuc = batch_legacy['data'][...,0].ravel()


# df = pl.DataFrame({
#     'new_nuc': new_nuc,
#     'legacy_nuc': legacy_nuc,
# })

# print(df.head())
# print(df.unpivot())


# chart = alt.Chart(df.unpivot()).mark_bar().encode(
#     x=alt.X('value:Q'),
#     y=alt.Y('count():Q'),
# ).properties(width=500, height=500).facet(
#     'variable:N',
#     columns=2
# )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output_path", type=str)
    
#     args = parser.parse_args()

#     chart.save(args.output_path)



import os
import sys
import yaml
import argparse
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LegacyMethylDataset, LabeledMemmapDataset, compute_log_normalization_stats

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

    ds_new = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=2626190)
    ds_legacy = LegacyMethylDataset(
        'data/01_processed/val_sets/pacbio_standard_train.parquet',
        means=train_means,
        stds=train_stds,
        context=32,
        single_strand=True,
        restrict_row_groups=10
    )

    batch_new = next(iter(DataLoader(ds_new, batch_size=1_000_000, shuffle=False)))
    batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=1_000_000, shuffle=False)))

    new_nuc = batch_new[0][..., 0].ravel().numpy()
    legacy_nuc = batch_legacy['data'][..., 0].ravel().numpy()

    df = pl.DataFrame({
        'new_nuc': new_nuc,
        'legacy_nuc': legacy_nuc
    })

    chart = alt.Chart(df.unpivot()).mark_bar().encode(
        x=alt.X('value:N'),
        y=alt.Y('count():Q'),
        color=alt.Color('variable:N', legend=None)
    ).properties(
        width=400, 
        height=400
    ).facet(
        'variable:N',
        columns=2
    )

    chart.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)