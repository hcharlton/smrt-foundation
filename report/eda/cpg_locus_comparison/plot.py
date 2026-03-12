import yaml, os, sys
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
import argparse
import torch

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LegacyMethylDataset, LabeledMemmapDataset, compute_log_normalization_stats, ChunkedRandomSampler
alt.data_transformers.enable('vegafusion')

config_path = './configs/supervised.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

KINETICS_FEATURES = ['fi', 'fp', 'ri', 'rp']

q = (
    pl.scan_parquet('data/01_processed/val_sets/pacbio_standard_train.parquet')
    .head(2_000_000)
    )
train_df = q.collect()

train_means, train_stds = compute_log_normalization_stats(train_df, KINETICS_FEATURES)

ds_new = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=int(2626190))

ds_legacy = LegacyMethylDataset('data/01_processed/val_sets/pacbio_standard_train.parquet',
                                means=train_means,
                                stds=train_stds,
                                context=32,
                                single_strand=True,
                                restrict_row_groups=5)
print('calculating lengths')
print(len(ds_new))
print(len(ds_legacy))

sampler = ChunkedRandomSampler(ds_new, 2048, shuffle_within=True)
batch_new =  next(iter(DataLoader(ds_new, batch_size=1_000_000, sampler = sampler)))
batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=1_000_000)))

print(batch_new[0].shape)
print(batch_legacy['data'].shape)

new_mask = batch_new[1].bool()
new_means_pos = batch_new[0][new_mask].mean(dim=0)
new_means_neg = batch_new[0][~new_mask].mean(dim=0)

leg_mask = batch_legacy['label'].bool()
legacy_means_pos = batch_legacy['data'][leg_mask].mean(dim=0)
legacy_means_neg = batch_legacy['data'][~leg_mask].mean(dim=0)

print(new_means_pos.shape)
print(new_means_neg.shape)


df = pl.DataFrame({
    'ipd_new_pos': new_means_pos[:,1],
    'ipd_legacy_pos': legacy_means_pos[:,1],
    'ipd_new_neg': new_means_neg[:,1],
    'ipd_legacy_neg': legacy_means_neg[:,1]
}).with_row_index("idx")

print(df.head())
print(df.unpivot())

# chart = alt.Chart(df.unpivot(index='idx')).mark_line().encode(
#     alt.X('idx:O'),
#     alt.Y('value:Q'),
#     alt.Color('variable')
# ).properties(
#     width=800,
#     height=500,
# )


domain = ['ipd_new_pos', 'ipd_new_neg', 'ipd_legacy_pos', 'ipd_legacy_neg']
range_ = [[5, 5], [5, 5], [1, 0], [1, 0]]

chart = alt.Chart(df.unpivot(index='idx')).mark_line().encode(
    x=alt.X('idx:O'),
    y=alt.Y('value:Q'),
    color=alt.Color('variable:N'),
    strokeDash=alt.StrokeDash('variable:N', scale=alt.Scale(domain=domain, range=range_))
).properties(width=800, height=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()

    chart.save(args.output_path)