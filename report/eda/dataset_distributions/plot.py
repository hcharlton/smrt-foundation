import yaml, os, sys
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
import argparse

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import LegacyMethylDataset, LabeledMemmapDataset, compute_log_normalization_stats
alt.data_transformers.enable('vegafusion')
print('loaded modules')
config_path = './configs/supervised.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

KINETICS_FEATURES = ['fi', 'fp', 'ri', 'rp']
print('instantiating normalization dataframe')
q = (
    pl.scan_parquet('data/01_processed/val_sets/pacbio_standard_train.parquet')
    .head(2_000_000)
    )
train_df = q.collect()
print('calculating statistics')
train_means, train_stds = compute_log_normalization_stats(train_df, KINETICS_FEATURES)
print('instantiating new dataset class')
ds_new = LabeledMemmapDataset(config.get('pos_data_train'), config.get('neg_data_train'), limit=int(2626190/4))
print('instantiating legacy dataset class')
ds_legacy = LegacyMethylDataset('data/01_processed/val_sets/pacbio_standard_train.parquet',
                                means=train_means,
                                stds=train_stds,
                                context=32,
                                single_strand=True,
                                restrict_row_groups=5)
print('calculating lengths')
print(len(ds_new))
print(len(ds_legacy))

print('getting large data (new) samples')
batch_new =  next(iter(DataLoader(ds_new, batch_size=500_000, shuffle=True)))
print('getting large data (legacy) samples')
batch_legacy = next(iter(DataLoader(ds_legacy, batch_size=2_000_000)))

print(batch_new[0].ravel().shape)
print(batch_legacy['data'].ravel().shape)

df = pl.DataFrame({
    'nuc_new': batch_new[0][...,0].ravel(),
    'ipd_new': batch_new[0][...,1].ravel(),
    'pw_new': batch_new[0][...,2].ravel(),
    'pad_new': batch_new[0][...,3].ravel(),
    'nuc_legacy': batch_legacy['data'][...,0].ravel(),
    'ipd_legacy': batch_legacy['data'][...,1].ravel(),
    'pw_legacy': batch_legacy['data'][...,2].ravel(),
    'pad_legacy': batch_legacy['data'][...,3].ravel(),
})

print(df.head())
print(df.unpivot())

chart = alt.Chart(df.unpivot()).mark_bar().encode(
    alt.X('value:Q').bin(maxbins=100),
    alt.Y('count()')
).properties(
    width=500,
    height=500,
).facet(
    'variable:N',
    columns=2
).resolve_scale(
    x='independent',
    y='independent'
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    
    args = parser.parse_args()

    chart.save(args.output_path)