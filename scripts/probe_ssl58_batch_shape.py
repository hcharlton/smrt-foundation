"""One-shot probe to confirm what shape the ssl_58 SSL DataLoader yields.

The training run on GenomeDK fails inside SmrtAutoencoder.apply_input_mask
with an indexing mismatch that should be impossible if the batch is the
expected [B, T, C] = [128, 512, 4]. This script reproduces the loader setup
(ShardedMemmapDataset -> KineticsNorm -> NormedDataset crop=512 -> default
DataLoader bs=128) and prints the actual shapes at each stage. Run on a
login node, single process, no accelerate.
"""
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'scripts/experiments/ssl_58_autoencoder_grid')
from _shared_train import NormedDataset
from smrt_foundation.dataset import ShardedMemmapDataset
from smrt_foundation.normalization import KineticsNorm

ds = ShardedMemmapDataset(
    'data/01_processed/ssl_sets/yoran_raw.memmap', limit=2000
)
print('inner sample[0]:', ds[0].shape, ds[0].dtype)

norm = KineticsNorm(ds, max_samples=512)
print('norm means/stds:', norm.means.shape, norm.stds.shape)

normed = NormedDataset(ds, norm, crop_len=512)
print('one normed sample:', normed[0].shape, normed[0].dtype)

dl = DataLoader(normed, batch_size=128, shuffle=False, num_workers=0)
batch = next(iter(dl))
print('batch:', batch.shape, batch.dtype)
