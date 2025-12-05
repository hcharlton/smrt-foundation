import glob
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class ShardedMemmapDataset(Dataset):
    def __init__(self, data_dir):
        self.shard_paths = sorted(glob.glob(f"{data_dir}/*.npy"))
        
        # Load the first shard just to read metadata (shape/dtype)
        tmp = np.load(self.shard_paths[0], mmap_mode='r')
        self.shard_size = tmp.shape[0]
        self.seq_len = tmp.shape[1]
        self.n_feats = tmp.shape[2]
        
        # Calculate total length (approximate if last shard is smaller, 
        # but exact calculation is better if you have a metadata file)
        self.total_len = len(self.shard_paths) * self.shard_size 
        
        # Cache open memmaps (optional, depends on RAM)
        self.memmaps = [None] * len(self.shard_paths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Locate shard and local index
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        
        # Lazy loading of memmaps
        if self.memmaps[shard_idx] is None:
            self.memmaps[shard_idx] = np.load(self.shard_paths[shard_idx], mmap_mode='r')
            
        # Get data (still on disk/OS cache)
        # Copy to a fresh array to force load into RAM before converting to Tensor
        # This prevents the "Negative Strides" error in PyTorch
        data = np.array(self.memmaps[shard_idx][local_idx])
        
        # Split mask and features (since you stacked them)
        features = data[:, :-1]
        mask = data[:, -1]
        
        return {
            "input": torch.from_numpy(features).long(), # or float
            "mask": torch.from_numpy(mask).bool()
        }

ds = ShardedMemmapDataset("./ob007.memmap")
dl = DataLoader(
    ds,
    batch_size=512,
    num_workers=4,
    pin_memory=False,
    shuffle=False,
    persistent_workers=True
)
steps = 1000
for i, batch in tqdm(enumerate(dl), total=steps, unit='batch'):
    _ = batch['input']
    if i >= steps:
        break
