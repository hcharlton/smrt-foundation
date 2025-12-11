import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class ShardedMemmapDataset(Dataset):
    def __init__(self, data_dir):
        expanded_dir = os.path.expandvars(data_dir)
        self.shard_paths = sorted(glob.glob(os.path.join(expanded_dir, "*.npy")))
        
        # use first shard for metadata -> this makes the script break on the
        # last shard
        tmp = np.load(self.shard_paths[0], mmap_mode='r')
        self.shard_size = tmp.shape[0]
        self.seq_len = tmp.shape[1]
        self.n_feats = tmp.shape[2]
        
        self.total_len = len(self.shard_paths) * self.shard_size 
        
        self.memmaps = [None] * len(self.shard_paths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # find shard and index
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        
        # load memmaps lazily
        if self.memmaps[shard_idx] is None:
            self.memmaps[shard_idx] = np.load(self.shard_paths[shard_idx], mmap_mode='r')
            
        # get data and copy to a new array to force into RAM
        data = np.array(self.memmaps[shard_idx][local_idx])
        
        # Split mask and features (since you stacked them)
        features = data[:, :-1]
        mask = data[:, -1]
        
        return {
            "input": torch.from_numpy(features).long(), # or float
            "mask": torch.from_numpy(mask).bool()
        }
def main():
    ds = ShardedMemmapDataset("$TMPDIR")
    dl = DataLoader(
        ds,
        batch_size=512,
        num_workers=8,
        pin_memory=False,
        shuffle=True,
        persistent_workers=False,
        prefetch_factor=8
    )
    steps = 3200
    iterator = iter(dl)
    for i in tqdm(range(steps), unit='batch'):
        try:
            batch = next(iterator)
            if i % 100 == 0:
                print(batch['input'].shape)
        except StopIteration:
            break
    
        _ = batch['input']
    del iterator
    del dl
if __name__ == "__main__":
    main()
