import glob
import torch
import numpy as np
from torch.utils.data import Dataset, Dataloader
from tqdm import tqdm

class ShardedMemmapDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(f"{data_dir}/shard_*.npy"))
        self.file_indices = []
        self.cumulative_sizes = [0]
        
        # Quick index build (reads headers only)
        for f in self.files:
            # parsing header without loading data
            shape = np.load(f, mmap_mode='r').shape 
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + shape[0])
            
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        # Binary search or simple bisect to find file_idx
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[file_idx]

        data = np.load(self.files[file_idx], mmap_mode='r')
        return torch.from_numpy(data[local_idx].copy()) 
    
# ds = GenomicZarrDataset('./ob007.memmap')
y = np.load('ob007.memmap/shard_00001.npy', mmap_mode='r')
# dl = DataLoader(
    #     ds, 
    #     batch_size=1, 
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=False, 
    #     # prefetch_factor=4
    # )


# for batch in tqdm(dl):
    # x = batch

print(y.shape)