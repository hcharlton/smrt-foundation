import glob
import torch
from torch.utils.data import Dataset

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
        
        # Load specific slice from disk (zero-copy if cached)
        # Using mmap_mode='r' is crucial here
        data = np.load(self.files[file_idx], mmap_mode='r')
        return torch.from_numpy(data[local_idx].copy()) # copy() triggers the actual read