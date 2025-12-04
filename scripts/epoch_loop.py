import os
import tensorstore as ts
import torch
import os
from torch.utils.data import Dataset, DataLoader
import zarr

class GenomicZarrDataset(Dataset):
    def __init__(self, zarr_root):
            self.zarr_root = zarr_root
            # Open in read-only mode
            self.root = zarr.open_group(zarr_root, mode='r')
            
            # Load indptr entirely into CPU memory (it's small, just integers)
            print("Loading index pointers into memory...")
            self.indptr = torch.from_numpy(self.root['indptr'][:].astype('int64'))
            self.num_reads = len(self.indptr) - 1
            
            # Keep a handle to the data array
            self.data = self.root['data']

    def _init_worker_store(self):
        if self.data_store is None:
            spec_data = {
                'driver': 'zarr3',
                'kvstore': {
                    'driver': 'file',
                    'path': self.zarr_root # Points to ob007.zarr
                },
                'path': 'data'  # Points to internal array 'data'
            }
            self.data_store = ts.open(spec_data).result()
            
    def __len__(self):
        return self.num_reads

    def __getitem__(self, idx):
        # Fast memory access
        start = self.indptr[idx].item()
        end = self.indptr[idx+1].item()
        
        # Single slice from Zarr
        # Note: standard zarr slicing is often faster than tensorstore for simple local reads
        data_np = self.data[:, start:end] 
        
        return torch.from_numpy(data_np).T

# import os
# import torch
# import tensorstore as ts
# from torch.utils.data import Dataset, DataLoader

# class GenomicZarrDataset(Dataset):
#     def __init__(self, zarr_path, feature_shape=(8,)):
#         self.zarr_root = os.path.abspath(zarr_path)
#         self.feature_shape = feature_shape
        
#         spec_indptr = {
#             'driver': 'zarr3',
#             'kvstore': {'driver': 'file', 'path': self.zarr_root},
#             'path': 'indptr'
#         }
#         # Optimization 1: Use shared memory for the index structure
#         indptr_np = ts.open(spec_indptr).result().read().result()
#         self.indptr = torch.from_numpy(indptr_np).share_memory_()
#         self.num_reads = len(self.indptr) - 1
#         self.data_store = None

#     def _init_worker_store(self):
#         if self.data_store is None:
#             # Optimization 2: Constrain TS threads per worker
#             # This prevents 32 workers from spawning 32 threads each
#             context_spec = {
#                 'data_copy_concurrency': {'limit': 1},
#                 'file_io_concurrency': {'limit': 1}
#             }
            
#             spec_data = {
#                 'driver': 'zarr3',
#                 'kvstore': {'driver': 'file', 'path': self.zarr_root},
#                 'path': 'data',
#                 'context': context_spec 
#             }
#             self.data_store = ts.open(spec_data).result()
            
#     def __len__(self):
#         return self.num_reads

#     def __getitem__(self, idx):
#         self._init_worker_store()
        
#         # Access shared memory tensor
#         start = self.indptr[idx].item()
#         end = self.indptr[idx+1].item()
        
#         data_np = self.data_store[:, start:end].read().result()
#         return torch.from_numpy(data_np).T

ds = GenomicZarrDataset('../data/01_processed/ssl_sets/ob007.zarr')


def collate_ragged(batch):
    """
    Pads sequences to the longest in the batch.
    """
    # batch is a list of Tensors [(L1, F), (L2, F), ...]

    # Get lengths for masking later if needed
    lengths = torch.tensor([x.size(0) for x in batch], dtype=torch.long)

    # Pad sequences (batch_first=True -> [B, MaxLen, F])
    # padding_value=0 is standard, adjust if your padding token is different
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=4)

    return padded_batch, lengths

dl = DataLoader(
    ds, 
    batch_size=1, 
    shuffle=False,
    num_workers=0,
    collate_fn=collate_ragged,
    pin_memory=False, # Critical for H100 throughput
    # prefetch_factor=4
)

from tqdm import tqdm

for batch in tqdm(dl):
    x = batch



