import os
import math
import numpy as np
import zarr
from tqdm import tqdm

def zarr_to_sharded_memmap_masked(
    zarr_path: str,
    output_dir: str,
    seq_len: int = 4096,
    shard_size: int = 16384,
    pad_value: int = 0, # Value for data padding (ignored by model due to mask)
):
    os.makedirs(output_dir, exist_ok=True)
    
    root = zarr.open(zarr_path, mode='r')
    z_data = root['data'] 
    indptr = root['indptr'][:]
    
    # +1 for the mask channel
    n_features = z_data.shape[0]
    output_features = n_features + 1
    
    total_reads = len(indptr) - 1
    current_shard = []
    shard_idx = 0
    batch_size = 1000 
    
    print(f"Converting to shape: (N, {output_features}, {seq_len}) [Data + Mask]")

    for i in tqdm(range(0, total_reads, batch_size), desc="Processing reads"):
        end_batch = min(i + batch_size, total_reads)
        
        idx_start = indptr[i]
        idx_end = indptr[end_batch]
        
        # Load batch into RAM
        chunk_data = z_data[:, idx_start:idx_end]
        local_start = 0
        
        for r in range(i, end_batch):
            r_len = indptr[r+1] - indptr[r]
            
            # Extract read: (n_features, r_len)
            read = chunk_data[:, local_start : local_start + r_len]
            local_start += r_len
            
            num_segments = math.ceil(r_len / seq_len)
            
            for seg in range(num_segments):
                seg_start = seg * seq_len
                seg_end = min(seg_start + seq_len, r_len)
                
                # 1. Extract Data Segment
                # shape: (n_features, current_seg_len)
                segment = read[:, seg_start:seg_end]
                current_seg_len = segment.shape[1]
                
                # 2. Create Mask Segment
                # shape: (1, current_seg_len) - All 1s because this is real data
                mask_segment = np.ones((1, current_seg_len), dtype=np.uint8)
                
                # 3. Stack Data and Mask
                # shape: (n_features + 1, current_seg_len)
                combined_segment = np.vstack([segment, mask_segment])
                
                # 4. Pad if necessary
                if current_seg_len < seq_len:
                    pad_width = seq_len - current_seg_len
                    # Pad time dimension (axis 1) with 0
                    # We pad the mask with 0 as well, correctly marking it as invalid
                    combined_segment = np.pad(
                        combined_segment, 
                        ((0, 0), (0, pad_width)), 
                        'constant', 
                        constant_values=pad_value
                    )
                
                current_shard.append(combined_segment)
                
                if len(current_shard) >= shard_size:
                    arr = np.stack(current_shard)
                    save_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
                    np.save(save_path, arr)
                    current_shard = []
                    shard_idx += 1

    if current_shard:
        arr = np.stack(current_shard)
        save_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
        np.save(save_path, arr)

if __name__ == "__main__":
    zarr_to_sharded_memmap_masked(
        zarr_path='../data/01_processed/ssl_sets/ob007.zarr',
        output_dir='./ob007.memmap',
        seq_len=4096,
        shard_size=8192
    )