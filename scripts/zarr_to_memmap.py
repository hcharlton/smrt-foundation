import os
import math
import numpy as np
import zarr
from tqdm import tqdm

def zarr_to_sharded_memmap_channels_last(
    zarr_path: str,
    output_dir: str,
    seq_len: int = 4096,
    shard_size: int = 16384,
    pad_value: int = 0
):
    os.makedirs(output_dir, exist_ok=True)
    
    root = zarr.open(zarr_path, mode='r')
    z_data = root['data'] 
    indptr = root['indptr'][:]
    
    
    # target output: (N, Seq_Len, Features + 1) -> +1 for padding mask
    # adopted channel last as standard
    
    total_reads = len(indptr) - 1
    current_shard = []
    shard_idx = 0
    batch_size = 1000 
    
    for i in tqdm(range(0, total_reads, batch_size), desc="Processing reads"):
        end_batch = min(i + batch_size, total_reads)
        idx_start = indptr[i]
        idx_end = indptr[end_batch]
        
        # Load batch: (Batch_Time, Features)
        chunk_data = z_data[idx_start:idx_end, :]
        local_start = 0
        
        for r in range(i, end_batch):
            r_len = indptr[r+1] - indptr[r]
            
            # Extract read: (r_lesn, Features)
            read = chunk_data[local_start : local_start + r_len, :]
            local_start += r_len
            
            num_segments = math.ceil(r_len / seq_len)
            
            for seg in range(num_segments):
                seg_start = seg * seq_len
                seg_end = min(seg_start + seq_len, r_len)
                
                # 1. Extract Data
                segment = read[seg_start:seg_end, :]
                current_seg_len = segment.shape[0]
                
                # 2. Create Mask: (current_seg_len, 1)
                mask_segment = np.ones((current_seg_len, 1), dtype=np.uint8)
                
                # 3. Stack Features + Mask -> (current_seg_len, Features+1)
                # Concatenate along axis 1 (columns)
                combined_segment = np.hstack([segment, mask_segment]).astype(np.uint8)
                
                # 4. Pad Time Dimension (axis 0)
                if current_seg_len < seq_len:
                    pad_width = seq_len - current_seg_len
                    # Pad axis 0 with pad_value, axis 1 (features) with 0
                    combined_segment = np.pad(
                        combined_segment, 
                        ((0, pad_width), (0, 0)), 
                        'constant', 
                        constant_values=pad_value
                    )
                    # Re-cast to ensure strict uint8
                    combined_segment = combined_segment.astype(np.uint8)

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
    zarr_to_sharded_memmap_channels_last(
        zarr_path='../data/01_processed/ssl_sets/ob007.zarr',
        output_dir='./ob007_2.memmap',
        seq_len=4096,
        shard_size=8192
    )
