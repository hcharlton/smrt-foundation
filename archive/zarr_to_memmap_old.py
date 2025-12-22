import os
import math
import numpy as np
import zarr
import sys
from tqdm import tqdm
import argparse
from smrt_foundation.utils import parse_yaml

# --- Profile boilerplate begin ---
import os
import builtins
import atexit

if os.environ.get('TimeLINE_PROFILE'):
    # only imports line_profiler if the env var is set
    from line_profiler import LineProfiler
    lp = LineProfiler()
    
    # decorator definition
    def profile(func):
        return lp(func)
        
    # save log
    def save_profile():
        # filename
        profiler_out_path = 'zarr_to_memmap.lprof'
        lp.dump_stats(profiler_out_path) 
        print("\n[Profiler] Stats saved to 'profile_output.lprof'")
    
    atexit.register(save_profile)
else:
    def profile(func):
        return func
builtins.profile = profile
# --- Profile boilerplate end ---

@profile
def zarr_to_sharded_memmap(
    zarr_path: str,
    output_dir: str,
    max_shards: int,
    config,
    target_Features = ['seq', 'fi'],
    seq_len: int = 4096,
    shard_size: int = 16384,
    pad_value: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    
    root = zarr.open(zarr_path, mode='r')
    log_norm = root.attrs['log_norm']
    tag_to_idx = root.attrs['tag_to_idx']
    z_data = root['data'] 
    indptr = root['indptr'][:]
    
    
    # target output: (N, Seq_Len, Features + 1) -> +1 for padding mask
    # adopted channel last as standard
    
    total_reads = len(indptr) - 1
    current_shard = []
    shard_idx = 0
    batch_size = 1000
    # generate vector of normalization constants
    kinetics_features = config['data']['kinetics_features']
    
    for i in range(0, total_reads, batch_size):
        if shard_idx > max_shards and max_shards != 0:
            break
        end_batch = min(i + batch_size, total_reads)
        idx_start = indptr[i]
        idx_end = indptr[end_batch]
        
        # Load batch: (Batch_Time, Features)
        forward_features = ['seq', 'fi', 'fp']
        reverse_features = ['seq', 'ri', 'rp']
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
                
                # get data
                segment = read[seg_start:seg_end, :]
                current_seg_len = segment.shape[0]
                
                # make mask (currently contains no information)
                mask_segment = np.ones((current_seg_len, 1), dtype=np.uint8)
                
                # stack mask and features
                combined_segment = np.hstack([segment, mask_segment]).astype(np.uint8)
                
                # add information to mask
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

def main():
    parser = argparse.ArgumentParser(
        description='Converts a large zarr file into padded binary numpy tensors in shards'
    )
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='path to input zarr file')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='path for output -> will create a directory')
    parser.add_argument('--optional_tags',
                        type=str,
                        default=[],
                        help='which tags to include in the output. seq and kinetics always included')
    parser.add_argument('--config_path',
                        type=str,
                        required=True,
                        help='path to config file')
    parser.add_argument('--shard_size',
                        type=int,
                        required=True,
                        help='number of samples per shard'
                        )
    parser.add_argument('--max_shards',
                        type=int,
                        required=True,
                        help='the max number of shards to output. 0 -> process entire zarr file into shards' )
    args = parser.parse_args()
    if args.max_shards < 0:
        print("Error: n_reads should be positive or 0 (to indicate all reads).")
        sys.exit(1)

    config = parse_yaml(args.config_path)
    context = config['model']['params']['context']
    zarr_to_sharded_memmap(
        zarr_path=args.input_path,
        output_dir=os.path.expanduser(args.output_path),
        config=config,
        seq_len=context,
        shard_size=args.shard_size,
        max_shards= args.max_shards
    )
if __name__ == "__main__":
    main()
