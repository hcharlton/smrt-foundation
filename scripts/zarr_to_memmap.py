import os
import math
import json
import numpy as np
import zarr
import argparse
import yaml

class ShardWriter:
    def __init__(self, output_dir, shard_size, seq_len, num_features, pad_value=0.0):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shape = (shard_size, seq_len, num_features)
        self.buffer = np.full(self.shape, pad_value, dtype=np.float16)
        self.ptr = 0 # tracks sample idx in shard
        self.shard_idx = 0 # tracks 

    def add(self, data):
        """Adds a single segment to the buffer. Flushes if full."""
        # Flush if full
        if self.ptr >= self.shard_size:
            self._flush()
        
        # write step
        # data shape: (seq_len, features)
        # buffer shape: (shard, seq_len, features)
        self.buffer[self.ptr, :data.shape[0], :] = data
        self.ptr += 1

    def _flush(self):
        save_path = os.path.join(self.output_dir, f"shard_{self.shard_idx:05d}.npy")
        data_to_save = self.buffer[:self.ptr] if self.ptr < self.shard_size else self.buffer
        np.save(save_path, data_to_save)
        
        self.shard_idx += 1
        self.ptr = 0
        self.buffer.fill(0.0) # Reset

    def finalize(self):
        if self.ptr > 0:
            self._flush()
            
def build_rc_lookup(config):
    """
    Creates a numpy lookup table for RC conversion based on config maps.
    Returns: np.array where index=input_token, value=rc_token
    """
    token_map = config['data']['token_map']
    rc_map = config['data']['rc_map']
    
    max_token = max(token_map.values())
    
    lookup = np.arange(max_token + 1, dtype=np.int8)
    
    for base, idx in token_map.items():
        if base in rc_map:
            comp_base = rc_map[base]
            if comp_base in token_map:
                lookup[idx] = token_map[comp_base]
                
    return lookup

def get_normalization_vectors(fwd_feats, rev_feats, zarr_attrs):
    """Generates mean/std vectors and metadata dictionary."""
    norm_stats = zarr_attrs.get('log_norm', {})
    num = len(fwd_feats)
    
    # Vectors (Float32 for precision during math)
    fwd_m, fwd_s = np.zeros(num, dtype=np.float32), np.ones(num, dtype=np.float32)
    rev_m, rev_s = np.zeros(num, dtype=np.float32), np.ones(num, dtype=np.float32)
    stats_meta = {}

    for i, (f, r) in enumerate(zip(fwd_feats, rev_feats)):
        if f in norm_stats: fwd_m[i], fwd_s[i] = norm_stats[f]['mean'], norm_stats[f]['std']
        if r in norm_stats: rev_m[i], rev_s[i] = norm_stats[r]['mean'], norm_stats[r]['std']
        
        stats_meta[f"col_{i}_{f}"] = {
            "fwd": {"name": f, "mean": float(fwd_m[i]), "std": float(fwd_s[i])},
            "rev": {"name": r, "mean": float(rev_m[i]), "std": float(rev_s[i])}
        }
    return fwd_m, fwd_s, rev_m, rev_s, stats_meta



def zarr_to_sharded_memmap(
    zarr_path, output_dir, config, 
    fwd_features=['seq', 'fi', 'fp'], 
    rev_features=['seq', 'ri', 'rp'],
    seq_len=4096, shard_size=16384, max_shards=0, 
    use_rc=False
):
    os.makedirs(output_dir, exist_ok=True)
    root = zarr.open(zarr_path, mode='r')
    
    f_mean, f_std, r_mean, r_std, stats_meta = get_normalization_vectors(
        fwd_features, rev_features, root.attrs
    )

    rc_lookup = build_rc_lookup(config)
    
    # collect features
    output_feats = fwd_features + ['mask']
    # make normalization mask (no normalization of the padding or seq)
    is_loggable = np.array([f != 'seq' for f in fwd_features]) # Mask for log1p
    try: seq_idx = fwd_features.index('seq')
    except ValueError: seq_idx = None
    
    # save data schema
    with open(os.path.join(output_dir, "schema.json"), "w") as f:
        json.dump({
            "output_shape": ["Batch", seq_len, len(output_feats)],
            "features": output_feats,
            "normalization": stats_meta,
            "reverse_complement": use_rc,
            "dtype": "float16"
        }, f, indent=4)

    # 2. Setup IO
    writer = ShardWriter(output_dir, shard_size, seq_len, len(output_feats))
    z_data = root['data']
    indptr = root['indptr'][:]
    
    # Prepare indices for loading
    all_feats = root.attrs['features']
    load_indices = list(set([all_feats.index(f) for f in fwd_features + rev_features]))
    idx_map = {orig: i for i, orig in enumerate(load_indices)}
    fwd_local = [idx_map[all_feats.index(f)] for f in fwd_features]
    rev_local = [idx_map[all_feats.index(f)] for f in rev_features]

    # loop over batches
    batch_size = 1000
    total_reads = len(indptr) - 1
    
    for i in range(0, total_reads, batch_size):
        if max_shards and writer.shard_idx > max_shards: break
        
        idx_start, idx_end = indptr[i], indptr[min(i + batch_size, total_reads)]
        
        # Load & Pre-process Batch
        chunk = z_data[idx_start:idx_end, load_indices].astype(np.float32)
        b_fwd, b_rev = chunk[:, fwd_local], chunk[:, rev_local]

        # apply masked log1 and normalize
        np.log1p(b_fwd, out=b_fwd, where=is_loggable)
        np.log1p(b_rev, out=b_rev, where=is_loggable)
        
        b_fwd = ((b_fwd - f_mean) / f_std).astype(np.float16)
        b_rev = ((b_rev - r_mean) / r_std).astype(np.float16)

        # separate into reads
        local_ptr = 0
        for r in range(i, min(i + batch_size, total_reads)):
            r_len = indptr[r+1] - indptr[r]
            read_fwd = b_fwd[local_ptr : local_ptr + r_len]
            read_rev = b_rev[local_ptr : local_ptr + r_len]
            local_ptr += r_len

            # pad and write 
            num_segs = math.ceil(r_len / seq_len)
            for s in range(num_segs):
                start, end = s * seq_len, min((s + 1) * seq_len, r_len)
                
                # --- forward ---
                seg = np.zeros((end-start, len(output_feats)), dtype=np.float16)
                seg[:, :-1] = read_fwd[start:end]
                seg[:, -1] = 1.0 # Mask
                writer.add(seg)

                # --- reverse ---
                seg_rev_data = read_rev[start:end]
                seg_rev_data = np.flip(seg_rev_data, axis=0) # Reverse Time
                if seq_idx is not None:
                    seq_floats = seg_rev_data[:, seq_idx]
                    seq_ints = seq_floats.astype(np.int8)
                    seq_rc = rc_lookup[seq_ints]
                    seg_rev_data[:,seq_idx] = seq_rc.astype(np.float16)          
                seg = np.zeros((end-start, len(output_feats)), dtype=np.float16)
                seg[:, :-1] = seg_rev_data
                seg[:, -1] = 1.0 # Mask
                writer.add(seg)

    writer.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=16384)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--fwd_features", nargs='+', default=['seq', 'fi', 'fp'])
    parser.add_argument("--rev_features", nargs='+', default=['seq', 'ri', 'rp'])
    parser.add_argument("--reverse_complement", action="store_true")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)

    zarr_to_sharded_memmap(
        zarr_path=args.input_path, output_dir=args.output_path, config=config,
        seq_len=args.seq_len, shard_size=args.shard_size, max_shards=args.max_shards,
        fwd_features=args.fwd_features, rev_features=args.rev_features,
        use_rc=args.reverse_complement
    )
