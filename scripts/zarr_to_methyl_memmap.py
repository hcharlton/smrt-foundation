import os
import math
import json
import numpy as np
import zarr
import argparse
import yaml
from smrt_foundation.normalization import build_rc_lookup, normalize_read_mad

class ShardWriter:
    def __init__(self, output_dir, shard_size, context, num_features):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.pad_value = 1.0
        self.data_value = 0.0
        self.shape = (shard_size, context, num_features)
        self.buffer = np.zeros(self.shape, dtype=np.float16)
        self.buffer[:,:,-1] = self.pad_value
        self.ptr = 0
        self.shard_idx = 0

    def add(self, data):
        if self.ptr >= self.shard_size:
            self._flush()
        self.buffer[self.ptr, :data.shape[0], :] = data
        self.ptr += 1

    def _flush(self):
        save_path = os.path.join(self.output_dir, f"shard_{self.shard_idx:05d}.npy")
        data_to_save = self.buffer[:self.ptr] if self.ptr < self.shard_size else self.buffer
        np.save(save_path, data_to_save)
        self.shard_idx += 1
        self.ptr = 0
        self.buffer.fill(0.0)
        self.buffer[:,:,-1] = self.pad_value

    def finalize(self):
        if self.ptr > 0:
            self._flush()

def extract_pattern_windows_2d(read_data, channel_idx, window_size, pattern):
    seq = read_data[:, channel_idx]
    p_len = len(pattern)
    if (window_size - p_len) % 2 != 0:
        raise ValueError("Difference between window_size and pattern length must be even.")
    pad = (window_size - p_len) // 2
    p_view = np.lib.stride_tricks.sliding_window_view(seq, p_len)
    matches = np.nonzero(np.all(p_view == pattern, axis=1))[0]
    starts = matches - pad
    valid = starts[(starts >= 0) & (starts + window_size <= len(seq))]
    if not len(valid):
        return np.empty((0, window_size, read_data.shape[1]), dtype=read_data.dtype)
    full_view = np.lib.stride_tricks.sliding_window_view(read_data, window_size, axis=0)
    return np.swapaxes(full_view, 1, 2)[valid]

def zarr_to_sharded_memmap(
    zarr_path, output_dir, config, 
    fwd_features=['seq', 'fi', 'fp'], 
    rev_features=['seq', 'ri', 'rp'],
    context=4096, shard_size=16384, max_shards=0, 
    use_rc=False,
    normalize=False,
):
    os.makedirs(output_dir, exist_ok=True)
    root = zarr.open(zarr_path, mode='r')
    
    rc_lookup = build_rc_lookup(config)
    seq_map = config['data']['token_map']
    cpg_pattern = [seq_map['C'], seq_map['G']]
    
    output_feats = fwd_features + ['mask']
    is_continuous_fwd = np.array([f != 'seq' for f in fwd_features])
    is_continuous_rev = np.array([f != 'seq' for f in rev_features])

    try: seq_idx_fwd = fwd_features.index('seq')
    except ValueError: seq_idx_fwd = None

    try: seq_idx_rev = rev_features.index('seq')
    except ValueError: seq_idx_rev = None
    
    # save data schema
    with open(os.path.join(output_dir, "schema.json"), "w") as f:
        json.dump({
            "output_shape": ["Batch", context, len(output_feats)],
            "features": output_feats,
            "normalization": 'per-read-MAD' if normalize else 'raw',
            "reverse_complement": use_rc,
            "dtype": "float16"
        }, f, indent=4)

    writer = ShardWriter(output_dir, shard_size, context, len(output_feats))
    z_data = root['data']
    indptr = root['indptr'][:]
    
    # Prepare indices for loading
    all_feats = root.attrs['features']
    load_indices = list(set([all_feats.index(f) for f in fwd_features + rev_features]))
    idx_map = {orig: i for i, orig in enumerate(load_indices)}
    fwd_local = [idx_map[all_feats.index(f)] for f in fwd_features]
    rev_local = [idx_map[all_feats.index(f)] for f in rev_features]

    batch_size = 1000
    total_reads = len(indptr) - 1
    
    for i in range(0, total_reads, batch_size):
        if max_shards and writer.shard_idx > max_shards: break
        
        idx_start, idx_end = indptr[i], indptr[min(i + batch_size, total_reads)]
        chunk = z_data[idx_start:idx_end, load_indices].astype(np.float32)
        b_fwd, b_rev = chunk[:, fwd_local], chunk[:, rev_local]

        local_ptr = 0
        for r in range(i, min(i + batch_size, total_reads)):
            r_len = indptr[r+1] - indptr[r]
            read_fwd = b_fwd[local_ptr : local_ptr + r_len]
            read_rev = b_rev[local_ptr : local_ptr + r_len]
            local_ptr += r_len

            if normalize:
                read_fwd = normalize_read_mad(read_fwd, is_continuous_mask=is_continuous_fwd).astype(np.float16)
                read_rev = normalize_read_mad(read_rev, is_continuous_mask=is_continuous_rev).astype(np.float16)

            fwd_windows = extract_pattern_windows_2d(read_fwd, seq_idx_fwd, context, cpg_pattern)
            for w in fwd_windows:
                seg = np.zeros((context, len(output_feats)), dtype=np.float16)
                seg[:, :-1] = w
                seg[:, -1] = writer.data_value
                writer.add(seg)

            read_rev = np.flip(read_rev, axis=0)
            if use_rc and (seq_idx_rev is not None):
                seq_ints = read_rev[:, seq_idx_rev].astype(np.int8)
                read_rev[:, seq_idx_rev] = rc_lookup[seq_ints].astype(np.float16)

            rev_windows = extract_pattern_windows_2d(read_rev, seq_idx_rev, context, cpg_pattern)
            for w in rev_windows:
                seg = np.zeros((context, len(output_feats)), dtype=np.float16)
                seg[:, :-1] = w
                seg[:, -1] = writer.data_value
                writer.add(seg)

    writer.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=2e20)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--fwd_features", nargs='+', default=['seq', 'fi', 'fp'])
    parser.add_argument("--rev_features", nargs='+', default=['seq', 'ri', 'rp'])
    parser.add_argument("--reverse_complement", action="store_true")
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)

    zarr_to_sharded_memmap(
        zarr_path=args.input_path, output_dir=args.output_path, config=config,
        context=args.context, shard_size=args.shard_size, max_shards=args.max_shards,
        fwd_features=args.fwd_features, rev_features=args.rev_features,
        use_rc=args.reverse_complement, normalize=args.normalize,
    )