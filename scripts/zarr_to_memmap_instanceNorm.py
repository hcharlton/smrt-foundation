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
        self.buffer[:,:,-1] = self.pad_value # padding channel set to pdding value
        self.ptr = 0 # tracks sample idx in shard
        self.shard_idx = 0 # tracks

    def add(self, data):
        """Adds a single segment to the buffer. Flushes if full."""
        # Flush if full
        if self.ptr >= self.shard_size:
            self._flush()
        
        # write step
        # data shape: (context, features)
        # buffer shape: (shard, context, features)
        self.buffer[self.ptr, :data.shape[0], :] = data
        self.ptr += 1

    def _flush(self):
        save_path = os.path.join(self.output_dir, f"shard_{self.shard_idx:05d}.npy")
        data_to_save = self.buffer[:self.ptr] if self.ptr < self.shard_size else self.buffer
        np.save(save_path, data_to_save)
        
        self.shard_idx += 1
        self.ptr = 0
        self.buffer.fill(0.0) # Reset to null
        self.buffer[:,:,-1] = self.pad_value # mask the entirety

    def finalize(self):
        if self.ptr > 0:
            self._flush()

def zarr_to_sharded_memmap(
    zarr_path, output_dir, config, 
    fwd_features=['seq', 'fi', 'fp'], 
    rev_features=['seq', 'ri', 'rp'],
    context=4096, shard_size=16384, max_shards=0, 
    use_rc=False,
    normalize = False,
):
    os.makedirs(output_dir, exist_ok=True)
    root = zarr.open(zarr_path, mode='r')
    
    rc_lookup = build_rc_lookup(config)
    print(rc_lookup)
    
    # collect features
    output_feats = fwd_features + ['mask']
    # make normalization mask (no normalization of the padding or seq)
    is_continuous_fwd = np.array([f != 'seq' for f in fwd_features])
    is_continuous_rev = np.array([f != 'seq' for f in rev_features])

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

    # 2. Setup IO
    writer = ShardWriter(output_dir, shard_size, context, len(output_feats))
    z_data = root['data']
    indptr = root['indptr'][:]
    
    # Prepare indices for loading
    all_feats = root.attrs['features']
    # find the col indices for the features we actually will use
    load_indices = list(set([all_feats.index(f) for f in fwd_features + rev_features]))
    # make a new map
    idx_map = {orig: i for i, orig in enumerate(load_indices)}
    # get indices in the loaded array for the forward and reverse views
    fwd_local = [idx_map[all_feats.index(f)] for f in fwd_features]
    rev_local = [idx_map[all_feats.index(f)] for f in rev_features]

    # write in batches
    batch_size = 1000
    total_reads = len(indptr) - 1
    
    for i in range(0, total_reads, batch_size):
        if max_shards and writer.shard_idx > max_shards: break
        
        # define a the actual indices in the data for our batch of b reads
        idx_start, idx_end = indptr[i], indptr[min(i + batch_size, total_reads)]
        
        # get a chunk from the zarr of b reads
        chunk = z_data[idx_start:idx_end, load_indices].astype(np.float32)
        # get the forward and reverse batches (each have b reads)
        b_fwd, b_rev = chunk[:, fwd_local], chunk[:, rev_local]

        # apply log1 and normalize (only cont. features)
        # np.log1p(b_fwd, out=b_fwd, where=is_continuous_fwd)
        # np.log1p(b_rev, out=b_rev, where=is_continuous_rev)

        # separate into reads
        local_ptr = 0
        for r in range(i, min(i + batch_size, total_reads)):
            r_len = indptr[r+1] - indptr[r] # read length
            read_fwd = b_fwd[local_ptr : local_ptr + r_len] # a single fwd read
            read_rev = b_rev[local_ptr : local_ptr + r_len] # a single rev read
            local_ptr += r_len

            # normalize per-read
            if normalize:
                read_fwd = normalize_read_mad(
                    read_fwd, is_continuous_mask=is_continuous_fwd
                    ).astype(np.float16)
                read_rev = normalize_read_mad(
                    read_rev, is_continuous_mask=is_continuous_rev
                    ).astype(np.float16)

            # pad and write into padded blocks
            num_segs = math.ceil(r_len / context)
            for s in range(num_segs):
                start, end = s * context, min((s + 1) * context, r_len)
                
                # --- forward ---
                seg = np.zeros((end-start, len(output_feats)), dtype=np.float16)
                seg[:, :-1] = read_fwd[start:end]
                seg[:, -1] = writer.data_value # overwrite mask to activate data segment
                writer.add(seg)

                # --- reverse ---
                seg_rev_data =np.flip(read_rev[start:end], axis=0)
                if use_rc and (seq_idx_rev is not None):
                    seq_floats = seg_rev_data[:, seq_idx_rev]
                    seq_ints = seq_floats.astype(np.int8)
                    seq_rc = rc_lookup[seq_ints]
                    seg_rev_data[:,seq_idx_rev] = seq_rc.astype(np.float16)          
                seg = np.zeros((end-start, len(output_feats)), dtype=np.float16)
                seg[:, :-1] = seg_rev_data
                seg[:, -1] = writer.data_value # overwrite mask to activate data segment
                writer.add(seg)

    writer.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=16384)
    parser.add_argument("--context", type=int, default=4096)
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
