"""
SSL memmap creation with paired forward+reverse kinetics (6-channel).

Reads from a Zarr store produced by `scripts.bam_to_zarr` and emits
sharded `.npy` shards laid out for `ShardedMemmapDataset`. Each output
row is a (context, 6) float16 array with channel layout:

    [seq, fi, fp, ri, rp, mask]

where `fi/fp` are forward-strand IPD/PW at each position and `ri/rp`
are reverse-strand IPD/PW *aligned to the same forward-strand
position* (the v2 fix below). The mask channel is 1.0 = pad, 0.0 =
real, matching the project convention.

Reverse-strand kinetics indexing
--------------------------------
PacBio per-base tags `ri` and `rp` are stored in reverse-strand order
in the Zarr: `ri[i]` is the reverse-strand IPD at forward-strand
position `L-1-i`. To get reverse kinetics aligned to a forward window
`[start, end)` of a read of length `L`, the correct slice is

    ri_aligned = ri[L-end : L-start][::-1]
    rp_aligned = rp[L-end : L-start][::-1]

This mirrors `scripts.zarr_to_methyl_memmap_v2.extract_cpg_windows`,
which fixed exactly this bug in the legacy methyl pipeline (commit
6d9ea32). The older SSL script `zarr_to_memmap_instanceNorm.py` does
`np.flip(read_rev[start:end])` instead, which is benign there only
because the script writes the reverse view as a separate sample (no
positional pairing with the forward view). Once fwd and rev become
paired channels of one sample, the v2 indexing is required for
correctness.

What this script does NOT do
----------------------------
- No normalization. Outputs raw zarr values (uint8 cast to float16),
  consistent with the project convention that on-disk data is raw and
  normalization is the dataloader's job.
- No reverse-complement augmentation. With fwd+rev paired, an RC
  augmentation reduces to a permutation (reverse along time, RC the
  seq channel, swap fi<->ri / fp<->rp). That is a dataloader-side
  operation if needed; nothing on disk is duplicated.
- No quality filtering. The existing `--filter_qual` flag of the
  legacy script is omitted; if filtering is wanted it can be added
  later.

Usage
-----
    python -m scripts.zarr_to_memmap_fwdrev \\
        --input_path data/01_processed/ssl_sets/yoran.zarr \\
        --output_path data/01_processed/ssl_sets/yoran_fwdrev.memmap \\
        --config_path configs/data.yaml \\
        --context 4096
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import yaml
import zarr


N_OUTPUT_FEATS = 6  # [seq, fi, fp, ri, rp, mask]
PAD_IDX = 5         # mask channel
PAD_VALUE = np.float16(1.0)
DATA_VALUE = np.float16(0.0)
OUTPUT_FEATS = ['seq', 'fi', 'fp', 'ri', 'rp', 'mask']


class ShardWriter:
    """Buffers (context, 6) float16 rows; flushes shard_NNNNN.npy."""

    def __init__(self, output_dir, shard_size, context):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.context = context
        self.buffer = np.zeros((shard_size, context, N_OUTPUT_FEATS), dtype=np.float16)
        self.ptr = 0
        self.shard_idx = 0
        self.total_written = 0

    def add(self, row):
        """Add a single (context, N_OUTPUT_FEATS) row to the buffer."""
        self.buffer[self.ptr] = row
        self.ptr += 1
        if self.ptr >= self.shard_size:
            self._flush()

    def _flush(self):
        if self.ptr == 0:
            return
        path = os.path.join(self.output_dir, f"shard_{self.shard_idx:05d}.npy")
        np.save(path, self.buffer[:self.ptr])
        self.total_written += self.ptr
        self.shard_idx += 1
        self.ptr = 0
        self.buffer[:] = 0

    def finalize(self):
        self._flush()
        return self.total_written


def _build_segment(read_data, feature_idx, start, end, context):
    """Build one (context, 6) row covering forward window [start, end).

    `read_data` is the full read of shape (L, n_loaded_features).
    Reverse kinetics are pulled from `ri[L-end:L-start][::-1]` and
    `rp[L-end:L-start][::-1]` per the v2 alignment rule.
    """
    L = read_data.shape[0]
    seg_len = end - start
    rev_start = L - end
    rev_end = L - start

    row = np.zeros((context, N_OUTPUT_FEATS), dtype=np.float16)
    row[:seg_len, 0] = read_data[start:end, feature_idx['seq']]
    row[:seg_len, 1] = read_data[start:end, feature_idx['fi']]
    row[:seg_len, 2] = read_data[start:end, feature_idx['fp']]
    row[:seg_len, 3] = read_data[rev_start:rev_end, feature_idx['ri']][::-1]
    row[:seg_len, 4] = read_data[rev_start:rev_end, feature_idx['rp']][::-1]
    # mask: real for [0, seg_len), pad for [seg_len, context)
    row[:seg_len, PAD_IDX] = DATA_VALUE
    row[seg_len:, PAD_IDX] = PAD_VALUE
    return row


def zarr_to_memmap_fwdrev(
    zarr_path,
    output_dir,
    context=4096,
    shard_size=16384,
    max_shards=0,
):
    """Convert Zarr to 6-channel paired-kinetics SSL shards.

    Args:
        zarr_path: Path to Zarr store from bam_to_zarr.py.
        output_dir: Output directory (will be created).
        context: Per-row time dimension. Reads of length > context are
            split into ceil(L/context) consecutive segments; the final
            segment is right-padded.
        shard_size: Rows per shard file.
        max_shards: Stop after this many shards (0 = no limit).
    """
    os.makedirs(output_dir, exist_ok=True)

    root = zarr.open(zarr_path, mode='r')
    z_data = root['data']
    indptr = root['indptr'][:]

    all_feats = list(root.attrs['features'])
    needed = ['seq', 'fi', 'fp', 'ri', 'rp']
    missing = [f for f in needed if f not in all_feats]
    if missing:
        raise ValueError(
            f"Zarr at {zarr_path} is missing required features: {missing}. "
            f"Has: {all_feats}"
        )

    # Map needed features to their column indices in the source zarr.
    src_indices = [all_feats.index(f) for f in needed]
    # Local indices within the loaded sub-array (in order of `needed`).
    feature_idx = {f: i for i, f in enumerate(needed)}

    # Schema sidecar — read by build_ssl_pair_val.py to find pad_idx.
    schema = {
        "output_shape": ["N", context, N_OUTPUT_FEATS],
        "features": OUTPUT_FEATS,
        "pad_idx": PAD_IDX,
        "dtype": "float16",
        "context": context,
        "shard_size": shard_size,
        "source": "zarr_to_memmap_fwdrev",
        "rev_kinetics_indexing": "ri[L-end:L-start][::-1]",
    }
    with open(os.path.join(output_dir, "schema.json"), "w") as f:
        json.dump(schema, f, indent=4)

    writer = ShardWriter(output_dir, shard_size, context)

    total_reads = len(indptr) - 1
    batch_size = 1000

    for batch_start in range(0, total_reads, batch_size):
        if max_shards and writer.shard_idx >= max_shards:
            break
        batch_end = min(batch_start + batch_size, total_reads)
        chunk_start = int(indptr[batch_start])
        chunk_end = int(indptr[batch_end])
        if chunk_start == chunk_end:
            continue

        # Load the kinetics columns for the whole batch in one call.
        chunk = z_data[chunk_start:chunk_end, src_indices].astype(np.float32)

        for r in range(batch_start, batch_end):
            if max_shards and writer.shard_idx >= max_shards:
                break
            r_start = int(indptr[r]) - chunk_start
            r_end = int(indptr[r + 1]) - chunk_start
            read_data = chunk[r_start:r_end]
            L = read_data.shape[0]
            if L == 0:
                continue

            # Segment the read into ceil(L/context) consecutive windows.
            n_segs = math.ceil(L / context)
            for s in range(n_segs):
                start = s * context
                end = min(start + context, L)
                row = _build_segment(read_data, feature_idx, start, end, context)
                writer.add(row)
                if max_shards and writer.shard_idx >= max_shards:
                    break

    total = writer.finalize()
    print(f"[zarr_to_memmap_fwdrev] done. wrote {total} rows in "
          f"{writer.shard_idx} shards to {output_dir}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-channel paired-kinetics SSL memmap creation")
    parser.add_argument("--input_path", required=True, help="Zarr store path")
    parser.add_argument("--output_path", required=True, help="Output memmap directory")
    parser.add_argument("--config_path", required=True, help="data.yaml config path (unused but kept for parity)")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=16384)
    parser.add_argument("--max_shards", type=int, default=0)

    args = parser.parse_args()
    # config is loaded for parity with the legacy script; the fwdrev
    # pipeline derives token/feature info from the zarr's own attrs.
    with open(args.config_path, 'r') as f:
        _ = yaml.safe_load(f)

    zarr_to_memmap_fwdrev(
        zarr_path=args.input_path,
        output_dir=args.output_path,
        context=args.context,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
    )
