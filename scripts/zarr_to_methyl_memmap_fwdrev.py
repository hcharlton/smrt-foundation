"""
CpG methylation memmap creation with paired forward+reverse kinetics
(6-channel).

Reads from a Zarr store of CpG-bearing reads (cpg_pos.zarr or
cpg_neg.zarr) and writes sharded `.npy` files containing one
forward-strand-anchored window per CpG site. Each output row has shape
(context, 6) with the channel layout:

    [seq, fi, fp, ri, rp, mask]

Same layout as `scripts.zarr_to_memmap_fwdrev`, the SSL counterpart;
the encoder consumes both with no architectural difference. The mask
channel is 0.0 throughout each row because every base in a CpG window
is real data.

Reverse-strand kinetics indexing
--------------------------------
Mirrors the v2 fix from `zarr_to_methyl_memmap_v2.extract_cpg_windows`:
for a forward window `[win_start, win_end)` on a read of length `L`,
the reverse strand window is `[L - win_end, L - win_start)`, and the
kinetics there are stored in reverse-strand order. To align them with
the forward-strand orientation:

    ri_aligned = ri[L - win_end : L - win_start][::-1]
    rp_aligned = rp[L - win_end : L - win_start][::-1]

`find_cpg_positions` (copied verbatim from v2) only yields windows
where both the forward and reverse spans fit inside the read, so the
slice indices are always valid.

What this script does NOT do (vs. v2)
-------------------------------------
v2 writes two samples per CpG (a forward-anchored 4-channel row and a
reverse-anchored 4-channel row). This script writes ONE 6-channel
sample per CpG: forward-strand seq + forward kinetics (fi/fp) + aligned
reverse kinetics (ri/rp) + zero mask. No RC view of the same CpG.

Usage
-----
    python -m scripts.zarr_to_methyl_memmap_fwdrev \\
        --input_path data/01_processed/ssl_sets/cpg_pos.zarr \\
        --output_path data/01_processed/val_sets/cpg_pos_fwdrev.memmap \\
        --config_path configs/data.yaml \\
        --context 32
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml
import zarr


N_OUTPUT_FEATS = 6  # [seq, fi, fp, ri, rp, mask]
PAD_IDX = 5
OUTPUT_FEATS = ['seq', 'fi', 'fp', 'ri', 'rp', 'mask']

# Token IDs from configs/data.yaml token_map. Hardcoded here to mirror
# v2 (which also hardcoded C=1, G=2 in find_cpg_positions); avoids a
# config dependency in the inner loop.
C_TOKEN = 1
G_TOKEN = 2


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


def find_cpg_positions(seq, context):
    """Yield (win_start, win_end, rev_start, rev_end) for each valid CpG.

    Copied verbatim from `zarr_to_methyl_memmap_v2.find_cpg_positions`.
    A CpG is "valid" iff both the forward window and the
    reverse-indexed window fit inside the read.
    """
    pad = (context - 2) // 2
    L = len(seq)
    for i in range(L - 1):
        if seq[i] == C_TOKEN and seq[i + 1] == G_TOKEN:
            win_start = i - pad
            win_end = i + 2 + pad
            rev_start = L - win_end
            rev_end = L - win_start
            if (win_start >= 0 and win_end <= L and
                    rev_start >= 0 and rev_end <= L):
                yield win_start, win_end, rev_start, rev_end


def extract_cpg_windows_fwdrev(read_data, feature_idx, context):
    """Extract one 6-channel window per CpG site in this read.

    Returns a list of (context, 6) float16 arrays. The mask channel is
    left at 0.0 because every base in a CpG window is real data.
    """
    seq = read_data[:, feature_idx['seq']]
    fi = read_data[:, feature_idx['fi']]
    fp = read_data[:, feature_idx['fp']]
    ri = read_data[:, feature_idx['ri']]
    rp = read_data[:, feature_idx['rp']]

    rows = []
    for win_start, win_end, rev_start, rev_end in find_cpg_positions(seq, context):
        row = np.zeros((context, N_OUTPUT_FEATS), dtype=np.float16)
        row[:, 0] = seq[win_start:win_end]
        row[:, 1] = fi[win_start:win_end]
        row[:, 2] = fp[win_start:win_end]
        row[:, 3] = ri[rev_start:rev_end][::-1]
        row[:, 4] = rp[rev_start:rev_end][::-1]
        # mask channel stays 0.0 (full window is real data)
        rows.append(row)
    return rows


def zarr_to_methyl_memmap_fwdrev(
    zarr_path,
    output_dir,
    context=32,
    shard_size=4194304,
    max_shards=0,
    val_pct=0.2,
    seed=42,
):
    """Convert CpG-bearing Zarr to 6-channel labeled shards.

    Args:
        zarr_path: Path to cpg_pos.zarr or cpg_neg.zarr.
        output_dir: Output directory (train/ and val/ subdirs are created).
        context: Window size around each CpG.
        shard_size: Max rows per shard file.
        max_shards: Stop after this many train shards (0 = no limit).
        val_pct: Fraction of source reads routed to the val/ split.
        seed: RNG seed for the train/val split.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

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
    feature_idx = {f: all_feats.index(f) for f in needed}

    schema = {
        "output_shape": ["N", context, N_OUTPUT_FEATS],
        "features": OUTPUT_FEATS,
        "pad_idx": PAD_IDX,
        "context": context,
        "dtype": "float16",
        "source": "zarr_to_methyl_memmap_fwdrev",
        "rev_kinetics_indexing": "ri[L-win_end:L-win_start][::-1]",
        "samples_per_cpg": 1,
    }
    for split_dir in (train_dir, val_dir):
        with open(os.path.join(split_dir, "schema.json"), "w") as f:
            json.dump(schema, f, indent=4)

    # Per-read train/val mask (deterministic given seed).
    total_reads = len(indptr) - 1
    rng = np.random.RandomState(seed)
    is_val = np.zeros(total_reads, dtype=bool)
    val_indices = rng.choice(total_reads, int(total_reads * val_pct), replace=False)
    is_val[val_indices] = True

    train_writer = ShardWriter(train_dir, shard_size, context)
    val_writer = ShardWriter(val_dir, shard_size, context)

    batch_size = 500
    for batch_start in range(0, total_reads, batch_size):
        if max_shards and train_writer.shard_idx >= max_shards:
            break
        batch_end = min(batch_start + batch_size, total_reads)
        chunk_start = int(indptr[batch_start])
        chunk_end = int(indptr[batch_end])
        if chunk_start == chunk_end:
            continue

        chunk = z_data[chunk_start:chunk_end, :]

        for r in range(batch_start, batch_end):
            r_start = int(indptr[r]) - chunk_start
            r_end = int(indptr[r + 1]) - chunk_start
            read_data = chunk[r_start:r_end, :]

            writer = val_writer if is_val[r] else train_writer
            for row in extract_cpg_windows_fwdrev(read_data, feature_idx, context):
                writer.add(row)

    train_count = train_writer.finalize()
    val_count = val_writer.finalize()
    print(f"[zarr_to_methyl_memmap_fwdrev] done. train: {train_count} CpGs, "
          f"val: {val_count} CpGs in {output_dir}")
    return train_count, val_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="6-channel paired-kinetics CpG methylation memmap creation"
    )
    parser.add_argument("--input_path", required=True, help="Zarr store path")
    parser.add_argument("--output_path", required=True, help="Output memmap directory")
    parser.add_argument("--config_path", required=True, help="data.yaml config path (unused, kept for parity)")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--shard_size", type=int, default=4194304)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--val_pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        _ = yaml.safe_load(f)

    zarr_to_methyl_memmap_fwdrev(
        zarr_path=args.input_path,
        output_dir=args.output_path,
        context=args.context,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
        val_pct=args.val_pct,
        seed=args.seed,
    )
