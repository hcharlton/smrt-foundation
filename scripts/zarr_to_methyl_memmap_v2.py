"""
CpG methylation memmap creation (v2) — clean rewrite mirroring the
unmodified legacy script's extraction logic.

Reads from Zarr (BAM→Zarr fidelity is verified by existing tests),
extracts forward-strand CpG windows with correctly aligned kinetics,
and writes sharded .npy files for LabeledMemmapDataset.

Key differences from zarr_to_methyl_memmap.py (v1):
  - Forward-strand CpGs only (matching legacy)
  - Reverse kinetics use explicit reverse indexing (ri[L-end:L-start])
    instead of np.flip on the whole read
  - Writes both fwd (fi/fp) and rev (aligned ri/rp) views per CpG
  - Simple direct slicing — no stride tricks or complex index mapping

Usage:
    python -m scripts.zarr_to_methyl_memmap_v2 \\
        --input_path data/01_processed/ssl_sets/cpg_pos.zarr \\
        --output_path data/01_processed/val_sets/cpg_pos_v2.memmap \\
        --config_path configs/data.yaml \\
        --context 32
"""

import os
import json
import argparse
import numpy as np
import zarr
import yaml


# --- Reverse complement lookup ---

def build_rc_lookup(token_map, rc_map):
    """Build a numpy array where rc_lookup[token] = complement token."""
    max_tok = max(token_map.values())
    lookup = np.arange(max_tok + 1, dtype=np.uint8)
    for base, idx in token_map.items():
        comp = rc_map.get(base, base)
        if comp in token_map:
            lookup[idx] = token_map[comp]
    return lookup


# --- Shard writer ---

class ShardWriter:
    def __init__(self, output_dir, shard_size, context, n_features):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.context = context
        self.n_features = n_features
        self.buffer = np.zeros((shard_size, context, n_features), dtype=np.float16)
        self.ptr = 0
        self.shard_idx = 0
        self.total_written = 0

    def add(self, row):
        """Add a single row of shape (context, n_features)."""
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


# --- CpG extraction (mirrors unmodified legacy) ---

def find_cpg_positions(seq, context):
    """Find positions of CG dinucleotides whose context window fits in the read.

    Yields (win_start, cg_pos) for each valid CpG.
    Mirrors legacy_dataset_script_unmodified.py lines 95-119.
    """
    pad = (context - 2) // 2
    L = len(seq)
    C_TOKEN, G_TOKEN = 1, 2  # from token_map: C=1, G=2

    for i in range(L - 1):
        if seq[i] == C_TOKEN and seq[i + 1] == G_TOKEN:
            win_start = i - pad
            win_end = i + 2 + pad
            # reverse strand window bounds (for ri/rp)
            rev_start = L - win_end
            rev_end = L - win_start
            # both windows must fit within the read
            if win_start >= 0 and win_end <= L and rev_start >= 0 and rev_end <= L:
                yield win_start, win_end, rev_start, rev_end


def extract_cpg_windows(read_data, feature_idx, context, rc_lookup):
    """Extract CpG windows from a single read.

    Returns list of (fwd_row, rev_row) tuples, each of shape (context, 4).
    Feature layout: [seq, kinetic1, kinetic2, mask=0].

    Mirrors legacy_dataset_script_unmodified.py:
      - Forward view: seq + fi + fp
      - Reverse view: RC(seq) reversed + ri[L-end:L-start] flipped + rp[L-end:L-start] flipped
    """
    seq = read_data[:, feature_idx['seq']]
    fi = read_data[:, feature_idx['fi']]
    fp = read_data[:, feature_idx['fp']]
    ri = read_data[:, feature_idx['ri']]
    rp = read_data[:, feature_idx['rp']]

    windows = []
    for win_start, win_end, rev_start, rev_end in find_cpg_positions(seq, context):
        # --- Forward view: [seq, fi, fp, mask=0] ---
        fwd_row = np.zeros((context, 4), dtype=np.float16)
        fwd_row[:, 0] = seq[win_start:win_end]
        fwd_row[:, 1] = fi[win_start:win_end]
        fwd_row[:, 2] = fp[win_start:win_end]
        # mask stays 0.0 (real data)

        # --- Reverse view: [RC(seq) reversed, ri aligned, rp aligned, mask=0] ---
        # Sequence: reverse the window, then complement each token
        rev_seq = rc_lookup[seq[win_start:win_end].astype(np.uint8)][::-1]

        # Kinetics: ri/rp at reverse-indexed positions, then flip to align with RC'd seq
        # Legacy: ri_values[L-win_end : L-win_start] gives ri in reverse strand order
        # Then LegacyMethylDataset flips this to get forward-strand alignment
        ri_window = ri[rev_start:rev_end]  # reverse strand order
        rp_window = rp[rev_start:rev_end]
        ri_aligned = ri_window[::-1]  # flip to forward-strand order
        rp_aligned = rp_window[::-1]

        rev_row = np.zeros((context, 4), dtype=np.float16)
        rev_row[:, 0] = rev_seq
        rev_row[:, 1] = ri_aligned
        rev_row[:, 2] = rp_aligned
        # mask stays 0.0

        windows.append((fwd_row, rev_row))

    return windows


# --- Main pipeline ---

def zarr_to_methyl_memmap_v2(
    zarr_path, output_dir, config,
    context=32, shard_size=4194304, max_shards=0,
    val_pct=0.2, seed=42
):
    """Convert Zarr to CpG methylation shards.

    Args:
        zarr_path: Path to Zarr store (from bam_to_zarr.py)
        output_dir: Output directory (will contain train/ and val/ subdirs)
        config: Parsed data.yaml config dict
        context: Window size around each CpG
        shard_size: Max rows per shard file
        max_shards: Stop after this many train shards (0 = no limit)
        val_pct: Fraction of reads for validation
        seed: Random seed for train/val split
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load Zarr
    root = zarr.open(zarr_path, mode='r')
    z_data = root['data']
    indptr = root['indptr'][:]
    all_feats = root.attrs['features']

    # Feature index map
    feature_idx = {f: all_feats.index(f) for f in ['seq', 'fi', 'fp', 'ri', 'rp']}

    # RC lookup
    token_map = config['data']['token_map']
    rc_map = config['data']['rc_map']
    rc_lookup = build_rc_lookup(token_map, rc_map)

    # Schema
    output_feats = ['seq', 'kin1', 'kin2', 'mask']
    schema = {
        "output_shape": ["N", context, len(output_feats)],
        "features": output_feats,
        "fwd_kinetics": ['fi', 'fp'],
        "rev_kinetics": ['ri', 'rp'],
        "context": context,
        "dtype": "float16",
        "source": "zarr_to_methyl_memmap_v2",
    }
    for split_dir in (train_dir, val_dir):
        with open(os.path.join(split_dir, "schema.json"), "w") as f:
            json.dump(schema, f, indent=4)

    # Train/val split at read level
    total_reads = len(indptr) - 1
    rng = np.random.RandomState(seed)
    is_val = np.zeros(total_reads, dtype=bool)
    val_indices = rng.choice(total_reads, int(total_reads * val_pct), replace=False)
    is_val[val_indices] = True

    # Writers
    train_writer = ShardWriter(train_dir, shard_size, context, len(output_feats))
    val_writer = ShardWriter(val_dir, shard_size, context, len(output_feats))

    # Process reads in batches for Zarr read efficiency
    batch_size = 500

    for batch_start in range(0, total_reads, batch_size):
        if max_shards and train_writer.shard_idx >= max_shards:
            break

        batch_end = min(batch_start + batch_size, total_reads)
        chunk_start = indptr[batch_start]
        chunk_end = indptr[batch_end]

        if chunk_start == chunk_end:
            continue

        chunk = z_data[chunk_start:chunk_end, :]

        for r in range(batch_start, batch_end):
            read_start = indptr[r] - chunk_start
            read_end = indptr[r + 1] - chunk_start
            read_data = chunk[read_start:read_end, :]

            writer = val_writer if is_val[r] else train_writer
            windows = extract_cpg_windows(read_data, feature_idx, context, rc_lookup)

            for fwd_row, rev_row in windows:
                writer.add(fwd_row)
                writer.add(rev_row)

    train_count = train_writer.finalize()
    val_count = val_writer.finalize()
    print(f"Done. Train: {train_count} windows, Val: {val_count} windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CpG memmap creation v2")
    parser.add_argument("--input_path", required=True, help="Zarr store path")
    parser.add_argument("--output_path", required=True, help="Output memmap directory")
    parser.add_argument("--config_path", required=True, help="data.yaml config path")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--shard_size", type=int, default=4194304)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--val_pct", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    zarr_to_methyl_memmap_v2(
        zarr_path=args.input_path,
        output_dir=args.output_path,
        config=config,
        context=args.context,
        shard_size=args.shard_size,
        max_shards=args.max_shards,
        val_pct=args.val_pct,
        seed=args.seed,
    )
