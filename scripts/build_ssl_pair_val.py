"""Build a permanent SSL pair validation set with known gap distances.

The output is the SSL analogue of the CpG val sets at
`data/01_processed/val_sets/cpg_pos_v2.memmap/` etc. — sharded `.npy`
files containing pre-extracted positive pairs at known separations,
suitable for evaluating any contrastive SSL encoder.

What gets stored:
  <output_dir>/
    pairs/
      shard_00000.npy    # shape (shard_size, 2, target_len, n_features)
      shard_00001.npy
      ...
    gaps.npy              # shape (N_total,) int32 — gap_bp per pair
    read_ids.npy          # shape (N_total,) int64 — global read idx in
                          #   the source memmap each pair was drawn from
    anchors.npy           # shape (N_total,) int32 — view1 start position
                          #   within that source read
    metadata.yaml         # gaps used, target_len, source dataset, seed, etc.

The `read_ids` and `anchors` sidecars are not needed for normal
evaluation (the eval loop uses pairs + gaps only) but they make every
pair independently re-derivable from the source memmap, which:
  - lets tests cross-check the build by re-slicing the source read at
    `[anchor : anchor+target_len]` and `[anchor+target_len+gap_bp : ...]`
    and comparing against the stored pair
  - lets future debugging trace any anomalous pair back to its source

Each pair is a positive: view1 and view2 are non-overlapping
`target_len`-base windows from the same molecule, separated by `gap_bp`
real (non-pad) bases. The encoder under evaluation produces embeddings
for both views; per-gap top-k accuracy and cosine-similarity statistics
fall out of standard NTXent-style matching.

Why this design (vs. on-the-fly sampling):
  - Reproducibility across experiments: same val set, same seed,
    same numbers. ssl_56 vs ssl_57 vs eventual masked-prediction runs
    are directly comparable rather than approximately comparable.
  - Frozen evaluation: no augmentation noise mixed with the metric.
  - Cheap to build (~5 GB at the 10M cap) and trivial to load.

Usage:
    python -m scripts.build_ssl_pair_val \\
        --source_memmap data/01_processed/ssl_sets/yoran_raw.memmap \\
        --output_path data/01_processed/val_sets/ssl_pair_val_v1.memmap \\
        --total_cap 10000000

Source memmap must follow the `ShardedMemmapDataset` layout (sharded
`.npy` files where each row is a (T_max, n_features) read with a final
pad channel). Reads with insufficient unpadded length to fit a given
gap's required span (`2*target_len + gap_bp`) are skipped for that gap;
the actual per-gap count is the lesser of the user-requested target and
the count of reads that satisfied the constraint.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional, Sequence

import numpy as np
import yaml


# Default gap sweep — 0..512 in 32-bp increments plus two far points.
# The 0-512 dense grid covers the locality regime where ssl_56 trains;
# {1024, 2048} extend into the long-range regime to confirm the
# accuracy-vs-gap curve actually degrades at distances where the encoder
# should not have learned to discriminate.
DEFAULT_GAPS_BP = (
    0, 32, 64, 96, 128, 160, 192, 224, 256, 288,
    320, 352, 384, 416, 448, 480, 512,
    1024, 2048,
)
assert len(DEFAULT_GAPS_BP) == 19


CH_PAD = 3  # last channel is the pad channel by project convention


def unpadded_length(read: np.ndarray) -> int:
    """Length of the prefix of `read` (shape (T, C)) that is not padded.

    Pad channel convention: 0.0 = real base, 1.0 = pad. Mirrors
    `smrt_foundation.augment._unpadded_length` but operates on numpy
    arrays directly so the build script doesn't need torch.
    """
    pad = read[:, CH_PAD]
    not_pad = (pad == 0.0)
    if bool(not_pad.all()):
        return int(read.shape[0])
    # First padded position from the left.
    first_pad = int(np.argmax(pad == 1.0))
    return first_pad


class PairShardWriter:
    """Sharded writer for (2, T, C) positive pairs.

    Mirrors the writer pattern in `scripts/zarr_to_methyl_memmap_v2.py`
    but the per-row shape is (2, T, C) instead of (T, C). Buffer is
    allocated upfront and reused across shards.
    """

    def __init__(self, output_dir: str, shard_size: int, target_len: int, n_features: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.target_len = target_len
        self.n_features = n_features
        self.buffer = np.zeros((shard_size, 2, target_len, n_features), dtype=np.float16)
        self.ptr = 0
        self.shard_idx = 0
        self.total_written = 0

    def add(self, pair: np.ndarray) -> None:
        """Add a single (2, target_len, n_features) pair to the buffer."""
        assert pair.shape == (2, self.target_len, self.n_features), (
            f"expected pair shape (2, {self.target_len}, {self.n_features}), got {pair.shape}"
        )
        self.buffer[self.ptr] = pair
        self.ptr += 1
        if self.ptr >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if self.ptr == 0:
            return
        path = os.path.join(self.output_dir, f"shard_{self.shard_idx:05d}.npy")
        np.save(path, self.buffer[:self.ptr])
        self.total_written += self.ptr
        self.shard_idx += 1
        self.ptr = 0
        self.buffer[:] = 0

    def finalize(self) -> int:
        self._flush()
        return self.total_written


class _ShardedMemmapReader:
    """Minimal read-only iterator over a `ShardedMemmapDataset`-format
    directory. Avoids importing torch in the build script.

    Returns reads in shard order so iteration is sequential on disk.
    """

    def __init__(self, source_dir: str):
        self.source_dir = os.path.expandvars(source_dir)
        self.shard_paths = sorted(glob.glob(os.path.join(self.source_dir, "*.npy")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {self.source_dir}")

    def iter_reads(self):
        """Yield (read_idx, read_array) tuples in shard order. Each
        `read_array` is a numpy view, shape (T, n_features)."""
        global_idx = 0
        for path in self.shard_paths:
            shard = np.load(path, mmap_mode='r')
            # shard shape: (n_reads_in_shard, T, n_features)
            for local_idx in range(shard.shape[0]):
                yield global_idx, np.asarray(shard[local_idx])
                global_idx += 1

    def n_features(self) -> int:
        first = np.load(self.shard_paths[0], mmap_mode='r')
        return int(first.shape[-1])

    def context(self) -> int:
        first = np.load(self.shard_paths[0], mmap_mode='r')
        return int(first.shape[1])


def build_ssl_pair_val(
    source_memmap: str,
    output_dir: str,
    gaps_bp: Sequence[int] = DEFAULT_GAPS_BP,
    target_len: int = 32,
    total_cap: int = 10_000_000,
    shard_size: int = 100_000,
    seed: int = 42,
    max_reads_to_scan: Optional[int] = None,
) -> dict:
    """Build the SSL pair val set.

    Args:
        source_memmap: Path to a directory of `.npy` shards in the
            `ShardedMemmapDataset` format. Reads are drawn from this
            source uniformly at random (with a fixed seed for
            reproducibility).
        output_dir: Output directory. Will be created if missing.
        gaps_bp: Gap distances to sample, in real (non-pad) base pairs.
        target_len: Window length for each view. Both view1 and view2
            are this many bases wide.
        total_cap: Maximum total pairs across all gaps (default 10M).
            Per-gap target = total_cap // len(gaps_bp).
        shard_size: Pairs per output shard file.
        seed: RNG seed for reproducibility.
        max_reads_to_scan: If set, stop after scanning this many reads
            (for fast tests). 0 / None = no limit.

    Returns:
        Dict with summary statistics: per-gap counts, total written,
        skipped count, etc. Also written to `metadata.yaml` in output_dir.
    """
    pairs_dir = os.path.join(output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    reader = _ShardedMemmapReader(source_memmap)
    n_features = reader.n_features()
    src_context = reader.context()

    gaps_bp = tuple(int(g) for g in gaps_bp)
    n_gaps = len(gaps_bp)
    pairs_per_gap = total_cap // n_gaps

    # Validate: largest gap must fit in source context.
    max_gap = max(gaps_bp)
    max_required = 2 * target_len + max_gap
    if max_required > src_context:
        raise ValueError(
            f"max gap_bp={max_gap} requires span {max_required} > source "
            f"context {src_context}; either lower max gap or use a source "
            f"with longer reads."
        )

    rng = np.random.default_rng(seed)

    # Per-gap counters and a list of (gap_index, anchor_seed) work items.
    # Strategy: each read scan tries to fill *all* gaps that still need
    # pairs and that fit in the read's unpadded length, choosing one
    # anchor per (read, gap) combination. This single-pass design avoids
    # re-iterating the disk for each gap value.
    counts = {g: 0 for g in gaps_bp}
    targets = {g: pairs_per_gap for g in gaps_bp}
    skipped_short = 0  # read couldn't satisfy any remaining gap
    reads_scanned = 0

    writer = PairShardWriter(pairs_dir, shard_size, target_len, n_features)
    gaps_list = []
    read_ids_list = []
    anchors_list = []

    print(f"[build_ssl_pair_val] source: {source_memmap}")
    print(f"[build_ssl_pair_val] target_len: {target_len}, n_features: {n_features}, src_context: {src_context}")
    print(f"[build_ssl_pair_val] gaps: {gaps_bp}")
    print(f"[build_ssl_pair_val] pairs_per_gap: {pairs_per_gap} (total_cap={total_cap})")
    print(f"[build_ssl_pair_val] shard_size: {shard_size}, seed: {seed}")

    for read_idx, read in reader.iter_reads():
        reads_scanned += 1
        if max_reads_to_scan and reads_scanned > max_reads_to_scan:
            break
        # Stop once every gap has met its target.
        if all(counts[g] >= targets[g] for g in gaps_bp):
            break

        unp = unpadded_length(read)

        any_used = False
        for g in gaps_bp:
            if counts[g] >= targets[g]:
                continue
            required = 2 * target_len + g
            if unp < required:
                continue
            max_start = unp - required
            anchor = int(rng.integers(0, max_start + 1))
            v1 = read[anchor:anchor + target_len].astype(np.float16)
            v2 = read[anchor + target_len + g:anchor + 2 * target_len + g].astype(np.float16)
            pair = np.stack([v1, v2], axis=0)
            writer.add(pair)
            gaps_list.append(g)
            read_ids_list.append(read_idx)
            anchors_list.append(anchor)
            counts[g] += 1
            any_used = True

        if not any_used:
            skipped_short += 1

        if reads_scanned % 50_000 == 0:
            done = sum(counts.values())
            print(f"  scanned {reads_scanned} reads, {done} pairs written, "
                  f"shard {writer.shard_idx}, smallest count: "
                  f"{min(counts.values())} / {pairs_per_gap}")

    total = writer.finalize()

    # Write sidecars (gap, source-read id, anchor offset).
    gaps_arr = np.asarray(gaps_list, dtype=np.int32)
    read_ids_arr = np.asarray(read_ids_list, dtype=np.int64)
    anchors_arr = np.asarray(anchors_list, dtype=np.int32)
    assert len(gaps_arr) == total, (
        f"gaps sidecar length ({len(gaps_arr)}) != pairs written ({total})"
    )
    assert len(read_ids_arr) == total
    assert len(anchors_arr) == total
    np.save(os.path.join(output_dir, "gaps.npy"), gaps_arr)
    np.save(os.path.join(output_dir, "read_ids.npy"), read_ids_arr)
    np.save(os.path.join(output_dir, "anchors.npy"), anchors_arr)

    # Write metadata.
    metadata = {
        "schema_version": 1,
        "source_memmap": os.path.abspath(os.path.expandvars(source_memmap)),
        "gaps_bp": list(gaps_bp),
        "target_len": int(target_len),
        "n_features": int(n_features),
        "src_context": int(src_context),
        "pairs_per_gap_target": int(pairs_per_gap),
        "total_cap": int(total_cap),
        "shard_size": int(shard_size),
        "seed": int(seed),
        "total_written": int(total),
        "per_gap_counts": {int(g): int(c) for g, c in counts.items()},
        "reads_scanned": int(reads_scanned),
        "skipped_short": int(skipped_short),
    }
    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    # Quick post-build sanity log.
    print(f"[build_ssl_pair_val] done. total pairs: {total}")
    print(f"[build_ssl_pair_val] per-gap counts:")
    for g in gaps_bp:
        bar = "#" * int(40 * counts[g] / max(targets[g], 1))
        print(f"  gap={g:5d}: {counts[g]:>10d} / {pairs_per_gap}  {bar}")
    print(f"[build_ssl_pair_val] reads scanned: {reads_scanned} (skipped {skipped_short} as too short)")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SSL pair validation set with known gap distances")
    parser.add_argument("--source_memmap", required=True,
                        help="Path to ShardedMemmapDataset-format source (e.g. yoran_raw.memmap)")
    parser.add_argument("--output_path", required=True,
                        help="Output directory (e.g. data/01_processed/val_sets/ssl_pair_val_v1.memmap)")
    parser.add_argument("--gaps", type=int, nargs="+", default=list(DEFAULT_GAPS_BP),
                        help="Gap distances in real bases between view1 and view2 edges")
    parser.add_argument("--target_len", type=int, default=32,
                        help="Window length per view (must match downstream context)")
    parser.add_argument("--total_cap", type=int, default=10_000_000,
                        help="Maximum total pairs across all gaps")
    parser.add_argument("--shard_size", type=int, default=100_000,
                        help="Pairs per output shard file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_reads_to_scan", type=int, default=0,
                        help="If > 0, stop after scanning this many reads (for fast iteration)")

    args = parser.parse_args()

    build_ssl_pair_val(
        source_memmap=args.source_memmap,
        output_dir=args.output_path,
        gaps_bp=tuple(args.gaps),
        target_len=args.target_len,
        total_cap=args.total_cap,
        shard_size=args.shard_size,
        seed=args.seed,
        max_reads_to_scan=args.max_reads_to_scan if args.max_reads_to_scan > 0 else None,
    )
