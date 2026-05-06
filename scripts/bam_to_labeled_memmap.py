"""
Single-pass BAM -> labeled memmap for tissue-provenance classification.

Reads a PacBio HiFi BAM and a per-read tissue label file, samples reads per
tissue, walks the BAM once, and writes:

  output_dir/
    schema.json           - feature names, tissue/cell ID maps, params
    manifest.parquet      - one row per output window: (shard_idx, row_idx,
                            read_name, tissue_str, cell_str, tissue_id,
                            cell_id, crop_start, read_length). Single source
                            of truth for labels and split logic.
    shard_NNNNN.npy       - (<=shard_size, context, n_features) uint8

Output is uint8 raw BAM values (no normalization). Normalization is the
dataloader's job.

Each output row holds a single random crop of length `context` from one read,
with feature channels:
  [seq, fi, fp, ri, rp, *optional_tags, mask]

Both forward and reverse kinetics are stored, with reverse-strand kinetics
indexed at the matching forward-strand position (i.e. ri[L-1-j] sits at output
column position j after cropping). This sidesteps the v1 reverse-kinetics
misalignment bug. The mask channel is always 0 because reads shorter than
`context` are dropped.

Label assignment happens in the same iteration of the same loop that writes
the kinetics, so misattribution is structurally impossible.

Usage:
    python -m scripts.bam_to_labeled_memmap \\
        --bam_path data/00_raw/unlabeled/yoran_kinetics_diploid.bam \\
        --label_path data/01_processed/ssl_sets/yoran_read_labels.txt \\
        --output_dir data/01_processed/tissue_sets/yoran_ctx4096 \\
        --config configs/data.yaml \\
        --tissues kidney liver lung muscle skin spleen testis colon \\
        --context 4096 --max_reads_per_tissue 200000 --seed 42
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import polars as pl
import pysam

from smrt_foundation.utils import parse_yaml


# ---------------------------------------------------------------------------
# Filters / decoders (copied verbatim from scripts/bam_to_zarr.py:38-100, 112-125
# so any future divergence between the two scripts is intentional, not silent).
# ---------------------------------------------------------------------------

def _process_read(read, tags):
    """
    Processes a single pysam.AlignmentRead.
    Checks for required tags and extracts all full-length tag data.
    Returns None if the read is missing tags or feature lengths disagree.
    """
    if not all(read.has_tag(tag) for tag in set(tags) - {'seq', 'qual'}):
        return None
    read_data = {}
    for tag in set(tags) - {'seq', 'qual'}:
        tag_data = read.get_tag(tag) if read.has_tag(tag) else None
        read_data[tag] = tag_data
    read_data |= {
        'seq': read.query_sequence,
        'qual': np.frombuffer(read.qual.encode('ascii'), dtype=np.uint8) - 33
    }
    if any(read_data[tag] is None or len(read_data[tag]) == 0 for tag in tags):
        return None
    if len(set([len(v) for k, v in read_data.items()])) != 1:
        return None
    return {
        "name": read.query_name,
        "data": read_data,
        "seq_len": read.query_length,
    }


def _check_tags(bam_path, tags, n_reads=20, threshold=0.8):
    """
    Checks the first n_reads. Raises ValueError if any tag is missing in
    >threshold fraction of reads.
    """
    if not tags:
        return
    tag_counts = {t: 0 for t in tags}
    reads_checked = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if i >= n_reads:
                break
            reads_checked += 1
            for t in tags:
                if read.has_tag(t):
                    tag_counts[t] += 1
    if reads_checked == 0:
        raise ValueError("BAM file appears empty.")
    for t, count in tag_counts.items():
        missing_rate = 1.0 - (count / reads_checked)
        if missing_rate > threshold:
            raise ValueError(
                f"Tag '{t}' is missing in {missing_rate:.1%} of the first "
                f"{reads_checked} reads. Threshold is {threshold:.0%}. Aborting."
            )
    print(f"Validation successful: All requested tags present in >{1-threshold:.0%} of checked reads.")


def _build_seq_lookup(token_map):
    """Map ASCII codes -> integer token ids (uint8)."""
    lookup = np.zeros(128, dtype=np.uint8)
    for base, val in token_map.items():
        if len(base) == 1:
            lookup[ord(base)] = val
    return lookup


# ---------------------------------------------------------------------------
# Labels parser (also called directly from tests for the ZMW disambiguation
# regression test).
# ---------------------------------------------------------------------------

def parse_labels(label_path, tissues_keep):
    """
    Parse a `<read_name> <tissue>` label file.

    Returns
    -------
    label_map : dict[str, str]
        Maps full read_name (e.g. ``cell/zmw/ccs``) to its tissue string,
        filtered to ``tissues_keep``. Uses the FULL read_name string as the
        key; never splits out the ZMW integer (which is only unique within a
        cell, not globally).
    cell_to_id : dict[str, int]
        Sorted alphabetical assignment of cell-prefix string to integer id.
        Includes every cell prefix that appears in any kept-tissue row.
    """
    keep = set(tissues_keep)
    label_map = {}
    cells_seen = set()
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            read_name, tissue = parts[0], parts[1]
            if tissue not in keep:
                continue
            label_map[read_name] = tissue
            cell = read_name.split('/', 1)[0]
            cells_seen.add(cell)
    cell_to_id = {cell: i for i, cell in enumerate(sorted(cells_seen))}
    return label_map, cell_to_id


# ---------------------------------------------------------------------------
# Sharded writers
# ---------------------------------------------------------------------------

class ShardWriter:
    """Buffers (context, n_features) uint8 rows; flushes shard_NNNNN.npy."""

    def __init__(self, output_dir, shard_size, context, n_features):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.context = context
        self.n_features = n_features
        self.buffer = np.zeros((shard_size, context, n_features), dtype=np.uint8)
        self.ptr = 0
        self.shard_idx = 0
        self.total_written = 0

    def add(self, row):
        """Add one (context, n_features) row. Returns (shard_idx, row_idx) of the slot it landed in."""
        shard_idx = self.shard_idx
        row_idx = self.ptr
        self.buffer[self.ptr] = row
        self.ptr += 1
        if self.ptr >= self.shard_size:
            self._flush()
        return shard_idx, row_idx

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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Fixed feature schema. Optional tags from --optional_tags get inserted
# between `rp` and `mask`.
FIXED_FEATURES_PRE_OPTIONAL = ['seq', 'fi', 'fp', 'ri', 'rp']
MASK_FEATURE = 'mask'


def _build_read_array(read_data, lookup_table, features):
    """
    Build the (L, n_features) uint8 array for a single read.

    `seq, fi, fp` are taken in forward order. `ri, rp` are reversed
    (`[::-1].copy()`) so that column j corresponds to forward-strand position j.
    Optional tags are taken in forward order. Mask column stays all-zero (set
    by the caller's pre-zeroed buffer or here).
    """
    L = len(read_data['seq'])
    n_feat = len(features)
    out = np.zeros((L, n_feat), dtype=np.uint8)
    for i, feat in enumerate(features):
        if feat == 'seq':
            seq_bytes = np.frombuffer(read_data['seq'].upper().encode('ascii'), dtype=np.uint8)
            out[:, i] = lookup_table[seq_bytes]
        elif feat == 'mask':
            # Already zero from np.zeros initialization
            pass
        elif feat in ('ri', 'rp'):
            # Reverse-strand kinetics: reverse the per-base array so column j
            # corresponds to forward-strand position j.
            arr = np.array(read_data[feat], dtype=np.uint8)
            out[:, i] = arr[::-1].copy()
        else:
            # Forward-order tag (fi, fp, sm, sx, ...)
            out[:, i] = np.array(read_data[feat], dtype=np.uint8)
    return out


def bam_to_labeled_memmap(
    bam_path,
    label_path,
    output_dir,
    config,
    tissues,
    context,
    max_reads_per_tissue=0,
    optional_tags=(),
    seed=42,
    shard_size=16384,
):
    os.makedirs(output_dir, exist_ok=True)

    # ----- 1. Parse labels file (filter to requested tissues) -----
    label_map, cell_to_id = parse_labels(label_path, tissues)

    # ----- 2. ID maps -----
    tissue_to_id = {t: i for i, t in enumerate(sorted(tissues))}

    # ----- Feature layout -----
    # Validate optional tags don't collide with fixed names
    for t in optional_tags:
        if t in FIXED_FEATURES_PRE_OPTIONAL or t == MASK_FEATURE:
            raise ValueError(
                f"--optional_tags {t!r} collides with a fixed feature name. "
                f"Fixed: {FIXED_FEATURES_PRE_OPTIONAL + [MASK_FEATURE]}"
            )
    features = list(FIXED_FEATURES_PRE_OPTIONAL) + list(optional_tags) + [MASK_FEATURE]
    n_features = len(features)

    # Tags we need to extract from the BAM (everything except seq and mask)
    bam_tags = [f for f in features if f not in ('seq', 'mask')]

    # ----- 3. Pre-sample reads per tissue -----
    rng_sample = np.random.RandomState(seed)
    rng_crop = np.random.RandomState(seed + 1)

    by_tissue = defaultdict(list)
    for read_name, tissue in label_map.items():
        by_tissue[tissue].append(read_name)

    accept_set = set()
    sample_counts = {}  # for the final report
    for tissue in sorted(tissues):
        candidates = np.array(by_tissue.get(tissue, []), dtype=object)
        if len(candidates) == 0:
            sample_counts[tissue] = 0
            continue
        if max_reads_per_tissue and len(candidates) > max_reads_per_tissue:
            n = max_reads_per_tissue
            sampled = rng_sample.choice(candidates, size=n, replace=False)
        else:
            n = len(candidates)
            sampled = candidates
        accept_set.update(sampled.tolist())
        sample_counts[tissue] = int(n)

    # ----- 4. Tag check -----
    _check_tags(bam_path=bam_path, tags=set(bam_tags))

    # ----- 5. Walk BAM once -----
    seq_map = config['data']['token_map']
    lookup_table = _build_seq_lookup(seq_map)

    shard_writer = ShardWriter(output_dir, shard_size, context, n_features)

    manifest_rows = []
    counters = {
        'bam_seen': 0,
        'in_accept': 0,
        'filter_failed': 0,
        'too_short': 0,
        'written': 0,
    }
    per_tissue_written = defaultdict(int)
    seen_accepted = set()

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False, threads=5) as bam:
        for read in bam:
            counters['bam_seen'] += 1
            if read.query_name not in accept_set:
                continue
            counters['in_accept'] += 1
            seen_accepted.add(read.query_name)

            read_dict = _process_read(read, tags=bam_tags + ['seq'])
            if read_dict is None:
                counters['filter_failed'] += 1
                # Still count as "seen" for early-termination — we won't try again
                if len(seen_accepted) == len(accept_set):
                    break
                continue

            L = read_dict['seq_len']
            if L < context:
                counters['too_short'] += 1
                if len(seen_accepted) == len(accept_set):
                    break
                continue

            tissue = label_map[read.query_name]
            tissue_id = tissue_to_id[tissue]
            cell = read.query_name.split('/', 1)[0]
            cell_id = cell_to_id[cell]

            # Random crop start (advances rng_crop once per accepted+passing read)
            crop_start = int(rng_crop.randint(0, L - context + 1))

            # Build the (L, n_features) uint8 array, then slice
            full = _build_read_array(read_dict['data'], lookup_table, features)
            window = full[crop_start:crop_start + context]

            shard_idx, row_idx = shard_writer.add(window)

            manifest_rows.append({
                'shard_idx': shard_idx,
                'row_idx': row_idx,
                'read_name': read.query_name,
                'tissue_str': tissue,
                'cell_str': cell,
                'tissue_id': tissue_id,
                'cell_id': cell_id,
                'crop_start': crop_start,
                'read_length': L,
            })
            counters['written'] += 1
            per_tissue_written[tissue] += 1

            if len(seen_accepted) == len(accept_set):
                break

    # ----- 6. Finalize -----
    shard_writer.finalize()

    # Manifest (polars)
    manifest_schema = {
        'shard_idx': pl.UInt32,
        'row_idx': pl.UInt32,
        'read_name': pl.String,
        'tissue_str': pl.String,
        'cell_str': pl.String,
        'tissue_id': pl.Int32,
        'cell_id': pl.Int32,
        'crop_start': pl.UInt32,
        'read_length': pl.UInt32,
    }
    manifest_df = pl.DataFrame(manifest_rows, schema=manifest_schema)
    manifest_df.write_parquet(os.path.join(output_dir, 'manifest.parquet'))

    # Schema
    schema = {
        'output_shape': ['N', context, n_features],
        'features': features,
        'feature_notes': (
            'ri/rp at row j are reverse-strand kinetics measured at forward-strand '
            'position j (BAM\'s ri[L-1-j], rp[L-1-j]). seq, fi, fp, optional tags '
            'are in forward order. mask is always 0 because reads shorter than '
            'context are dropped.'
        ),
        'context': context,
        'dtype': 'uint8',
        'normalize': 'none',
        'normalize_note': 'Raw BAM values are stored on disk; normalization is the dataloader\'s job.',
        'tissue_to_id': tissue_to_id,
        'cell_to_id': cell_to_id,
        'individual': os.path.basename(label_path).replace('_read_labels.txt', ''),
        'bam_path': bam_path,
        'label_path': label_path,
        'seed': seed,
        'max_reads_per_tissue': max_reads_per_tissue,
        'shard_size': shard_size,
        'source': 'bam_to_labeled_memmap',
    }
    with open(os.path.join(output_dir, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=2)

    # Report
    print('--- bam_to_labeled_memmap counters ---')
    print(f"BAM reads seen           : {counters['bam_seen']}")
    print(f"In accept_set            : {counters['in_accept']}")
    print(f"  filter_failed          : {counters['filter_failed']}")
    print(f"  too_short (< {context}) : {counters['too_short']}")
    print(f"  written                : {counters['written']}")
    print(f"Per-tissue sampled / written:")
    for t in sorted(tissues):
        print(f"  {t:<10s}: sampled={sample_counts.get(t, 0):>8d}  written={per_tissue_written.get(t, 0):>8d}")
    print(f"Shards written: {shard_writer.shard_idx}")
    print(f"Total accepted reads encountered: {len(seen_accepted)} / {len(accept_set)}")
    print('--------------------------------------')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Single-pass BAM -> labeled memmap for tissue-provenance classification.'
    )
    parser.add_argument('--bam_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True,
                        help='Path to <read_name> <tissue> label file.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configs/data.yaml.')
    parser.add_argument('--tissues', nargs='+', required=True,
                        help='Space-separated list of tissues to keep.')
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--max_reads_per_tissue', type=int, default=0,
                        help='0 means no cap.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard_size', type=int, default=16384)
    parser.add_argument('--optional_tags', nargs='*', default=[],
                        help='Extra per-base tags to include (e.g. sm sx).')

    args = parser.parse_args()

    if args.context <= 0:
        print('Error: --context must be positive', file=sys.stderr)
        sys.exit(1)
    if args.shard_size <= 0:
        print('Error: --shard_size must be positive', file=sys.stderr)
        sys.exit(1)
    if args.max_reads_per_tissue < 0:
        print('Error: --max_reads_per_tissue must be >= 0 (0 = no cap)', file=sys.stderr)
        sys.exit(1)

    config = parse_yaml(args.config)

    bam_to_labeled_memmap(
        bam_path=os.path.expanduser(args.bam_path),
        label_path=os.path.expanduser(args.label_path),
        output_dir=os.path.expanduser(args.output_dir),
        config=config,
        tissues=args.tissues,
        context=args.context,
        max_reads_per_tissue=args.max_reads_per_tissue,
        optional_tags=args.optional_tags,
        seed=args.seed,
        shard_size=args.shard_size,
    )


if __name__ == '__main__':
    main()
