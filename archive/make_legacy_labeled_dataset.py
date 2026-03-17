"""
Legacy labeled dataset creator for CpG methylation classification.

Reads a BAM file, finds CpG sites, extracts context-sized windows of
sequence + kinetics around each site (both strands), and writes the
result as a Parquet file consumable by LegacyMethylDataset.

Output schema:
    seq       : string of length `context` (nucleotide letters)
    fi, fp    : list[uint8] of length `context` (forward kinetics)
    ri, rp    : list[uint8] of length `context` (reverse kinetics)
    label     : int (1 = methylated, 0 = unmethylated)
    read_name : str
    cg_pos    : int (0-based position of 'C' in the CpG within the read)
    strand    : str ('fwd' or 'rev')

Key differences from the new zarr_to_methyl_memmap pipeline:
  - Normalization is deferred to the LegacyMethylDataset at load time
    (global log-Z normalization), NOT applied here.
  - Reverse strand always gets reverse-complemented sequence.
  - Both strands' kinetics are stored in their own columns (fi/fp vs ri/rp)
    rather than being mixed into the same feature positions.
"""

import argparse
import os
import numpy as np
import pysam
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


RC_MAP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def _reverse_complement(seq_str):
    return "".join(RC_MAP.get(b, 'N') for b in reversed(seq_str))


def _find_cpg_windows(seq_str, context):
    """Yield (start, cg_pos) for every CpG whose context-window fits in the read."""
    pad = (context - 2) // 2
    for i in range(len(seq_str) - 1):
        if seq_str[i] == 'C' and seq_str[i + 1] == 'G':
            start = i - pad
            if start >= 0 and start + context <= len(seq_str):
                yield start, i


def bam_to_legacy_parquet(
    bam_path, output_path, context=32, label=1, n_reads=0,
    row_group_size=50_000
):
    """Convert a BAM to the legacy parquet format.

    Parameters
    ----------
    bam_path : str
        Input BAM with kinetics tags (fi, fp, ri, rp).
    output_path : str
        Output parquet path.
    context : int
        Window size around each CpG site.
    label : int
        Label to assign (1 = methylated, 0 = unmethylated).
    n_reads : int
        Max reads to process (0 = all).
    row_group_size : int
        Parquet row group size.
    """
    required_tags = {'fi', 'fp', 'ri', 'rp'}
    kin_tags = ['fi', 'fp', 'ri', 'rp']

    records = {
        'seq': [], 'fi': [], 'fp': [], 'ri': [], 'rp': [],
        'label': [], 'read_name': [], 'cg_pos': [], 'strand': [],
    }

    writer = None
    schema = None
    n_buffered = 0

    def _flush():
        nonlocal writer, schema, n_buffered
        if n_buffered == 0:
            return
        table = pa.table({
            'seq': pa.array(records['seq'], type=pa.string()),
            'fi': pa.array(records['fi'], type=pa.list_(pa.uint8())),
            'fp': pa.array(records['fp'], type=pa.list_(pa.uint8())),
            'ri': pa.array(records['ri'], type=pa.list_(pa.uint8())),
            'rp': pa.array(records['rp'], type=pa.list_(pa.uint8())),
            'label': pa.array(records['label'], type=pa.int32()),
            'read_name': pa.array(records['read_name'], type=pa.string()),
            'cg_pos': pa.array(records['cg_pos'], type=pa.int32()),
            'strand': pa.array(records['strand'], type=pa.string()),
        })
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        for k in records:
            records[k] = []
        n_buffered = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if n_reads and i >= n_reads:
                break

            if not all(read.has_tag(t) for t in required_tags):
                continue

            seq_str = read.query_sequence.upper()
            read_name = read.query_name
            read_len = len(seq_str)

            tag_data = {}
            for t in kin_tags:
                arr = np.array(read.get_tag(t), dtype=np.uint8)
                if len(arr) != read_len:
                    break
                tag_data[t] = arr
            else:
                # Forward strand windows
                for start, cg_pos in _find_cpg_windows(seq_str, context):
                    end = start + context
                    records['seq'].append(seq_str[start:end])
                    for t in kin_tags:
                        records[t].append(tag_data[t][start:end].tolist())
                    records['label'].append(label)
                    records['read_name'].append(read_name)
                    records['cg_pos'].append(cg_pos)
                    records['strand'].append('fwd')
                    n_buffered += 1

                # Reverse strand windows
                rc_seq = _reverse_complement(seq_str)
                rev_tags = {t: np.flip(tag_data[t]).copy() for t in kin_tags}
                for start, cg_pos in _find_cpg_windows(rc_seq, context):
                    end = start + context
                    records['seq'].append(rc_seq[start:end])
                    for t in kin_tags:
                        records[t].append(rev_tags[t][start:end].tolist())
                    records['label'].append(label)
                    records['read_name'].append(read_name)
                    records['cg_pos'].append(cg_pos)
                    records['strand'].append('rev')
                    n_buffered += 1

                if n_buffered >= row_group_size:
                    _flush()

    _flush()
    if writer is not None:
        writer.close()
        print(f"Wrote {output_path}")
    else:
        print("No CpG windows found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create legacy labeled parquet dataset from a BAM file."
    )
    parser.add_argument("--input_path", required=True, help="Input BAM file")
    parser.add_argument("--output_path", required=True, help="Output parquet path")
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--label", type=int, default=1, help="1=methylated, 0=unmethylated")
    parser.add_argument("--n_reads", type=int, default=0, help="0=all reads")

    args = parser.parse_args()
    bam_to_legacy_parquet(
        bam_path=args.input_path,
        output_path=args.output_path,
        context=args.context,
        label=args.label,
        n_reads=args.n_reads,
    )
