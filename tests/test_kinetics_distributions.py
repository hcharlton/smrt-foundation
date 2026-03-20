"""
Compare fi vs ri (and fp vs rp) distributions at CpG sites.

If these distributions differ significantly, mixing them in the same
model input columns (as the new pipeline does) forces the kin_embed
layer to handle two different distributions — which could explain
worse supervised performance.

Run standalone on HPC:
    python tests/test_kinetics_distributions.py

Or with pytest:
    python -m pytest tests/test_kinetics_distributions.py -v -s
"""

import os
import sys
import numpy as np

LABELED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', '00_raw', 'labeled')
POS_BAM_CANDIDATES = ['methylated_subset.bam', 'methylated_hifi_reads.bam']
NEG_BAM_CANDIDATES = ['unmethylated_subset.bam', 'unmethylated_hifi_reads.bam']
CONTEXT = 32


def find_bam(candidates, base_dir=LABELED_DIR):
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def extract_cpg_kinetics(bam_path, context=CONTEXT, max_reads=0):
    """Extract fi, fp, ri, rp values at CpG center positions from a BAM."""
    import pysam

    pad = (context - 2) // 2
    results = {'fi': [], 'fp': [], 'ri': [], 'rp': []}

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for i, read in enumerate(bam):
            if max_reads and i >= max_reads:
                break
            if not all(read.has_tag(t) for t in ['fi', 'fp', 'ri', 'rp']):
                continue

            seq = read.query_sequence.upper()
            read_len = len(seq)

            tags = {}
            for t in ['fi', 'fp', 'ri', 'rp']:
                arr = np.array(read.get_tag(t), dtype=np.uint8)
                if len(arr) != read_len:
                    break
                tags[t] = arr
            else:
                for j in range(read_len - 1):
                    if seq[j] == 'C' and seq[j + 1] == 'G':
                        start = j - pad
                        if start >= 0 and start + context <= read_len:
                            center = j
                            for t in ['fi', 'fp', 'ri', 'rp']:
                                results[t].append(tags[t][center])

    return {k: np.array(v, dtype=np.float64) for k, v in results.items()}


def compare_distributions(name_a, vals_a, name_b, vals_b):
    """Compare two distributions with basic stats and KS test."""
    from scipy.stats import ks_2samp

    stat, pval = ks_2samp(vals_a, vals_b)

    print(f"\n  {name_a}: mean={vals_a.mean():.2f}  std={vals_a.std():.2f}  "
          f"median={np.median(vals_a):.1f}  n={len(vals_a)}")
    print(f"  {name_b}: mean={vals_b.mean():.2f}  std={vals_b.std():.2f}  "
          f"median={np.median(vals_b):.1f}  n={len(vals_b)}")
    print(f"  KS statistic: {stat:.4f}  p-value: {pval:.2e}")

    if pval < 0.01:
        print(f"  DIFFERENT distributions (p < 0.01)")
    else:
        print(f"  Similar distributions (p >= 0.01)")

    return stat, pval


def run_comparison(bam_path, label, max_reads=0):
    print(f"\n{'='*60}")
    print(f"BAM: {os.path.basename(bam_path)} ({label})")
    print(f"{'='*60}")

    kin = extract_cpg_kinetics(bam_path, max_reads=max_reads)
    n = len(kin['fi'])
    print(f"CpG sites found: {n}")

    if n == 0:
        print("No CpG sites found, skipping")
        return

    print(f"\n--- IPD: fi (forward) vs ri (reverse) ---")
    compare_distributions('fi', kin['fi'], 'ri', kin['ri'])

    print(f"\n--- Pulse width: fp (forward) vs rp (reverse) ---")
    compare_distributions('fp', kin['fp'], 'rp', kin['rp'])

    # Also compare after log1p (since that's what the model sees)
    print(f"\n--- After log1p: fi vs ri ---")
    compare_distributions('log1p(fi)', np.log1p(kin['fi']), 'log1p(ri)', np.log1p(kin['ri']))

    print(f"\n--- After log1p: fp vs rp ---")
    compare_distributions('log1p(fp)', np.log1p(kin['fp']), 'log1p(rp)', np.log1p(kin['rp']))


def test_kinetics_distributions():
    """Pytest-compatible test — reports distributions, always passes."""
    pos_bam = find_bam(POS_BAM_CANDIDATES)
    neg_bam = find_bam(NEG_BAM_CANDIDATES)

    if pos_bam is None and neg_bam is None:
        import pytest
        pytest.skip(f"No BAM files found in {LABELED_DIR}")

    if pos_bam:
        run_comparison(pos_bam, "methylated")
    if neg_bam:
        run_comparison(neg_bam, "unmethylated")


if __name__ == '__main__':
    max_reads = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    pos_bam = find_bam(POS_BAM_CANDIDATES)
    neg_bam = find_bam(NEG_BAM_CANDIDATES)

    if pos_bam is None and neg_bam is None:
        print(f"No BAM files found in {LABELED_DIR}")
        print(f"Tried: {POS_BAM_CANDIDATES + NEG_BAM_CANDIDATES}")
        sys.exit(1)

    if pos_bam:
        run_comparison(pos_bam, "methylated", max_reads=max_reads)
    if neg_bam:
        run_comparison(neg_bam, "unmethylated", max_reads=max_reads)
