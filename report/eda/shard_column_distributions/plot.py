"""
New pipeline shard column distributions at CpG center.

Tests hypothesis: the new pipeline mixes fi and ri values in column 1
(and fp/rp in column 2). If the distribution is bimodal, the model's
kin_embed linear layer faces a harder learning problem than the legacy
pipeline where each feature type is separately normalized.

Uses new memmap shards directly.
"""

import os
import sys
import glob
import yaml
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)


def load_all_shards(directory, max_samples=50_000):
    paths = sorted(glob.glob(os.path.join(directory, "shard_*.npy")))
    if not paths:
        return np.empty((0,))
    arrays = []
    n = 0
    for p in paths:
        arr = np.load(p)
        arrays.append(arr)
        n += arr.shape[0]
        if max_samples and n >= max_samples:
            break
    out = np.concatenate(arrays, axis=0)
    return out[:max_samples] if max_samples else out


def main(output_path):
    with open('./configs/supervised.yaml', 'r') as f:
        sup_config = yaml.safe_load(f)

    pos_train = sup_config.get('pos_data_train', '')
    neg_train = sup_config.get('neg_data_train', '')

    if not os.path.isdir(os.path.expandvars(pos_train)):
        pos_train = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
        neg_train = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'

    with open('./configs/data.yaml', 'r') as f:
        context = yaml.safe_load(f)['cpg_pipeline']['context']
    center = context // 2

    pos_data = load_all_shards(pos_train)
    neg_data = load_all_shards(neg_train)

    if pos_data.ndim < 3 or neg_data.ndim < 3:
        print("ERROR: No shard data found")
        sys.exit(1)

    # Column 1 = IPD (fi or ri depending on window), Column 2 = PW (fp or rp)
    pos_ipd = pos_data[:, center, 1].astype(np.float64)
    neg_ipd = neg_data[:, center, 1].astype(np.float64)
    pos_pw = pos_data[:, center, 2].astype(np.float64)
    neg_pw = neg_data[:, center, 2].astype(np.float64)

    charts = []
    features = [
        ('Column 1: IPD (fi or ri mixed)', pos_ipd, neg_ipd),
        ('Column 2: PW (fp or rp mixed)', pos_pw, neg_pw),
    ]

    for feat_name, pos_vals, neg_vals in features:
        # Raw values
        n = min(50_000, len(pos_vals), len(neg_vals))
        raw_df = pl.DataFrame({
            'value': np.concatenate([pos_vals[:n], neg_vals[:n]]),
            'class': ['methylated'] * n + ['unmethylated'] * n,
        })

        raw_chart = alt.Chart(raw_df).mark_area(
            opacity=0.5, interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=60).title(f'{feat_name} (raw)'),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('class:N', scale=alt.Scale(
                domain=['methylated', 'unmethylated'],
                range=['#e45756', '#4c78a8']
            )),
        ).properties(width=350, height=200, title=f'{feat_name} — raw')

        # Log-transformed
        log_df = pl.DataFrame({
            'value': np.concatenate([np.log1p(pos_vals[:n]), np.log1p(neg_vals[:n])]),
            'class': ['methylated'] * n + ['unmethylated'] * n,
        })

        log_chart = alt.Chart(log_df).mark_area(
            opacity=0.5, interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=60).title(f'{feat_name} (log1p)'),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('class:N', scale=alt.Scale(
                domain=['methylated', 'unmethylated'],
                range=['#e45756', '#4c78a8']
            )),
        ).properties(width=350, height=200, title=f'{feat_name} — log1p')

        charts.append(alt.hconcat(raw_chart, log_chart))

    # Summary stats
    print("\nNew shard column statistics at CpG center:")
    for name, pos_vals, neg_vals in features:
        all_vals = np.concatenate([pos_vals, neg_vals])
        print(f"\n  {name}:")
        print(f"    Overall: mean={all_vals.mean():.2f}  std={all_vals.std():.2f}  "
              f"median={np.median(all_vals):.1f}")
        print(f"    Pos:     mean={pos_vals.mean():.2f}  std={pos_vals.std():.2f}")
        print(f"    Neg:     mean={neg_vals.mean():.2f}  std={neg_vals.std():.2f}")

    final = alt.vconcat(*charts).properties(
        title='New pipeline shard column distributions at CpG center (fi/ri and fp/rp mixed)'
    )
    final.save(output_path)
    print(f'\nSaved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
