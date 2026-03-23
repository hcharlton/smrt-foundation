"""
Class separation by kinetics feature at CpG center.

Tests hypothesis: if fi separates pos/neg better than ri (or vice versa),
mixing them in the same model column dilutes the classification signal.

Uses legacy parquet which has all 4 kinetics + labels.
"""

import os
import sys
import argparse
import numpy as np
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

LEGACY_PATH = 'data/01_processed/val_sets/pacbio_standard_train.parquet'
LEGACY_FALLBACK = 'data/01_processed/val_sets/legacy_subset_train.parquet'
CONTEXT = 32


def main(output_path):
    path = LEGACY_PATH if os.path.exists(LEGACY_PATH) else LEGACY_FALLBACK
    if not os.path.exists(path):
        print(f"ERROR: No legacy parquet found at {LEGACY_PATH} or {LEGACY_FALLBACK}")
        sys.exit(1)

    df = pl.read_parquet(path)
    if len(df) > 100_000:
        df = df.sample(n=100_000, seed=42)

    kin_feats = ['fi', 'fp', 'ri', 'rp']
    df = df.with_columns([pl.col(c).list.to_array(CONTEXT) for c in kin_feats])

    center = CONTEXT // 2
    labels = df['label'].to_numpy()

    charts = []
    feat_labels = {
        'fi': 'Forward IPD (fi)',
        'fp': 'Forward PW (fp)',
        'ri': 'Reverse IPD (ri)',
        'rp': 'Reverse PW (rp)',
    }

    print("\nClass separation at CpG center (log1p values):")
    print(f"{'Feature':<20} {'pos mean':>10} {'neg mean':>10} {'|diff|':>10} {'Cohen d':>10}")
    print("-" * 60)

    for feat in kin_feats:
        vals = np.log1p(df[feat].to_numpy()[:, center].astype(np.float64))
        pos_vals = vals[labels == 1]
        neg_vals = vals[labels == 0]

        diff = abs(pos_vals.mean() - neg_vals.mean())
        pooled_std = np.sqrt((pos_vals.std()**2 + neg_vals.std()**2) / 2)
        cohen_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"{feat:<20} {pos_vals.mean():>10.3f} {neg_vals.mean():>10.3f} {diff:>10.3f} {cohen_d:>10.3f}")

        n = len(vals)
        feat_df = pl.DataFrame({
            'value': vals,
            'class': ['methylated' if l == 1 else 'unmethylated' for l in labels],
        })

        chart = alt.Chart(feat_df).mark_area(
            opacity=0.5, interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=50).title(f'{feat_labels[feat]} (log1p)'),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('class:N', scale=alt.Scale(
                domain=['methylated', 'unmethylated'],
                range=['#e45756', '#4c78a8']
            )),
        ).properties(width=350, height=200, title=feat_labels[feat])

        charts.append(chart)

    final = alt.concat(*charts, columns=2).properties(
        title='Class separation by kinetics feature at CpG center'
    )
    final.save(output_path)
    print(f'\nSaved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
