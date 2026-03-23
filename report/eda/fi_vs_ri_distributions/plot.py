"""
fi vs ri (and fp vs rp) distributions at CpG center positions.

Tests hypothesis: if fi and ri have different distributions, the new pipeline's
ZNorm computes contaminated statistics (mixing fi+ri in the same column).
The legacy pipeline normalizes each feature independently.

Uses legacy parquet which stores all 4 kinetics per CpG window.
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

    center = CONTEXT // 2  # position of the C in CpG

    charts = []
    pairs = [('fi', 'ri', 'IPD'), ('fp', 'rp', 'Pulse width')]

    for fwd_feat, rev_feat, label in pairs:
        fwd_vals = df[fwd_feat].to_numpy()[:, center].astype(np.float64)
        rev_vals = df[rev_feat].to_numpy()[:, center].astype(np.float64)

        # Raw values
        n = min(len(fwd_vals), len(rev_vals))
        raw_df = pl.DataFrame({
            'value': np.concatenate([fwd_vals[:n], rev_vals[:n]]),
            'feature': [fwd_feat] * n + [rev_feat] * n,
        })

        raw_chart = alt.Chart(raw_df).mark_area(
            opacity=0.5, interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=60).title(f'{label} (raw)'),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('feature:N', scale=alt.Scale(
                domain=[fwd_feat, rev_feat],
                range=['#4c78a8', '#e45756']
            )),
        ).properties(width=350, height=200, title=f'{label}: {fwd_feat} vs {rev_feat} (raw)')

        # Log-transformed values
        fwd_log = np.log1p(fwd_vals)
        rev_log = np.log1p(rev_vals)

        log_df = pl.DataFrame({
            'value': np.concatenate([fwd_log[:n], rev_log[:n]]),
            'feature': [fwd_feat] * n + [rev_feat] * n,
        })

        log_chart = alt.Chart(log_df).mark_area(
            opacity=0.5, interpolate='step'
        ).encode(
            alt.X('value:Q').bin(maxbins=60).title(f'{label} (log1p)'),
            alt.Y('count():Q').stack(None).title('Count'),
            alt.Color('feature:N', scale=alt.Scale(
                domain=[fwd_feat, rev_feat],
                range=['#4c78a8', '#e45756']
            )),
        ).properties(width=350, height=200, title=f'{label}: {fwd_feat} vs {rev_feat} (log1p)')

        charts.append(alt.hconcat(raw_chart, log_chart))

    # Summary statistics
    for fwd_feat, rev_feat, label in pairs:
        fwd_vals = df[fwd_feat].to_numpy()[:, center].astype(np.float64)
        rev_vals = df[rev_feat].to_numpy()[:, center].astype(np.float64)
        print(f"\n{label} at CpG center:")
        print(f"  {fwd_feat}: mean={fwd_vals.mean():.2f}  std={fwd_vals.std():.2f}  median={np.median(fwd_vals):.1f}")
        print(f"  {rev_feat}: mean={rev_vals.mean():.2f}  std={rev_vals.std():.2f}  median={np.median(rev_vals):.1f}")
        print(f"  mean diff: {abs(fwd_vals.mean() - rev_vals.mean()):.2f}")

    final = alt.vconcat(*charts).properties(
        title='Forward vs reverse kinetics distributions at CpG center'
    )
    final.save(output_path)
    print(f'\nSaved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
