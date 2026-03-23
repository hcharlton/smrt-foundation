"""
Paired fi/ri correlation at CpG center positions.

Tests hypothesis: if fi and ri are uncorrelated, legacy's paired views
(single_strand=True showing both fi/fp and ri/rp perspectives) provide
valuable augmentation the new pipeline lacks.

Uses legacy parquet which has both fi and ri per CpG window.
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
    if len(df) > 50_000:
        df = df.sample(n=50_000, seed=42)

    kin_feats = ['fi', 'fp', 'ri', 'rp']
    df = df.with_columns([pl.col(c).list.to_array(CONTEXT) for c in kin_feats])

    center = CONTEXT // 2
    labels = df['label'].to_numpy()

    # Subsample for scatter readability
    rng = np.random.default_rng(42)
    n_plot = min(3000, len(df))
    idx = rng.choice(len(df), n_plot, replace=False)

    fi_vals = np.log1p(df['fi'].to_numpy()[idx, center].astype(np.float64))
    ri_vals = np.log1p(df['ri'].to_numpy()[idx, center].astype(np.float64))
    fp_vals = np.log1p(df['fp'].to_numpy()[idx, center].astype(np.float64))
    rp_vals = np.log1p(df['rp'].to_numpy()[idx, center].astype(np.float64))
    plot_labels = labels[idx]

    # Correlation stats
    fi_all = np.log1p(df['fi'].to_numpy()[:, center].astype(np.float64))
    ri_all = np.log1p(df['ri'].to_numpy()[:, center].astype(np.float64))
    fp_all = np.log1p(df['fp'].to_numpy()[:, center].astype(np.float64))
    rp_all = np.log1p(df['rp'].to_numpy()[:, center].astype(np.float64))

    corr_ipd = np.corrcoef(fi_all, ri_all)[0, 1]
    corr_pw = np.corrcoef(fp_all, rp_all)[0, 1]
    print(f"\nCorrelation at CpG center (log1p):")
    print(f"  fi vs ri (IPD):       r = {corr_ipd:.3f}")
    print(f"  fp vs rp (pulse width): r = {corr_pw:.3f}")

    # IPD scatter
    ipd_df = pl.DataFrame({
        'fi (log1p)': fi_vals,
        'ri (log1p)': ri_vals,
        'class': ['methylated' if l == 1 else 'unmethylated' for l in plot_labels],
    })

    ipd_scatter = alt.Chart(ipd_df).mark_circle(size=12, opacity=0.3).encode(
        alt.X('fi (log1p):Q').title('fi (forward IPD, log1p)'),
        alt.Y('ri (log1p):Q').title('ri (reverse IPD, log1p)'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8']
        )),
    ).properties(
        width=350, height=350,
        title=f'IPD: fi vs ri at CpG center (r={corr_ipd:.3f})'
    )

    # Pulse width scatter
    pw_df = pl.DataFrame({
        'fp (log1p)': fp_vals,
        'rp (log1p)': rp_vals,
        'class': ['methylated' if l == 1 else 'unmethylated' for l in plot_labels],
    })

    pw_scatter = alt.Chart(pw_df).mark_circle(size=12, opacity=0.3).encode(
        alt.X('fp (log1p):Q').title('fp (forward PW, log1p)'),
        alt.Y('rp (log1p):Q').title('rp (reverse PW, log1p)'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8']
        )),
    ).properties(
        width=350, height=350,
        title=f'PW: fp vs rp at CpG center (r={corr_pw:.3f})'
    )

    final = alt.hconcat(ipd_scatter, pw_scatter).properties(
        title='Paired forward/reverse kinetics correlation at CpG center'
    )
    final.save(output_path)
    print(f'\nSaved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
