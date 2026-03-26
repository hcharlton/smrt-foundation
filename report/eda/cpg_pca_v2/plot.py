"""
PCA of CpG kinetics using the v2 memmap pipeline.

Uses LabeledMemmapDataset + KineticsNorm (matching training) to load
balanced methylated/unmethylated samples, then runs PCA on the
normalized kinetics features (IPD + PW across context window).
"""

import os
import sys
import argparse
import numpy as np
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.normalization import KineticsNorm
from sklearn.decomposition import PCA


POS_TRAIN = 'data/01_processed/val_sets/cpg_pos_v2.memmap/train'
NEG_TRAIN = 'data/01_processed/val_sets/cpg_neg_v2.memmap/train'
POS_FALLBACK = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
NEG_FALLBACK = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'
LIMIT = 2_000_000


def main(output_path):
    alt.data_transformers.enable('vegafusion')

    pos_dir = POS_TRAIN if os.path.isdir(os.path.expandvars(POS_TRAIN)) else POS_FALLBACK
    neg_dir = NEG_TRAIN if os.path.isdir(os.path.expandvars(NEG_TRAIN)) else NEG_FALLBACK

    # Compute normalization stats
    tmp_ds = LabeledMemmapDataset(pos_dir, neg_dir, limit=LIMIT)
    norm = KineticsNorm(tmp_ds, log_transform=True)
    del tmp_ds

    # Load balanced dataset with normalization
    ds = LabeledMemmapDataset(pos_dir, neg_dir, limit=LIMIT, norm_fn=norm, balance=True)
    print(f"Dataset: {len(ds)} samples ({ds.pos_len} pos, {ds.neg_len} neg)")

    dl = DataLoader(ds, batch_size=min(len(ds), 500_000), num_workers=2)
    X_all, y_all = [], []
    for x_batch, y_batch in dl:
        X_all.append(x_batch[:, :, [1, 2]].flatten(start_dim=1))
        y_all.append(y_batch)
    X = torch.cat(X_all).numpy()
    y = torch.cat(y_all).numpy()

    print(f"Feature matrix: {X.shape}, class balance: {y.mean():.3f}")

    # PCA
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    print(f"Explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f}")

    # Subsample for plotting (full dataset makes SVG too large)
    rng = np.random.default_rng(42)
    n_plot = min(50_000, len(X_reduced))
    idx = rng.choice(len(X_reduced), n_plot, replace=False)

    df = pl.DataFrame({
        'PC1': X_reduced[idx, 0],
        'PC2': X_reduced[idx, 1],
        'class': ['methylated' if yi == 1 else 'unmethylated' for yi in y[idx]],
    })

    chart = alt.Chart(df).mark_circle(opacity=0.1, size=5).encode(
        alt.X('PC1:Q'),
        alt.Y('PC2:Q'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8'],
        )),
    ).properties(
        width=600,
        height=600,
        title=f'PCA of CpG kinetics — {len(X):,} samples, '
              f'var explained: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}',
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
