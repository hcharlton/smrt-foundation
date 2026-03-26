"""
UMAP projection of CpG kinetics features.

Compares to the PCA projection (cpg_pca_v2) — UMAP preserves local
structure and may reveal non-linear class separation that PCA misses.
"""

import os
import sys
import argparse
import numpy as np
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
import umap

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.normalization import KineticsNorm


POS_TRAIN = 'data/01_processed/val_sets/cpg_pos_v2.memmap/train'
NEG_TRAIN = 'data/01_processed/val_sets/cpg_neg_v2.memmap/train'
POS_FALLBACK = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
NEG_FALLBACK = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'
LIMIT = 2_000_000
N_UMAP = 200_000  # UMAP subsample size (fitting on 2M is prohibitively slow)
N_PLOT = 50_000   # points to render in the scatter


def main(output_path):
    alt.data_transformers.enable('vegafusion')

    pos_dir = POS_TRAIN if os.path.isdir(os.path.expandvars(POS_TRAIN)) else POS_FALLBACK
    neg_dir = NEG_TRAIN if os.path.isdir(os.path.expandvars(NEG_TRAIN)) else NEG_FALLBACK

    # Compute normalization stats
    tmp_ds = LabeledMemmapDataset(pos_dir, neg_dir, limit=LIMIT)
    norm = KineticsNorm(tmp_ds, log_transform=True)
    del tmp_ds

    # Load balanced dataset
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

    # Subsample for UMAP fitting
    rng = np.random.default_rng(42)
    n_umap = min(N_UMAP, len(X))
    idx = rng.choice(len(X), n_umap, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    print(f"Fitting UMAP on {n_umap:,} samples...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1)
    embedding = reducer.fit_transform(X_sub)
    print("UMAP complete.")

    # Subsample for plotting
    n_plot = min(N_PLOT, len(embedding))
    plot_idx = rng.choice(len(embedding), n_plot, replace=False)

    df = pl.DataFrame({
        'UMAP1': embedding[plot_idx, 0],
        'UMAP2': embedding[plot_idx, 1],
        'class': ['methylated' if yi == 1 else 'unmethylated' for yi in y_sub[plot_idx]],
    })

    chart = alt.Chart(df).mark_circle(opacity=0.15, size=5).encode(
        alt.X('UMAP1:Q'),
        alt.Y('UMAP2:Q'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8'],
        )),
    ).properties(
        width=600,
        height=600,
        title=f'UMAP of CpG kinetics (log1p z-scored) — {n_umap:,} samples',
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
