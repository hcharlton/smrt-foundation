"""PCA / UMAP scatter of pooled features, coloured by tissue and by cell.

Run: `python -m report.probe_tissue_yoran.eda_pca_umap`

Visual answer to "tissue-shaped clusters or cell-shaped clusters?". UMAP is
optional — falls back to PCA-only if the import fails.

Outputs:
  - figures/pca_pooled_by_tissue.svg
  - figures/pca_pooled_by_cell.svg
  - figures/umap_pooled_by_tissue.svg     (if umap-learn available)
  - figures/umap_pooled_by_cell.svg       (if umap-learn available)
"""

import os
import numpy as np
import polars as pl
import altair as alt

from sklearn.decomposition import PCA

from . import _shared


def _scatter(df, color_field, color_scale, title, out_path):
    chart = alt.Chart(df).mark_circle(size=12, opacity=0.45).encode(
        alt.X('x:Q'),
        alt.Y('y:Q'),
        alt.Color(f'{color_field}:N', scale=color_scale),
    ).properties(width=480, height=480, title=title)
    chart.save(out_path)
    print(f"  saved {out_path}")


def _stack_features(parts):
    """Concatenate train + both vals into one feature/label set."""
    F_all = np.concatenate([_shared.pool_summary(parts[sp]['X'])
                            for sp in ('train', 'val_s1', 'val_s2')])
    tissues = np.concatenate([parts[sp]['tissue_id'] for sp in ('train', 'val_s1', 'val_s2')])
    cells = np.concatenate([parts[sp]['cell_id'] for sp in ('train', 'val_s1', 'val_s2')])
    return F_all, tissues, cells


def main():
    _shared.ensure_dirs()
    _shared.assert_partition_sane()

    print("Computing KineticsNorm on train ...")
    norm = _shared.compute_norm()

    parts = {sp: _shared.load_split(sp, norm_fn=norm, context=2048,
                                    limit=_shared.DEFAULT_VAL_LIMIT * 2)
             for sp in ('train', 'val_s1', 'val_s2')}
    F, tissues, cells = _stack_features(parts)
    print(f"  combined feature matrix: {F.shape}")

    # Subsample to keep plots responsive.
    rng = np.random.default_rng(0)
    n_plot = min(6000, F.shape[0])
    idx = rng.choice(F.shape[0], n_plot, replace=False)
    F = F[idx]
    tissues = tissues[idx]
    cells = cells[idx]

    print("\nPCA(2) ...")
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(F)
    df = pl.DataFrame({
        'x': XY[:, 0], 'y': XY[:, 1],
        'tissue': _shared.tissue_label(tissues),
        'cell_label': _shared.cell_label(cells),
    })
    _scatter(df, 'tissue', _shared.tissue_color_scale(),
             f'PCA pooled features ({pca.explained_variance_ratio_[0]:.1%} + '
             f'{pca.explained_variance_ratio_[1]:.1%} var)',
             os.path.join(_shared.FIGURES_DIR, 'pca_pooled_by_tissue.svg'))
    _scatter(df, 'cell_label', _shared.cell_color_scale(),
             'PCA pooled features, coloured by cell',
             os.path.join(_shared.FIGURES_DIR, 'pca_pooled_by_cell.svg'))

    try:
        import umap
    except Exception as exc:
        print(f"\numap-learn not available ({exc}); skipping UMAP plots.")
        return

    print("\nUMAP(2) ...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    UV = reducer.fit_transform(F)
    df = pl.DataFrame({
        'x': UV[:, 0], 'y': UV[:, 1],
        'tissue': _shared.tissue_label(tissues),
        'cell_label': _shared.cell_label(cells),
    })
    _scatter(df, 'tissue', _shared.tissue_color_scale(),
             'UMAP pooled features, coloured by tissue',
             os.path.join(_shared.FIGURES_DIR, 'umap_pooled_by_tissue.svg'))
    _scatter(df, 'cell_label', _shared.cell_color_scale(),
             'UMAP pooled features, coloured by cell',
             os.path.join(_shared.FIGURES_DIR, 'umap_pooled_by_cell.svg'))


if __name__ == '__main__':
    main()
