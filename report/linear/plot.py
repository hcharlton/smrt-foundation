"""
Linear separability of CpG methylation.

Fits logistic regression on normalized kinetics features and produces
diagnostic plots. Two models: full 32-position context (64 features)
and center-6 positions around the CpG site (12 features).
"""

import os
import sys
import argparse
import numpy as np
import torch
import polars as pl
import altair as alt
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report
from sklearn.decomposition import PCA

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.normalization import KineticsNorm


POS_TRAIN = 'data/01_processed/val_sets/cpg_pos_v2.memmap/train'
NEG_TRAIN = 'data/01_processed/val_sets/cpg_neg_v2.memmap/train'
POS_VAL = 'data/01_processed/val_sets/cpg_pos_v2.memmap/val'
NEG_VAL = 'data/01_processed/val_sets/cpg_neg_v2.memmap/val'
POS_FALLBACK = 'data/01_processed/val_sets/cpg_pos_subset.memmap/train'
NEG_FALLBACK = 'data/01_processed/val_sets/cpg_neg_subset.memmap/train'
LIMIT = 2_000_000
CONTEXT = 32
CENTER_SLICE = slice(13, 19)  # 6 positions around CpG (positions 13-18)
KIN_COLS = [1, 2]             # IPD, PW


def resolve_paths():
    pos_train = POS_TRAIN if os.path.isdir(os.path.expandvars(POS_TRAIN)) else POS_FALLBACK
    neg_train = NEG_TRAIN if os.path.isdir(os.path.expandvars(NEG_TRAIN)) else NEG_FALLBACK
    pos_val = POS_VAL if os.path.isdir(os.path.expandvars(POS_VAL)) else POS_FALLBACK
    neg_val = NEG_VAL if os.path.isdir(os.path.expandvars(NEG_VAL)) else NEG_FALLBACK
    return pos_train, neg_train, pos_val, neg_val


def load_data(pos_dir, neg_dir, norm, limit=LIMIT):
    """Load balanced dataset, return raw tensors (before feature extraction)."""
    ds = LabeledMemmapDataset(pos_dir, neg_dir, limit=limit, norm_fn=norm, balance=True)
    print(f"  {len(ds)} samples ({ds.pos_len} pos, {ds.neg_len} neg)")
    dl = DataLoader(ds, batch_size=min(len(ds), 500_000), num_workers=2)
    X_all, y_all = [], []
    for x_batch, y_batch in dl:
        X_all.append(x_batch)
        y_all.append(y_batch)
    return torch.cat(X_all), torch.cat(y_all)


def extract_features(X_raw, center_only=False):
    """Extract kinetics features from raw sample tensors."""
    if center_only:
        return X_raw[:, CENTER_SLICE, :][:, :, KIN_COLS].flatten(start_dim=1).numpy()
    return X_raw[:, :, KIN_COLS].flatten(start_dim=1).numpy()


def plot_roc(fpr_full, tpr_full, auroc_full, acc_full,
             fpr_c6, tpr_c6, auroc_c6, acc_c6, output_path):
    """ROC curves for both models overlaid."""
    df = pl.DataFrame({
        'FPR': np.concatenate([fpr_full, fpr_c6]),
        'TPR': np.concatenate([tpr_full, tpr_c6]),
        'model': (
            [f'Full context (AUROC={auroc_full:.3f}, Acc={acc_full:.3f})'] * len(fpr_full) +
            [f'Center-6 (AUROC={auroc_c6:.3f}, Acc={acc_c6:.3f})'] * len(fpr_c6)
        ),
    })

    roc = alt.Chart(df).mark_line(strokeWidth=2).encode(
        alt.X('FPR:Q').title('False Positive Rate'),
        alt.Y('TPR:Q').title('True Positive Rate'),
        alt.Color('model:N').title('Model'),
    )

    diagonal = alt.Chart(
        pl.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]})
    ).mark_line(strokeDash=[5, 5], color='gray').encode(x='x:Q', y='y:Q')

    chart = (roc + diagonal).properties(
        width=500, height=500,
        title='Logistic Regression — CpG methylation classification',
    )
    chart.save(output_path)
    print(f'Saved ROC to {output_path}')


def plot_calibration(y_prob, y_val, output_path):
    """Calibration plot: predicted probability vs observed frequency."""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_predicted, observed_freq = [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mean_predicted.append(y_prob[mask].mean())
        observed_freq.append(y_val[mask].mean())

    df = pl.DataFrame({
        'Mean predicted probability': mean_predicted,
        'Observed frequency': observed_freq,
    })

    points = alt.Chart(df).mark_circle(size=60, color='#e45756').encode(
        alt.X('Mean predicted probability:Q').scale(domain=[0, 1]),
        alt.Y('Observed frequency:Q').scale(domain=[0, 1]),
    )
    line = alt.Chart(df).mark_line(color='#e45756').encode(
        alt.X('Mean predicted probability:Q'),
        alt.Y('Observed frequency:Q'),
    )

    diagonal = alt.Chart(
        pl.DataFrame({'x': [0.0, 1.0], 'y': [0.0, 1.0]})
    ).mark_line(strokeDash=[5, 5], color='gray').encode(x='x:Q', y='y:Q')

    chart = (points + line + diagonal).properties(
        width=500, height=500,
        title='Calibration — predicted P(methylated) vs observed frequency',
    )
    chart.save(output_path)
    print(f'Saved calibration to {output_path}')


def plot_coefficients(clf, output_path):
    """Heatmap of logistic regression coefficients reshaped to (positions x features)."""
    coefs = clf.coef_[0].reshape(CONTEXT, len(KIN_COLS))

    rows = []
    feature_names = ['IPD', 'PW']
    for pos in range(CONTEXT):
        for fi, fname in enumerate(feature_names):
            rows.append({
                'Position': pos - CONTEXT // 2,  # center at 0
                'Feature': fname,
                'Coefficient': float(coefs[pos, fi]),
            })

    df = pl.DataFrame(rows)

    chart = alt.Chart(df).mark_rect().encode(
        alt.X('Position:O').title('Position (0 = CpG center)'),
        alt.Y('Feature:N'),
        alt.Color('Coefficient:Q').scale(scheme='redblue', domainMid=0).title('Weight'),
    ).properties(
        width=600, height=80,
        title='Logistic regression coefficients by position and feature',
    )
    chart.save(output_path)
    print(f'Saved coefficients to {output_path}')


def plot_center6_scatter(X_c6_val, y_val, output_path):
    """PCA scatter of center-6 features colored by class."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_c6_val)
    evr = pca.explained_variance_ratio_

    # Subsample for plotting
    rng = np.random.default_rng(42)
    n_plot = min(50_000, len(X_pca))
    idx = rng.choice(len(X_pca), n_plot, replace=False)

    df = pl.DataFrame({
        'PC1': X_pca[idx, 0],
        'PC2': X_pca[idx, 1],
        'class': ['methylated' if yi == 1 else 'unmethylated' for yi in y_val[idx]],
    })

    chart = alt.Chart(df).mark_circle(opacity=0.1, size=5).encode(
        alt.X('PC1:Q').title(f'PC1 ({evr[0]:.1%} var)'),
        alt.Y('PC2:Q').title(f'PC2 ({evr[1]:.1%} var)'),
        alt.Color('class:N', scale=alt.Scale(
            domain=['methylated', 'unmethylated'],
            range=['#e45756', '#4c78a8'],
        )),
    ).properties(
        width=500, height=500,
        title='Center-6 kinetics — PCA colored by class',
    )
    chart.save(output_path)
    print(f'Saved center-6 scatter to {output_path}')


def main(output_path):
    alt.data_transformers.enable('vegafusion')
    out_dir = os.path.dirname(output_path)

    pos_train, neg_train, pos_val, neg_val = resolve_paths()

    # Compute normalization from training data
    tmp_ds = LabeledMemmapDataset(pos_train, neg_train, limit=LIMIT)
    norm = KineticsNorm(tmp_ds, log_transform=True)
    print(f"Norm — means: {norm.means[[1,2]].tolist()}, stds: {norm.stds[[1,2]].tolist()}")
    del tmp_ds

    # Load train and val (raw tensors, extract features as needed)
    print("Loading train:")
    X_raw_train, y_train = load_data(pos_train, neg_train, norm)
    print("Loading val:")
    X_raw_val, y_val = load_data(pos_val, neg_val, norm)

    X_full_train = extract_features(X_raw_train)
    X_full_val = extract_features(X_raw_val)
    X_c6_train = extract_features(X_raw_train, center_only=True)
    X_c6_val = extract_features(X_raw_val, center_only=True)
    y_train_np = y_train.numpy()
    y_val_np = y_val.numpy()

    print(f"\nFull model: train={X_full_train.shape}, val={X_full_val.shape}")
    print(f"Center-6:   train={X_c6_train.shape}, val={X_c6_val.shape}")
    print(f"Train balance: {y_train_np.mean():.3f}, Val balance: {y_val_np.mean():.3f}")

    # --- Full context model ---
    print("\nFitting full context model...")
    clf_full = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_full.fit(X_full_train, y_train_np)

    y_prob_full = clf_full.predict_proba(X_full_val)[:, 1]
    y_pred_full = clf_full.predict(X_full_val)
    acc_full = accuracy_score(y_val_np, y_pred_full)
    auroc_full = roc_auc_score(y_val_np, y_prob_full)
    fpr_full, tpr_full, _ = roc_curve(y_val_np, y_prob_full)

    print(f"Full — Accuracy: {acc_full:.4f}, AUROC: {auroc_full:.4f}")
    print(classification_report(y_val_np, y_pred_full, target_names=['unmethylated', 'methylated']))

    # --- Center-6 model ---
    print("Fitting center-6 model...")
    clf_c6 = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf_c6.fit(X_c6_train, y_train_np)

    y_prob_c6 = clf_c6.predict_proba(X_c6_val)[:, 1]
    y_pred_c6 = clf_c6.predict(X_c6_val)
    acc_c6 = accuracy_score(y_val_np, y_pred_c6)
    auroc_c6 = roc_auc_score(y_val_np, y_prob_c6)
    fpr_c6, tpr_c6, _ = roc_curve(y_val_np, y_prob_c6)

    print(f"Center-6 — Accuracy: {acc_c6:.4f}, AUROC: {auroc_c6:.4f}")
    print(classification_report(y_val_np, y_pred_c6, target_names=['unmethylated', 'methylated']))

    # --- Generate plots ---
    plot_roc(fpr_full, tpr_full, auroc_full, acc_full,
             fpr_c6, tpr_c6, auroc_c6, acc_c6, output_path)

    plot_calibration(y_prob_full, y_val_np, os.path.join(out_dir, 'calibration.svg'))

    plot_coefficients(clf_full, os.path.join(out_dir, 'coefficients.svg'))

    plot_center6_scatter(X_c6_val, y_val_np, os.path.join(out_dir, 'center6_scatter.svg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
