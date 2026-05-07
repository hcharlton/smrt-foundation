"""Binned summary feature probes for tissue classification.

Run: `python -m report.probe_tissue_yoran.probe_tissue_binned`

Same probe matrix as `probe_tissue_pooled` but with locality-aware features:
divide the 2048-position window into 16 bins; per bin per kinetics channel
report `[mean, std]`. 16 * 4 * 2 = 128 features. Catches per-bin shifts that
the global pooled-summary averages away.

Outputs: `results/probe_tissue_binned.csv`, rows appended to
`results/probe_matrix.csv`.
"""

import os
import time
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from . import _shared


REPRESENTATION = 'binned_summary_n16'
PROBE_MATRIX_CSV = os.path.join(_shared.RESULTS_DIR, 'probe_matrix.csv')
RESULTS_CSV = os.path.join(_shared.RESULTS_DIR, 'probe_tissue_binned.csv')


def _build_classifiers():
    return {
        'logreg': LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0),
        'random_forest': RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42,
        ),
        'hist_gb': HistGradientBoostingClassifier(max_iter=200, random_state=42),
    }


def _evaluate(clf, F, y, n_classes):
    p = clf.predict(F)
    proba = clf.predict_proba(F)
    return {
        'top1': float(accuracy_score(y, p)),
        'macro_f1': float(f1_score(y, p, average='macro', zero_division=0.0)),
        'ce_loss': float(log_loss(y, proba, labels=np.arange(n_classes))),
    }


def _append_to_matrix(rows):
    df_new = pl.DataFrame(rows)
    if os.path.exists(PROBE_MATRIX_CSV):
        existing = pl.read_csv(PROBE_MATRIX_CSV)
        existing = existing.filter(pl.col('representation') != REPRESENTATION)
        df = pl.concat([existing, df_new], how='diagonal_relaxed')
    else:
        df = df_new
    df.write_csv(PROBE_MATRIX_CSV)
    print(f"  appended {df_new.height} rows to {PROBE_MATRIX_CSV}")


def main():
    _shared.ensure_dirs()
    _shared.assert_partition_sane()

    print("Computing KineticsNorm on train ...")
    norm = _shared.compute_norm()

    limits = {'train': _shared.DEFAULT_TRAIN_LIMIT,
              'val_s1': _shared.DEFAULT_VAL_LIMIT,
              'val_s2': _shared.DEFAULT_VAL_LIMIT}
    parts = {sp: _shared.load_split(sp, norm_fn=norm, context=2048, limit=limits[sp])
             for sp in ('train', 'val_s1', 'val_s2')}

    print("Extracting binned summary features (16 bins) ...")
    F = {sp: _shared.bin_summary(parts[sp]['X'], n_bins=16) for sp in parts}
    Y = {sp: parts[sp]['tissue_id'] for sp in parts}
    print(f"  feature dim = {F['train'].shape[1]}; train n = {F['train'].shape[0]}")

    rows = []
    for name, clf in _build_classifiers().items():
        t0 = time.perf_counter()
        print(f"\nFitting {name} ...")
        clf.fit(F['train'], Y['train'])
        fit_s = time.perf_counter() - t0
        print(f"  fit took {fit_s:.1f} s")
        for sp in ('train', 'val_s1', 'val_s2'):
            m = _evaluate(clf, F[sp], Y[sp], n_classes=len(_shared.TISSUES))
            rows.append({
                'representation': REPRESENTATION, 'classifier': name, 'split': sp,
                'top1': m['top1'], 'macro_f1': m['macro_f1'], 'ce_loss': m['ce_loss'],
                'fit_time_s': fit_s if sp == 'train' else None,
            })
            print(f"  {sp:7s}  top1={m['top1']:.4f}  macro_f1={m['macro_f1']:.4f}  ce={m['ce_loss']:.4f}")

    pl.DataFrame(rows).write_csv(RESULTS_CSV)
    print(f"\nWrote {RESULTS_CSV}")
    _append_to_matrix(rows)


if __name__ == '__main__':
    main()
