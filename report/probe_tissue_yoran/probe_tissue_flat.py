"""Flattened raw kinetics probe for tissue classification.

Run: `python -m report.probe_tissue_yoran.probe_tissue_flat`

LogReg L2 (saga solver) on the full flattened normalised kinetics window.
At ctx=2048 × 4 channels = 8192 features. Skips RF and HGB because both
scale poorly with feature count and the linear-vs-pooled comparison is the
useful contrast here. If this beats the pooled-summary LogReg, the
per-position information matters; if not, summary features are sufficient
for the task.

Outputs: `results/probe_tissue_flat.csv`, rows appended to
`results/probe_matrix.csv`.
"""

import os
import time
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from . import _shared


REPRESENTATION = 'flat_kinetics_8192'
PROBE_MATRIX_CSV = os.path.join(_shared.RESULTS_DIR, 'probe_matrix.csv')
RESULTS_CSV = os.path.join(_shared.RESULTS_DIR, 'probe_tissue_flat.csv')


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

    print("Extracting flattened kinetics features ...")
    F = {sp: _shared.flatten_kinetics(parts[sp]['X']) for sp in parts}
    Y = {sp: parts[sp]['tissue_id'] for sp in parts}
    print(f"  feature dim = {F['train'].shape[1]}; train n = {F['train'].shape[0]}")

    clf = LogisticRegression(
        solver='saga', penalty='l2', C=1.0, max_iter=200, n_jobs=-1,
    )
    rows = []
    name = 'logreg_l2_saga'
    t0 = time.perf_counter()
    print(f"\nFitting {name} ... (saga, max_iter=200)")
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
