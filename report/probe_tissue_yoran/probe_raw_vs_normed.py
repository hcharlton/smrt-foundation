"""Pooled-summary HGB with and without KineticsNorm.

Run: `python -m report.probe_tissue_yoran.probe_raw_vs_normed`

If raw kinetics (uint8 cast to float32, no log1p, no z-score) beat the
normalised version, the normalisation step is destroying signal. This
directly tests the "preprocessing is the problem" hypothesis the user
raised.

Output: results/probe_raw_vs_normed.csv
"""

import os
import time
import numpy as np
import polars as pl

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from . import _shared


def _evaluate(clf, F, y, n_classes):
    p = clf.predict(F)
    proba = clf.predict_proba(F)
    return {
        'top1': float(accuracy_score(y, p)),
        'macro_f1': float(f1_score(y, p, average='macro', zero_division=0.0)),
        'ce_loss': float(log_loss(y, proba, labels=np.arange(n_classes))),
    }


def _run(label, norm_fn):
    limits = {'train': _shared.DEFAULT_TRAIN_LIMIT,
              'val_s1': _shared.DEFAULT_VAL_LIMIT,
              'val_s2': _shared.DEFAULT_VAL_LIMIT}
    parts = {sp: _shared.load_split(sp, norm_fn=norm_fn, context=2048, limit=limits[sp])
             for sp in ('train', 'val_s1', 'val_s2')}
    F = {sp: _shared.pool_summary(parts[sp]['X']) for sp in parts}
    Y = {sp: parts[sp]['tissue_id'] for sp in parts}

    clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    t0 = time.perf_counter()
    clf.fit(F['train'], Y['train'])
    fit_s = time.perf_counter() - t0
    rows = []
    for sp in ('train', 'val_s1', 'val_s2'):
        m = _evaluate(clf, F[sp], Y[sp], n_classes=len(_shared.TISSUES))
        rows.append({'normed': label, 'split': sp, **m, 'fit_time_s': fit_s if sp == 'train' else None})
        print(f"  [{label:>6s}] {sp:7s}  top1={m['top1']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}  ce={m['ce_loss']:.4f}")
    return rows


def main():
    _shared.ensure_dirs()
    _shared.assert_partition_sane()

    print("--- raw (no KineticsNorm) ---")
    raw_rows = _run('raw', norm_fn=None)

    print("\n--- normed (KineticsNorm) ---")
    norm = _shared.compute_norm()
    normed_rows = _run('normed', norm_fn=norm)

    pl.DataFrame(raw_rows + normed_rows).write_csv(
        os.path.join(_shared.RESULTS_DIR, 'probe_raw_vs_normed.csv')
    )


if __name__ == '__main__':
    main()
