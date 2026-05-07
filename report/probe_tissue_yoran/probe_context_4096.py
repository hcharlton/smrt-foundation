"""Pooled-summary HGB on the full 4096 window vs. cropped 2048.

Run: `python -m report.probe_tissue_yoran.probe_context_4096`

The deep model crops to 2048. If the full 4096 window beats cropped 2048
on the same probe, signal lives in the dropped half — the centre-crop is
throwing away information.

Output: results/probe_context_4096.csv
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


def _run(ctx, norm):
    limits = {'train': _shared.DEFAULT_TRAIN_LIMIT,
              'val_s1': _shared.DEFAULT_VAL_LIMIT,
              'val_s2': _shared.DEFAULT_VAL_LIMIT}
    parts = {sp: _shared.load_split(sp, norm_fn=norm, context=ctx, limit=limits[sp])
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
        rows.append({'context': ctx, 'split': sp, **m, 'fit_time_s': fit_s if sp == 'train' else None})
        print(f"  [ctx={ctx}] {sp:7s}  top1={m['top1']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}  ce={m['ce_loss']:.4f}")
    return rows


def main():
    _shared.ensure_dirs()
    _shared.assert_partition_sane()

    print("Computing KineticsNorm on train ...")
    norm = _shared.compute_norm()

    print("\n--- ctx=2048 (cropped, default) ---")
    rows_2048 = _run(2048, norm)
    print("\n--- ctx=4096 (full window) ---")
    rows_4096 = _run(4096, norm)

    pl.DataFrame(rows_2048 + rows_4096).write_csv(
        os.path.join(_shared.RESULTS_DIR, 'probe_context_4096.csv')
    )


if __name__ == '__main__':
    main()
