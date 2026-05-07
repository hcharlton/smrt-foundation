"""Sequence-composition-only probe.

Run: `python -m report.probe_tissue_yoran.probe_seq_only`

Per-window base frequencies `{A, C, G, T, N}` only — no kinetics. If this
beats chance, tissue separability comes (at least partly) from sequence
content rather than kinetic modifications: mappability differences, GC bias,
repeat-content shifts in the read pools.

Output: results/probe_seq_only.csv
"""

import os
import time
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


def main():
    _shared.ensure_dirs()
    _shared.assert_partition_sane()

    print("Computing KineticsNorm on train (only used to load shards) ...")
    norm = _shared.compute_norm()

    limits = {'train': _shared.DEFAULT_TRAIN_LIMIT,
              'val_s1': _shared.DEFAULT_VAL_LIMIT,
              'val_s2': _shared.DEFAULT_VAL_LIMIT}
    parts = {sp: _shared.load_split(sp, norm_fn=norm, context=2048, limit=limits[sp])
             for sp in ('train', 'val_s1', 'val_s2')}

    print("Extracting seq-composition features (5-d) ...")
    F = {sp: _shared.seq_composition(parts[sp]['X']) for sp in parts}
    Y = {sp: parts[sp]['tissue_id'] for sp in parts}

    rows = []
    for name, clf in [
        ('logreg', LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)),
        ('random_forest', RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42,
        )),
    ]:
        t0 = time.perf_counter()
        clf.fit(F['train'], Y['train'])
        fit_s = time.perf_counter() - t0
        for sp in ('train', 'val_s1', 'val_s2'):
            m = _evaluate(clf, F[sp], Y[sp], n_classes=len(_shared.TISSUES))
            rows.append({'classifier': name, 'split': sp, **m,
                         'fit_time_s': fit_s if sp == 'train' else None})
            print(f"  {name:13s} {sp:7s}  top1={m['top1']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}  ce={m['ce_loss']:.4f}")

    pl.DataFrame(rows).write_csv(
        os.path.join(_shared.RESULTS_DIR, 'probe_seq_only.csv')
    )


if __name__ == '__main__':
    main()
