"""Cell-classifier probe: s1 vs s2 from the same pooled features.

Run: `python -m report.probe_tissue_yoran.probe_cell_s1_vs_s2`

The headline batch-effect number. If LogReg / RF on pooled summary features
classifies the cell (s1 vs s2) at ≫ 0.5 accuracy, there is strong cell-batch
signal in the same features the tissue probe sees. Cell separability and
within-cell tissue separability are independent claims, but if the first is
high and the second is low the tissue task is dominated by batch.

Train: 5k samples from each cell (10k total) drawn from manifest rows
*outside* val_s1 / val_s2, so this probe doesn't peek at the held-out reads.
Val: 1k from each cell drawn from val_s1 (cell s1) and val_s2 (cell s2).

Output: results/probe_cell_s1_vs_s2.csv
"""

import os
import time
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from . import _shared


PER_CELL_TRAIN = 5_000
PER_CELL_VAL = 1_000


def _load_train_pool(norm):
    """Pull samples from each cell using train (s1) + val_s2 (s2) sources.

    The yoran partition only puts cell-s1 reads into train; cell-s2 reads only
    appear in val_s2. To build a balanced cell-discrimination training set we
    take 5k from train (cell s1) and 5k from val_s2 (cell s2). This means
    val_s2 is no longer held-out *for the cell task*, but val_s1 still is —
    we evaluate on val_s1 (s1 reads, never used) and a fresh slice of val_s2
    that wasn't in the train pool.
    """
    train_s1 = _shared.load_split(
        'train', norm_fn=norm, context=2048, limit=PER_CELL_TRAIN,
    )
    train_s2 = _shared.load_split(
        'val_s2', norm_fn=norm, context=2048, limit=PER_CELL_TRAIN + PER_CELL_VAL,
    )
    # Split val_s2 into train and val pieces.
    n_train = PER_CELL_TRAIN
    train_s2_X = train_s2['X'][:n_train]
    train_s2_cell = train_s2['cell_id'][:n_train]
    val_s2_X = train_s2['X'][n_train:n_train + PER_CELL_VAL]
    val_s2_cell = train_s2['cell_id'][n_train:n_train + PER_CELL_VAL]

    val_s1 = _shared.load_split(
        'val_s1', norm_fn=norm, context=2048, limit=PER_CELL_VAL,
    )

    X_tr = np.concatenate([train_s1['X'], train_s2_X])
    y_tr = np.concatenate([train_s1['cell_id'], train_s2_cell])
    X_va = np.concatenate([val_s1['X'], val_s2_X])
    y_va = np.concatenate([val_s1['cell_id'], val_s2_cell])
    return X_tr, y_tr, X_va, y_va


def main():
    _shared.ensure_dirs()
    _shared.assert_partition_sane()

    print("Computing KineticsNorm on train ...")
    norm = _shared.compute_norm()

    print("Loading per-cell pool ...")
    X_tr, y_tr, X_va, y_va = _load_train_pool(norm)
    F_tr = _shared.pool_summary(X_tr)
    F_va = _shared.pool_summary(X_va)
    print(f"  train n={F_tr.shape[0]}, val n={F_va.shape[0]}, "
          f"train cell balance={y_tr.mean():.3f}, val balance={y_va.mean():.3f}")

    rows = []
    for name, clf in [
        ('logreg', LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)),
        ('random_forest', RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42,
        )),
    ]:
        t0 = time.perf_counter()
        clf.fit(F_tr, y_tr)
        fit_s = time.perf_counter() - t0
        proba = clf.predict_proba(F_va)[:, 1]
        pred = clf.predict(F_va)
        rows.append({
            'classifier': name,
            'val_top1': float(accuracy_score(y_va, pred)),
            'val_auroc': float(roc_auc_score(y_va, proba)),
            'fit_time_s': fit_s,
        })
        print(f"  {name:13s} val top1={rows[-1]['val_top1']:.4f}  "
              f"AUROC={rows[-1]['val_auroc']:.4f}  fit={fit_s:.1f}s")

    pl.DataFrame(rows).write_csv(
        os.path.join(_shared.RESULTS_DIR, 'probe_cell_s1_vs_s2.csv')
    )


if __name__ == '__main__':
    main()
