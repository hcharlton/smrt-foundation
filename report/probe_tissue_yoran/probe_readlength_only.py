"""Read-length-only baseline.

Run: `python -m report.probe_tissue_yoran.probe_readlength_only`

LogReg on a single feature: `read_length` from the manifest. If accuracy is
meaningfully above chance (1/8 = 0.125), tissue separability is partly
length-driven (selection bias from upstream sample prep). Doesn't load any
shard data, so it's effectively instant.

Output: results/probe_readlength_only.csv
"""

import os
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
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

    partition = _shared.load_partition()
    manifest = _shared.load_manifest()

    parts = {}
    for sp in ('train', 'val_s1', 'val_s2'):
        names = partition.filter(pl.col('split') == sp)['read_name'].to_list()
        m = manifest.filter(pl.col('read_name').is_in(names))
        parts[sp] = {
            'F': m['read_length'].to_numpy().astype(np.float32).reshape(-1, 1),
            'y': m['tissue_id'].to_numpy().astype(np.int64),
        }

    print("Read-length stats per split:")
    for sp, d in parts.items():
        print(f"  {sp:7s} n={d['F'].shape[0]}  "
              f"length mean={d['F'].mean():.0f}  std={d['F'].std():.0f}")

    clf = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    clf.fit(parts['train']['F'], parts['train']['y'])

    rows = []
    for sp, d in parts.items():
        m = _evaluate(clf, d['F'], d['y'], n_classes=len(_shared.TISSUES))
        rows.append({'split': sp, **m})
        print(f"  {sp:7s}  top1={m['top1']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"ce={m['ce_loss']:.4f}")

    pl.DataFrame(rows).write_csv(
        os.path.join(_shared.RESULTS_DIR, 'probe_readlength_only.csv')
    )


if __name__ == '__main__':
    main()
