"""One-vs-rest binary tissue probes on pooled features.

Run: `python -m report.probe_tissue_yoran.probe_per_tissue_one_vs_rest`

Eight binary LogReg probes, one per tissue. Reports per-tissue val_s1 and
val_s2 AUROC. Catches the case where one or two tissues are separable but
the 8-way average masks it.

Outputs:
  - results/probe_per_tissue_one_vs_rest.csv
  - figures/per_tissue_auroc.svg
"""

import os
import time
import numpy as np
import polars as pl
import altair as alt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import _shared


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

    print("Extracting pooled summary features ...")
    F = {sp: _shared.pool_summary(parts[sp]['X']) for sp in parts}
    Y = {sp: parts[sp]['tissue_id'] for sp in parts}

    rows = []
    for tissue_id, tissue_name in enumerate(_shared.TISSUES):
        y_train = (Y['train'] == tissue_id).astype(np.int64)
        if y_train.sum() == 0:
            continue
        clf = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        t0 = time.perf_counter()
        clf.fit(F['train'], y_train)
        fit_s = time.perf_counter() - t0
        for sp in ('val_s1', 'val_s2'):
            y_eval = (Y[sp] == tissue_id).astype(np.int64)
            if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
                auroc = float('nan')
            else:
                proba = clf.predict_proba(F[sp])[:, 1]
                auroc = float(roc_auc_score(y_eval, proba))
            rows.append({
                'tissue': tissue_name, 'split': sp, 'auroc': auroc,
                'n_pos': int(y_eval.sum()), 'n_neg': int(len(y_eval) - y_eval.sum()),
                'fit_time_s': fit_s,
            })
            print(f"  {tissue_name:8s} {sp:7s}  AUROC={auroc:.4f}")

    df = pl.DataFrame(rows)
    df.write_csv(os.path.join(_shared.RESULTS_DIR, 'probe_per_tissue_one_vs_rest.csv'))

    chart = alt.Chart(df).mark_bar().encode(
        alt.X('tissue:N').sort(_shared.TISSUES),
        alt.Y('auroc:Q').scale(domain=[0.4, 1.0]).title('AUROC (1 vs rest)'),
        alt.Color('split:N'),
        alt.XOffset('split:N'),
    ).properties(
        width=520, height=300,
        title='Per-tissue 1-vs-rest AUROC on pooled summary features',
    )
    chart.save(os.path.join(_shared.FIGURES_DIR, 'per_tissue_auroc.svg'))
    print("Saved figures/per_tissue_auroc.svg")


if __name__ == '__main__':
    main()
