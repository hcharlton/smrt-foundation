"""Pooled-summary feature probes for tissue classification.

Run: `python -m report.probe_tissue_yoran.probe_tissue_pooled`

Trains three classifiers on 25-d pooled summary features and reports
top-1, macro-F1, and cross-entropy on `train`, `val_s1`, `val_s2`. The
per-tissue confusion matrix on val_s1 and val_s2 from the best classifier
is saved as a heatmap.

Features (`_shared.pool_summary`):
  - For each kinetics channel `{fi, fp, ri, rp}` after `KineticsNorm`:
    `[mean, std, p10, p50, p90]` over the 2048-position window.
  - Plus base-frequency on the seq channel `{A, C, G, T, N}`.
  Total: 25 features.

Classifiers:
  - LogisticRegression(multi_class='multinomial', solver='lbfgs')
  - RandomForestClassifier(n_estimators=300)
  - HistGradientBoostingClassifier(max_iter=200)

Outputs:
  - results/probe_tissue_pooled.csv (rows appended to results/probe_matrix.csv)
  - figures/confusion_pooled_<best>_val_s1.svg, _val_s2.svg
"""

import os
import time
import numpy as np
import polars as pl
import altair as alt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix,
)

from . import _shared


REPRESENTATION = 'pooled_summary'
PROBE_MATRIX_CSV = os.path.join(_shared.RESULTS_DIR, 'probe_matrix.csv')
RESULTS_CSV = os.path.join(_shared.RESULTS_DIR, 'probe_tissue_pooled.csv')


def _build_classifiers():
    return {
        'logreg': LogisticRegression(
            solver='lbfgs', max_iter=2000, C=1.0,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42,
        ),
        'hist_gb': HistGradientBoostingClassifier(
            max_iter=200, random_state=42,
        ),
    }


def _evaluate(clf, F, y, n_classes):
    p = clf.predict(F)
    proba = clf.predict_proba(F)
    return {
        'top1': float(accuracy_score(y, p)),
        'macro_f1': float(f1_score(y, p, average='macro', zero_division=0.0)),
        'ce_loss': float(log_loss(y, proba, labels=np.arange(n_classes))),
    }


def _confusion_chart(y_true, y_pred, title, out_path):
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(_shared.TISSUES)), normalize='true',
    )
    rows = []
    for ti, true_t in enumerate(_shared.TISSUES):
        for pi, pred_t in enumerate(_shared.TISSUES):
            rows.append({'true': true_t, 'pred': pred_t, 'value': float(cm[ti, pi])})
    df = pl.DataFrame(rows)

    chart = alt.Chart(df).mark_rect().encode(
        alt.X('pred:N').title('Predicted tissue').sort(_shared.TISSUES),
        alt.Y('true:N').title('True tissue').sort(_shared.TISSUES),
        alt.Color('value:Q').scale(scheme='blues', domain=[0, 1]).title('P(pred|true)'),
    ).properties(width=320, height=320, title=title)
    chart.save(out_path)
    print(f"  saved {out_path}")


def _append_to_matrix(rows):
    df_new = pl.DataFrame(rows)
    if os.path.exists(PROBE_MATRIX_CSV):
        existing = pl.read_csv(PROBE_MATRIX_CSV)
        # Drop any prior rows for this representation so reruns are idempotent.
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

    limits = {
        'train': _shared.DEFAULT_TRAIN_LIMIT,
        'val_s1': _shared.DEFAULT_VAL_LIMIT,
        'val_s2': _shared.DEFAULT_VAL_LIMIT,
    }
    parts = {}
    for sp in ('train', 'val_s1', 'val_s2'):
        parts[sp] = _shared.load_split(sp, norm_fn=norm, context=2048, limit=limits[sp])

    print("Extracting pooled summary features ...")
    F = {sp: _shared.pool_summary(parts[sp]['X']) for sp in parts}
    Y = {sp: parts[sp]['tissue_id'] for sp in parts}
    print(f"  feature dim = {F['train'].shape[1]}; train n = {F['train'].shape[0]}")

    classifiers = _build_classifiers()
    rows = []
    per_classifier_predictions = {}
    for name, clf in classifiers.items():
        t0 = time.perf_counter()
        print(f"\nFitting {name} ...")
        clf.fit(F['train'], Y['train'])
        fit_s = time.perf_counter() - t0
        print(f"  fit took {fit_s:.1f} s")

        per_classifier_predictions[name] = {}
        for sp in ('train', 'val_s1', 'val_s2'):
            metrics = _evaluate(clf, F[sp], Y[sp], n_classes=len(_shared.TISSUES))
            row = {
                'representation': REPRESENTATION,
                'classifier': name,
                'split': sp,
                'top1': metrics['top1'],
                'macro_f1': metrics['macro_f1'],
                'ce_loss': metrics['ce_loss'],
                'fit_time_s': fit_s if sp == 'train' else None,
            }
            rows.append(row)
            print(f"  {sp:7s}  top1={metrics['top1']:.4f}  "
                  f"macro_f1={metrics['macro_f1']:.4f}  ce={metrics['ce_loss']:.4f}")
            per_classifier_predictions[name][sp] = clf.predict(F[sp])

    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_CSV)
    print(f"\nWrote {RESULTS_CSV}")
    _append_to_matrix(rows)

    # Confusion matrices for the best classifier on val_s1 (proxy for "what
    # the probe is actually learning"). Choose by macro_f1 on val_s1.
    val_rows = [r for r in rows if r['split'] == 'val_s1']
    best = max(val_rows, key=lambda r: r['macro_f1'])['classifier']
    print(f"\nBest classifier by val_s1 macro_f1: {best}")
    for sp in ('val_s1', 'val_s2'):
        _confusion_chart(
            Y[sp], per_classifier_predictions[best][sp],
            title=f'{REPRESENTATION} / {best} confusion on {sp}',
            out_path=os.path.join(_shared.FIGURES_DIR,
                                  f'confusion_pooled_{best}_{sp}.svg'),
        )


if __name__ == '__main__':
    main()
