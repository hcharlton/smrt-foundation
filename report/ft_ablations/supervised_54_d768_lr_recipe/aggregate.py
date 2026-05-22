"""Stitch the supervised_54 per-leaf results.csv files plus the val3 evals
into one analysis-friendly polars frame.

Output:
    report/ft_ablations/supervised_54_d768_lr_recipe/combined_results.csv

Expected on-disk layout (written by scripts/ds_grid_v3.py and scripts/ft_eval.py):

    scripts/experiments/supervised_54_d768_lr_recipe/
        phase_a_lr_ablation/
            current_lr/<init>/<arch>/n<size>/results.csv
            sane_lr/<init>/<arch>/n<size>/results.csv
            frozen/<init>/<arch>/n<size>/results.csv
        phase_b_recipe_ablation/
            current_lr/<init>/<arch>/n<size>/results.csv
            sane_lr/<init>/<arch>/n<size>/results.csv

    report/ft_ablations/supervised_54_d768_lr_recipe/
        val3_eval_phase_a.csv   (from scripts/ft_eval.py evaluate)
        val3_eval_phase_b.csv

We stamp `phase` (`phase_a` / `phase_b`) and `cell` (the leaf dir name:
`current_lr` / `sane_lr` / `frozen`) onto every trajectory row, then
left-join the val3 numbers on (cell, init_name, arch, train_size) for the
winning step per combo. Cells whose runs have not landed yet are skipped
with a stderr note rather than failing -- partial aggregates are useful
while the queue is draining.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_ROOT = _REPO_ROOT / 'scripts' / 'experiments' / 'supervised_54_d768_lr_recipe'
_REPORT_ROOT = Path(__file__).resolve().parent

# (phase, cell) -> phase subdir relative to _EXP_ROOT
_PHASE_LAYOUT: dict[str, list[str]] = {
    'phase_a': ['current_lr', 'sane_lr', 'frozen'],
    'phase_b': ['current_lr', 'sane_lr'],
}


def _load_phase_trajectories(phase: str, cells: list[str]) -> pl.DataFrame:
    """Walk one phase's leaf directories and concat every per-combo results.csv.

    Stamps `phase`, `cell`, `arch`, `size` columns. Missing leaves print a
    warning and are skipped -- we want partial aggregates to work while
    submissions are still landing.
    """
    phase_dir = _EXP_ROOT / f'{phase}_lr_ablation' if phase == 'phase_a' else _EXP_ROOT / f'{phase}_recipe_ablation'
    if not phase_dir.is_dir():
        print(f"  skip {phase}: {phase_dir} not found", file=sys.stderr)
        return pl.DataFrame()

    frames = []
    for cell in cells:
        cell_dir = phase_dir / cell
        if not cell_dir.is_dir():
            print(f"  skip {phase}/{cell}: {cell_dir} not found", file=sys.stderr)
            continue
        csv_paths = list(cell_dir.rglob('results.csv'))
        if not csv_paths:
            print(f"  skip {phase}/{cell}: no results.csv yet (run pending?)", file=sys.stderr)
            continue
        for csv_path in csv_paths:
            rel = csv_path.relative_to(cell_dir)
            if len(rel.parts) != 4:
                # cell_dir / init / arch / nXXX / results.csv ; anything else
                # is junk (e.g. the merged top-level CSV, training_logs/...).
                continue
            init_name, arch, n_size, _ = rel.parts
            if not n_size.startswith('n'):
                continue
            try:
                size = int(n_size[1:])
            except ValueError:
                continue
            df = pl.read_csv(csv_path).with_columns([
                pl.lit(phase).alias('phase'),
                pl.lit(cell).alias('cell'),
                pl.lit(arch).alias('arch'),
                pl.lit(size, dtype=pl.Int64).alias('size'),
            ])
            frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how='diagonal_relaxed')


def _load_val3(phase: str) -> pl.DataFrame:
    """Read the per-phase val3 eval CSV written by `scripts.ft_eval evaluate`.

    Schema (from scripts/ft_eval.py:374): init_name, arch, size, recipe,
    step, val_metric, ckpt_path, params, test_top1, test_auroc, test_auprc,
    test_f1, test_loss.

    We rename `recipe` -> `cell` so the merge key matches the trajectory frame.
    """
    csv_path = _REPORT_ROOT / f'val3_eval_{phase}.csv'
    if not csv_path.exists():
        print(f"  skip {phase}: {csv_path.name} not found "
              f"(run `python -m scripts.ft_eval evaluate ...` first)", file=sys.stderr)
        return pl.DataFrame()
    df = pl.read_csv(csv_path)
    return df.rename({'recipe': 'cell'})


def aggregate() -> pl.DataFrame:
    traj_frames = []
    for phase, cells in _PHASE_LAYOUT.items():
        traj_frames.append(_load_phase_trajectories(phase, cells))
    non_empty = [f for f in traj_frames if not f.is_empty()]
    if not non_empty:
        print("No trajectory rows found across either phase; nothing to write.", file=sys.stderr)
        return pl.DataFrame()
    traj = pl.concat(non_empty, how='diagonal_relaxed')

    val3_frames = [_load_val3('phase_a'), _load_val3('phase_b')]
    val3 = pl.concat([f for f in val3_frames if not f.is_empty()], how='diagonal_relaxed') \
        if any(not f.is_empty() for f in val3_frames) else pl.DataFrame()

    # Left-join val3 numbers onto the trajectory row matching the winning step
    # for that (cell, init_name, arch, size). ft_eval already picked the best
    # step per combo on val_accuracy, so we join on `step` too.
    if not val3.is_empty():
        join_keys = ['cell', 'init_name', 'arch', 'size', 'step']
        val3_thin = val3.select(join_keys + [
            'val_metric', 'ckpt_path', 'params',
            'test_top1', 'test_auroc', 'test_auprc', 'test_f1', 'test_loss',
        ])
        combined = traj.join(val3_thin, on=join_keys, how='left')
    else:
        combined = traj

    # Stable sort: easier to scan / diff than random row order.
    sort_cols = [c for c in ['phase', 'cell', 'init_name', 'size', 'step']
                 if c in combined.columns]
    return combined.sort(sort_cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--out',
        default=str(_REPORT_ROOT / 'combined_results.csv'),
        help='Output CSV path. Default: <report dir>/combined_results.csv',
    )
    args = ap.parse_args()

    df = aggregate()
    if df.is_empty():
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df.head(5))


if __name__ == '__main__':
    main()
