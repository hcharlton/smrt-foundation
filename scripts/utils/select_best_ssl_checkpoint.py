"""Resolve the path to the highest-probe-accuracy SSL milestone checkpoint.

Used by the F-series fine-tune harnesses (supervised_53_finetune_revamp/*) to
load the *best* checkpoint per ssl_58 size, not the last. For some sizes
(particularly the smaller ones in the grid) probe_top1 peaks well before the
end of training; using `final_model.pt` blindly would understate the
representation quality available to fine-tune from.

CLI:
    python -m scripts.utils.select_best_ssl_checkpoint \
        scripts/experiments/ssl_58_autoencoder_grid/size_d512_L8

Importable:
    from scripts.utils.select_best_ssl_checkpoint import select_best_ssl_checkpoint
    path = select_best_ssl_checkpoint(exp_dir, smooth_window=3)
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml


def _load_run_metadata(exp_dir: str | Path) -> tuple[str, str]:
    """Return (experiment_type, experiment_name) from config.yaml. Falls back
    to ('ssl', basename(exp_dir)) if the config is missing or unreadable.
    """
    cfg_path = Path(exp_dir) / 'config.yaml'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return (
            str(cfg.get('experiment_type', 'ssl')),
            str(cfg.get('experiment_name', Path(exp_dir).name)),
        )
    return 'ssl', Path(exp_dir).name


def _latest_tb_run_dir(training_logs_root: Path, exp_type: str, exp_name: str) -> Path | None:
    """Locate the directory containing probe_history.csv for `<exp_type>/<exp_name>`.

    This codebase's Accelerate TB writer logs directly into
    `<training_logs_root>/<exp_type>/<exp_name>/` (flat layout — events
    files, hparams.yaml, and probe_history.csv all sit next to each
    other in the exp_name dir). Earlier versions of this resolver
    assumed a nested `run_N/` subdir layout that the writer never
    actually creates; that gave the false "No TB run dir" error
    despite probe_history.csv being right there.

    Flat layout is checked first; the nested fallback stays for
    compatibility with the test fixtures and any future writer change.
    """
    parent = training_logs_root / exp_type / exp_name
    if not parent.exists():
        return None
    # Flat layout: probe_history.csv directly under the exp_name dir.
    if (parent / 'probe_history.csv').exists():
        return parent
    # Nested fallback: a subdir containing probe_history.csv.
    candidates = [p for p in parent.iterdir() if p.is_dir() and (p / 'probe_history.csv').exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_step_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return the path to the highest-numbered step_<N>.pt in ckpt_dir, or None."""
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob('step_*.pt'))
    if not candidates:
        return None

    def _step_num(p: Path) -> int:
        try:
            return int(p.stem.split('_')[1])
        except (IndexError, ValueError):
            return -1
    return max(candidates, key=_step_num)


def _fallback_checkpoint(ckpt_dir: Path) -> Path | None:
    """When probe_history.csv is unreadable, prefer the highest-step
    milestone over `final_model.pt` (which most SSL runs never emit).
    Returns whichever exists, or None."""
    latest = _latest_step_checkpoint(ckpt_dir)
    if latest is not None:
        return latest
    final = ckpt_dir / 'final_model.pt'
    return final if final.exists() else None


def _read_probe_history(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                step = int(r['step'])
            except (KeyError, ValueError):
                continue
            try:
                top1 = float(r.get('probe_top1', 'nan'))
            except ValueError:
                top1 = float('nan')
            try:
                auroc = float(r.get('probe_auroc', 'nan'))
            except ValueError:
                auroc = float('nan')
            rows.append({'step': step, 'probe_top1': top1, 'probe_auroc': auroc})
    rows.sort(key=lambda r: r['step'])
    return rows


def _smooth(values: list[float], window: int) -> list[float]:
    """Centered moving-average smoothing. Window=1 means no smoothing.
    NaNs are excluded from each window's average; if a window is all-NaN the
    smoothed value is NaN."""
    if window <= 1:
        return list(values)
    out = []
    half = window // 2
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = [v for v in values[lo:hi] if v == v]  # NaN-safe
        out.append(sum(chunk) / len(chunk) if chunk else float('nan'))
    return out


DEFAULT_METRIC_CSV_PREFERENCE = ['probe_history_val1.csv', 'probe_history.csv']


def select_best_ssl_checkpoint(
    exp_dir: str | Path,
    metric: str = 'probe_top1',
    smooth_window: int = 1,
    min_step: int = 0,
    training_logs_root: str | Path = 'training_logs',
    metric_csv: str | None = None,
) -> str:
    """Return the absolute path to the best-metric step_<N>.pt for `exp_dir`.

    Algorithm:
      1. Resolve the metric CSV to argmax over. If `metric_csv` is None
         (the default), try `probe_history_val1.csv` first then
         `probe_history.csv`. This implements the post-2026-05-19 convention:
         if `scripts/utils/reprobe_ssl_checkpoint.py` has written a val1-based
         probe history, prefer it; otherwise fall back to the historical
         in-training probe history. If `metric_csv` is an explicit string,
         only that filename is tried.
      2. For each candidate filename: check `<exp_dir>/<filename>` (post-hoc
         reprobe convention, in the experiment dir alongside config.yaml).
         Then check `<training_logs_root>/<exp_type>/<exp_name>/<filename>`
         (in-training probe convention, where Accelerate's TB writer logs to).
         First existing file wins.
      3. Smooth the metric column with a centered moving average
         (window=smooth_window).
      4. Pick the step with the highest smoothed metric where step >= min_step.
      5. Map that step to `<exp_dir>/checkpoints/step_<step>.pt`. If that file
         doesn't exist (e.g., probe-eval landed but milestone not yet flushed
         to disk), fall back to the next-best step that does exist.
      6. If no metric CSV is readable or it has no usable rows, fall back to
         the highest-numbered `step_<N>.pt` in `checkpoints/`; then
         `final_model.pt`; otherwise raise FileNotFoundError.

    Raises FileNotFoundError if no usable checkpoint is found.
    """
    exp_dir = Path(exp_dir).resolve()
    ckpt_dir = exp_dir / 'checkpoints'
    candidates = [metric_csv] if metric_csv is not None else DEFAULT_METRIC_CSV_PREFERENCE

    csv_path: Path | None = None
    for cand in candidates:
        direct = exp_dir / cand
        if direct.exists():
            csv_path = direct
            break

    if csv_path is None:
        exp_type, exp_name = _load_run_metadata(exp_dir)
        run_dir = _latest_tb_run_dir(Path(training_logs_root).resolve(), exp_type, exp_name)
        if run_dir is not None:
            for cand in candidates:
                rd_csv = run_dir / cand
                if rd_csv.exists():
                    csv_path = rd_csv
                    break

    if csv_path is None:
        fb = _fallback_checkpoint(ckpt_dir)
        if fb is not None:
            return str(fb)
        raise FileNotFoundError(
            f"None of {candidates} found in {exp_dir} or under "
            f"{training_logs_root}/<type>/<name>/, and no step_<N>.pt or "
            f"final_model.pt in {ckpt_dir}. Cannot resolve a checkpoint."
        )

    rows = _read_probe_history(csv_path)
    rows = [r for r in rows if r['step'] >= min_step and r[metric] == r[metric]]  # NaN-drop
    if not rows:
        fb = _fallback_checkpoint(ckpt_dir)
        if fb is not None:
            return str(fb)
        raise FileNotFoundError(
            f"probe_history.csv at {csv_path} has no usable rows >= min_step={min_step} "
            f"and no step_<N>.pt or final_model.pt fallback exists."
        )

    smoothed = _smooth([r[metric] for r in rows], smooth_window)
    order = sorted(range(len(rows)), key=lambda i: smoothed[i], reverse=True)
    for idx in order:
        step = rows[idx]['step']
        candidate = ckpt_dir / f'step_{step}.pt'
        if candidate.exists():
            return str(candidate)
    fb = _fallback_checkpoint(ckpt_dir)
    if fb is not None:
        return str(fb)
    raise FileNotFoundError(
        f"No step_<N>.pt under {ckpt_dir} matches a probe_history.csv row "
        f"and no fallback (step_<N>.pt or final_model.pt) exists."
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument('exp_dir', help='SSL experiment directory')
    ap.add_argument('smooth_window', nargs='?', type=int, default=1,
                    help='Centered moving-average window (1 = no smoothing). '
                         'Positional for backwards compat with the old CLI.')
    ap.add_argument('--metric_csv', default=None,
                    help='CSV filename to argmax over. Default (None) auto-prefers '
                         'probe_history_val1.csv then probe_history.csv. Pass an explicit '
                         'filename to override. Checked under exp_dir first, then training_logs.')
    args = ap.parse_args()
    print(select_best_ssl_checkpoint(args.exp_dir, smooth_window=args.smooth_window,
                                     metric_csv=args.metric_csv))


if __name__ == '__main__':
    main()
