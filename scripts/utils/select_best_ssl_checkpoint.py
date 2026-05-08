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
import sys
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
    """The TB writer creates a fresh subdir per Accelerate `init_trackers` call.
    Pick the most-recently-modified one."""
    parent = training_logs_root / exp_type / exp_name
    if not parent.exists():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


def select_best_ssl_checkpoint(
    exp_dir: str | Path,
    metric: str = 'probe_top1',
    smooth_window: int = 1,
    min_step: int = 0,
    training_logs_root: str | Path = 'training_logs',
) -> str:
    """Return the absolute path to the best-metric step_<N>.pt for `exp_dir`.

    Algorithm:
      1. Read experiment_type/name from `<exp_dir>/config.yaml`.
      2. Locate the most-recent TB run dir under
         `<training_logs_root>/<type>/<name>/`.
      3. Read `probe_history.csv`. Smooth the metric column with a centered
         moving average (window=smooth_window).
      4. Pick the step with the highest smoothed metric where step >= min_step.
      5. Map that step to `<exp_dir>/checkpoints/step_<step>.pt`. If that file
         doesn't exist (e.g., probe-eval landed but milestone not yet flushed
         to disk), fall back to the next-best step that does exist.
      6. If no probe_history.csv is readable or the candidates list is empty,
         fall back to `<exp_dir>/checkpoints/final_model.pt`. If that doesn't
         exist either, raise FileNotFoundError.

    Raises FileNotFoundError if no usable checkpoint is found.
    """
    exp_dir = Path(exp_dir).resolve()
    ckpt_dir = exp_dir / 'checkpoints'
    final_path = ckpt_dir / 'final_model.pt'

    exp_type, exp_name = _load_run_metadata(exp_dir)
    run_dir = _latest_tb_run_dir(Path(training_logs_root).resolve(), exp_type, exp_name)
    if run_dir is None:
        if final_path.exists():
            return str(final_path)
        raise FileNotFoundError(
            f"No TB run dir under {training_logs_root}/{exp_type}/{exp_name}/ "
            f"and no final_model.pt at {final_path}. Cannot resolve a checkpoint."
        )

    csv_path = run_dir / 'probe_history.csv'
    if not csv_path.exists():
        if final_path.exists():
            return str(final_path)
        raise FileNotFoundError(
            f"No probe_history.csv in {run_dir} and no final_model.pt at {final_path}."
        )

    rows = _read_probe_history(csv_path)
    rows = [r for r in rows if r['step'] >= min_step and r[metric] == r[metric]]  # NaN-drop
    if not rows:
        if final_path.exists():
            return str(final_path)
        raise FileNotFoundError(
            f"probe_history.csv at {csv_path} has no usable rows >= min_step={min_step}."
        )

    smoothed = _smooth([r[metric] for r in rows], smooth_window)
    order = sorted(range(len(rows)), key=lambda i: smoothed[i], reverse=True)
    for idx in order:
        step = rows[idx]['step']
        candidate = ckpt_dir / f'step_{step}.pt'
        if candidate.exists():
            return str(candidate)
    if final_path.exists():
        return str(final_path)
    raise FileNotFoundError(
        f"No step_<N>.pt under {ckpt_dir} matches a probe_history.csv row "
        f"and no final_model.pt fallback exists."
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    exp_dir = sys.argv[1]
    smooth_window = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(select_best_ssl_checkpoint(exp_dir, smooth_window=smooth_window))


if __name__ == '__main__':
    main()
