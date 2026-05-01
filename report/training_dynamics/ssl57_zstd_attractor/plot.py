"""
embed_z_std vs cumulative learning rate for ssl_57 d=512_L8 (300k vs 1m).

Tests the shared-attractor hypothesis: both runs descend the same trajectory
in z_std-space; the 1m run's longer warmup integrates more "optimizer force"
per step, so it reaches the collapsed region faster. If the curves overlay
when re-parameterized by cumulative learning rate (instead of step count),
the hypothesis holds and the 300k run is also on its way to collapse, just
saved by the cosine schedule running out of LR.

Method:
  1. Read embed_z_std and learning_rate scalars from each run's TB events.
  2. Trapezoidal integration of LR over steps -> cumulative LR per run.
  3. Interpolate cumulative LR onto each z_std measurement's step.
  4. Plot z_std vs step (left, the obvious view) and z_std vs cumulative LR
     (right, the hypothesis-test view).

Output is a two-panel figure. Read it as: if right panel curves overlay where
the left panel curves diverge, the divergence is explained by integrated LR
alone (shared dynamics). If the right panel curves still diverge at the same
z_std values, something other than LR-integrated optimizer force is at play.

Usage (via plot.sh):
  bash plot.sh report/training_dynamics/ssl57_zstd_attractor

Pre-requisite: the cluster's training_logs/ssl/57_inputmask_grid_lnhead_*
directories must be synced locally.
"""

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Edit these if the run dirs are elsewhere. Each is the experiment-name dir
# under training_logs/<exp_type>/ that accelerator.init_trackers produces.
RUNS = {
    '300k': 'training_logs/ssl/57_inputmask_grid_lnhead_d512_L8',
    '1m':   'training_logs/ssl/57_inputmask_grid_lnhead_d512_L8_1m',
}
COLORS = {'300k': 'tab:gray', '1m': 'tab:purple'}


def find_event_dir(run_dir):
    """Return the directory containing TB event files for `run_dir`.

    Handles both layouts: events in run_dir directly, or in a single
    timestamped subdir (the most recent if multiple).
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if glob.glob(os.path.join(run_dir, 'events.out.tfevents.*')):
        return run_dir
    subdirs_with_events = [
        d for d in glob.glob(os.path.join(run_dir, '*'))
        if os.path.isdir(d) and glob.glob(os.path.join(d, 'events.out.tfevents.*'))
    ]
    if not subdirs_with_events:
        raise FileNotFoundError(
            f"No TB events in {run_dir} or its immediate subdirs. "
            f"Has the cluster's training_logs/ been synced locally?"
        )
    return max(subdirs_with_events, key=os.path.getmtime)


def load_scalar(run_dir, tag):
    """Read a single scalar series; return (steps, values) sorted by step."""
    event_dir = find_event_dir(run_dir)
    acc = EventAccumulator(event_dir, size_guidance={'scalars': 0})  # 0 = unlimited
    acc.Reload()
    if tag not in acc.Tags()['scalars']:
        available = acc.Tags()['scalars']
        raise KeyError(
            f"Tag {tag!r} not in TB events at {event_dir}. "
            f"Available scalars (first 20): {available[:20]}"
        )
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    order = np.argsort(steps)
    return steps[order], values[order]


def cumulative_lr(steps, lrs):
    """Trapezoidal integration of LR over steps. Returns array same length as
    `steps` with cum[0] = 0."""
    if len(steps) < 2:
        return np.zeros_like(lrs)
    dx = np.diff(steps)
    avg_lr = (lrs[:-1] + lrs[1:]) / 2
    return np.concatenate([[0.0], np.cumsum(avg_lr * dx)])


def main(output_path):
    runs = {}
    for name, path in RUNS.items():
        path = os.path.expandvars(path)
        print(f"Loading run {name!r} from {path}")
        lr_step, lr_val = load_scalar(path, 'learning_rate')
        zstd_step, zstd_val = load_scalar(path, 'embed_z_std')
        print(f"  LR: {len(lr_step):,} points (steps {lr_step[0]:,} - {lr_step[-1]:,})")
        print(f"  z_std: {len(zstd_step):,} points")
        runs[name] = {
            'lr_step': lr_step, 'lr_val': lr_val,
            'zstd_step': zstd_step, 'zstd_val': zstd_val,
        }

    fig, (ax_step, ax_cum) = plt.subplots(1, 2, figsize=(14, 5.5))

    for name, run in runs.items():
        cum = cumulative_lr(run['lr_step'], run['lr_val'])
        # Interpolate cumulative LR onto z_std's step grid (LR is logged every
        # step, z_std is logged every step too, but in practice the two series
        # may have slightly different step coverage due to logging cadence).
        zstd_cum = np.interp(run['zstd_step'], run['lr_step'], cum)
        c = COLORS.get(name)
        ax_step.plot(run['zstd_step'], run['zstd_val'], label=name, alpha=0.8, color=c, lw=1.0)
        ax_cum.plot(zstd_cum, run['zstd_val'], label=name, alpha=0.8, color=c, lw=1.0)

    ax_step.set_xlabel('step')
    ax_step.set_ylabel('embed_z_std')
    ax_step.set_title('z_std vs step\n(parameterized by training duration)')
    ax_step.legend(title='run')
    ax_step.grid(True, alpha=0.3)

    ax_cum.set_xlabel(r'$\sum$ lr $\cdot \Delta$step  (cumulative learning rate)')
    ax_cum.set_ylabel('embed_z_std')
    ax_cum.set_title('z_std vs cumulative LR\n(parameterized by integrated optimizer force)')
    ax_cum.legend(title='run')
    ax_cum.grid(True, alpha=0.3)

    fig.suptitle('ssl_57 d=512_L8: shared-attractor hypothesis test', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
