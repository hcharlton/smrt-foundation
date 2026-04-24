# Experiment 49: exp 48 + per-size schedule overrides

Same encoder / architecture / data / large-n schedule as exp 48. The only changes are on the small-n end of the grid, where exp 48's val-loss trajectory was pathological.

## What's different from exp 48

1. `scaling.min_steps` dropped from **10000 → 100**. In exp 48 this was a silent override: every size where `max_epochs × steps_per_epoch` was less than 10000 (i.e. every n ≤ 32000) trained for 10000 steps regardless of the nominal `max_epochs: 200` cap. At n=8000, bs=1000 that meant 1250 epochs actual vs. 200 epochs advertised.
2. A new `size_overrides:` block provides per-size `max_lr` and `max_epochs` for the n ∈ {100, 500, 1000, 2000, 4000} bucket, where lr=3e-3 saturates val_loss within a few hundred steps.

```yaml
size_overrides:
  100:   {max_lr: 3e-4, max_epochs: 50}
  500:   {max_lr: 3e-4, max_epochs: 60}
  1000:  {max_lr: 5e-4, max_epochs: 80}
  2000:  {max_lr: 1e-3, max_epochs: 100}
  4000:  {max_lr: 1e-3, max_epochs: 150}
```

Sizes n ≥ 8000 keep the defaults (`max_lr=3e-3`, `max_epochs=200`) unchanged — at those sizes the 200-epoch cap already fires before `min_steps` would have clamped it in exp 48, so those runs are mechanically identical.

## Resulting step budget

| n_train | train_bs | total_steps | epochs | vs exp 48 |
|-------:|---------:|-----------:|------:|:---------:|
| 100    | 64       | 100        | 50    | 10000 → 100 |
| 500    | 64       | 480        | 60    | 10000 → 480 |
| 1000   | 125      | 640        | 80    | 10000 → 640 |
| 2000   | 250      | 800        | 100   | 10000 → 800 |
| 4000   | 500      | 1200       | 150   | 10000 → 1200 |
| 8000   | 1000     | 1600       | 200   | 10000 → 1600 |
| 16000  | 2000     | 1600       | 200   | 10000 → 1600 |
| 32000  | 4000     | 1600       | 200   | 10000 → 1600 |
| 64000+ | 4096     | unchanged  | 200   | unchanged |

## Hypothesis

If shifting the step budget and lowering small-n lr does what I expect, the val_loss minimum per size should land near 40–60% of the run (exp 48: ~3%), and peak val_f1 at small n should be at least as high as exp 48 since the optimizer won't overshoot. If small-n val_f1 actually drops, the overrides are too aggressive and the next pass should lengthen `max_epochs` rather than raise `max_lr`.

## Caveat (per-size keying)

`size_overrides` keys match the *requested* `train_size` from `scaling.train_sizes`, not the post-balance `n_train` that the trainer gets after `LabeledMemmapDataset(..., balance=True)` trims to the min class count. For the CpG dataset these are close, but if you ever see log lines where `Train:` reports a different count than the size you overrode on, the override still applies — the resolver keys on the requested size.

## Submission

```bash
bash run.sh scripts/experiments/supervised_49_simclr_r2_d256_L8_persize
```
