# Experiment 41: Baseline data scaling with epoch-capped step budget

Rerun of exp 33/39 (DirectClassifier from scratch) with per-size step budgets instead of a fixed 400k steps. Same model, data, optimizer, and metrics.

## Why

Exp 33 used 400k steps for all dataset sizes, causing n=100 to train for 400k epochs and n=500 for ~400k epochs. These runs showed catastrophic overfitting: the model memorized the data within the first few hundred steps, then the remaining 399k+ steps drove it through mode collapse (val_accuracy = 0.5). The LR warmup (40k steps) was also misaligned — the model was fully memorized long before the LR peaked.

## Step budget formula

```
steps_per_epoch = ceil(dataset_size / batch_size)
total_steps = min(max(max_epochs * steps_per_epoch, min_steps), max_steps)
```

With `max_epochs=200`, `min_steps=10,000`, `max_steps=400,000`. The LR schedule (pct_start=0.1) scales proportionally so warmup is always 10% of the per-size total.

## Submission

```bash
bash run.sh scripts/experiments/supervised_41_baseline_epoch_cap
```
