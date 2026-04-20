# Experiment 45: OB007 gradual unfreeze with global small batch

Rerun of exp 43 with a fixed `batch_size=64` across every training-dataset size. Same structural fix as exp 44 (restore SGD noise at small n) but applied uniformly: all sizes share the same optimisation regime.

## Why

Exp 43 used `bs=4096`; at `n_train ≤ bs` every batch was the full dataset with replacement. Option A (exp 44) scales `bs` per size and preserves exp 43's throughput at the large end. Option B (here) commits to small-batch training everywhere — simpler and strictly comparable across sizes.

`bs=64` is below the smallest `train_size=100`, so every batch is a strict random subset at every size.

## LR scaling

Under Adam, rule-of-thumb `lr ∝ sqrt(bs)` gives a factor of `sqrt(64/4096) = 1/8`:

| Param             | Exp 43   | Exp 45 |
|-------------------|----------|--------|
| `frozen_lr`       | 3e-3     | 4e-4   |
| `head_lr`         | 3e-3     | 4e-4   |
| `encoder_lr`      | 3e-4     | 4e-5   |
| `encoder_start_lr`| 1e-5     | 1e-6   |

Step budget (`min_steps=10k`, `max_steps=400k`, `max_epochs=200`) is unchanged; epochs per size shift because `steps_per_epoch = n_train / 64` is smaller.

## Submission

```bash
bash run.sh scripts/experiments/supervised_45_ob007_gradual_small_bs
```
