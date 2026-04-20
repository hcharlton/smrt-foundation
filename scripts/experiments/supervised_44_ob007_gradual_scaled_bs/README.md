# Experiment 44: OB007 gradual unfreeze with per-size batch scaling

Rerun of exp 43 with per-size batch sizing to address small-n overfitting. All other hyperparameters (LRs, schedule fractions, encoder warmup) match exp 43 so any delta is attributable to batch-size regime.

## Why

Exp 43 used `bs=4096` across all sizes, with a `min(bs, n_train)` clamp. At `n_train=100..1000`, every batch was essentially the full dataset resampled with replacement — no SGD noise, deterministic descent straight into memorisation. Batches smaller than `n_train` restore real stochastic subsampling.

Batch size per run: `min(bs_cap, max(bs_floor, n_train // bs_k))`.

With `bs_cap=4096`, `bs_floor=64`, `bs_k=8`:
| n_train | train_bs |
|---------|---------:|
| 100     | 64  |
| 500     | 64  |
| 1,000   | 125 |
| 2,000   | 250 |
| 4,000   | 500 |
| 8,000   | 1,000 |
| 16,000  | 2,000 |
| 32,000  | 4,000 |
| ≥ 64,000 | 4,096 (cap) |

Large sizes keep exp 43's throughput (same bs, same schedule, same LRs).

## Submission

```bash
bash run.sh scripts/experiments/supervised_44_ob007_gradual_scaled_bs
```
