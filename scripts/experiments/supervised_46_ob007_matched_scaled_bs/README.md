# Experiment 46: exp 42 schedule + per-size batch scaling

Minimal delta from exp 42 (`supervised_42_ob007_matched_epoch_cap`): same OB007 pretrained encoder, same single-stage AdamW at `lr=3e-3` for all params, same 200-epoch / 400k-step budget. The *only* change is per-size batch sizing from exp 44.

## Why

Exps 43/44/45 introduced two things at once — gradual unfreeze (encoder LR capped at 3e-4) and per-size bs scaling. They fixed small-n overfitting but regressed large-n top-1 accuracy vs exp 42, because the encoder never trains at exp 42's 3e-3.

This experiment isolates the bs-scaling change. At `n ≥ bs_cap × bs_k = 32k`, `train_bs = 4096` (cap) and the run is identical to exp 42. At smaller n, `train_bs = max(64, n // 8)` — real SGD noise.

| n_train | train_bs |
|--------:|---------:|
| 100     | 64 |
| 500     | 64 |
| 1,000   | 125 |
| 8,000   | 1,000 |
| 32,000  | 4,000 |
| ≥ 64,000 | 4,096 |

## Hypothesis

If exp 46 matches exp 42 at large n (where they're mechanically identical) *and* beats exp 42 at small n (where bs scaling kicks in), then bs-scaling alone is the sufficient fix — the gradual-unfreeze structure in 43/44/45 was unnecessary complexity.

## Submission

```bash
bash run.sh scripts/experiments/supervised_46_ob007_matched_scaled_bs
```
