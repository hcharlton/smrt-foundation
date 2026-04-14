# Experiment 38: Data Scaling with OB007 Pretrained Encoder (Gradual Unfreeze)

## Objective

Same as exp 35 but with gradual encoder unfreezing in stage 2. Instead of jumping the encoder LR to 3e-4 immediately when unfrozen, it warms up from 1e-5 → 3e-4 over 100k steps. This protects pretrained features from being disrupted by large early gradients.

## Design

- **Pretrained checkpoint**: `scripts/experiments/ssl_29_large_pretrain/checkpoints/final_model.pt`
- **Stage 1** (steps 1–100k): Frozen encoder, head-only at lr=3e-3 (same as exp 35)
- **Stage 2** (steps 100k–400k):
  - Encoder: LR warmup 1e-5 → 3e-4 over 100k steps, then cosine decay
  - Head: standard cosine warmup to 3e-3 (pct_start=0.1 → 30k warmup), then cosine decay
  - Per-group schedules via `LambdaLR`
- **Eval schedule**: 40 log-spaced points (doubled from exp 35's 20)

## Comparison with exp 35

| | Exp 35 | Exp 38 |
|--|---|---|
| Stage 2 encoder LR | Jump to 3e-4 | 1e-5 → 3e-4 over 100k steps |
| Eval points | 20 (~4 in stage 2) | 40 (~7 in stage 2) |
| Everything else | Identical | Identical |

## Submission

```bash
bash run.sh scripts/experiments/supervised_38_ob007_gradual_unfreeze
```
