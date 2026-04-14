# Experiment 36: Data Scaling with Pretrained Encoder (Matched Schedule)

## Objective

Same as exp 34 but with exp 33's exact training schedule. The only difference from exp 33 is encoder initialization — pretrained (exp 25) vs random. No two-stage fine-tuning, no differential LR. This is the cleanest possible comparison to isolate the effect of pretraining.

## Design

- **Pretrained checkpoint**: `scripts/experiments/ssl_25_cpg_autoencoder/checkpoints/final_model.pt`
- **Training**: Single-stage, AdamW lr=3e-3 for all params, single cosine over 400k steps — identical to exp 33
- **Training sizes**: Same 16 sizes as exp 33 (100 to 8M)
- **Eval schedule**: 20 log-spaced points from step 100 to 400k
- **CSV format**: Identical to exp 33 (no `stage` column)

## Comparison

| | Exp 33 (scratch) | Exp 34 (two-stage) | Exp 36 (matched) |
|--|---|---|---|
| Encoder init | Random | Exp 25 | Exp 25 |
| Optimizer | AdamW 3e-3 all | Frozen → differential LR | AdamW 3e-3 all |
| Schedule | 1 cosine 400k | 2 cosines (100k+300k) | 1 cosine 400k |
| Confounds vs 33 | — | Schedule + LR | None |

## Submission

```bash
bash run.sh scripts/experiments/supervised_36_pretrained_matched_schedule
```
