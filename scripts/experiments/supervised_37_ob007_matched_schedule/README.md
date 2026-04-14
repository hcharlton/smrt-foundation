# Experiment 37: Data Scaling with OB007 Pretrained Encoder (Matched Schedule)

## Objective

Same as exp 35 but with exp 33's exact training schedule. The only difference from exp 33 is encoder initialization — exp 29 OB007 pretrained (25G subset, 63% probe top-1) vs random. No two-stage fine-tuning, no differential LR. Cleanest comparison to isolate the OB007 pretraining effect.

## Design

- **Pretrained checkpoint**: `scripts/experiments/ssl_29_large_pretrain/checkpoints/final_model.pt`
- **Pretraining details**: SmrtAutoencoder, d=128, L=4, H=4, ctx=128, 3000 epochs on 25G OB007 subset
- **Training**: Single-stage, AdamW lr=3e-3 for all params, single cosine over 400k steps — identical to exp 33
- **PE mismatch**: ctx=128→32, PE buffer reinitialized (handled by load_pretrained_encoder)

## Comparison

| | Exp 33 (scratch) | Exp 35 (two-stage) | Exp 37 (matched) |
|--|---|---|---|
| Encoder init | Random | Exp 29 OB007 | Exp 29 OB007 |
| Optimizer | AdamW 3e-3 all | Frozen → differential LR | AdamW 3e-3 all |
| Schedule | 1 cosine 400k | 2 cosines (100k+300k) | 1 cosine 400k |
| Confounds vs 33 | — | Schedule + LR | None |

## Submission

```bash
bash run.sh scripts/experiments/supervised_37_ob007_matched_schedule
```
