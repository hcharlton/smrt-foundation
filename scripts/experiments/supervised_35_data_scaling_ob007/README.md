# Experiment 35: Data Scaling with OB007 Pretrained Encoder

## Objective

Measure how CpG classification performance scales with training data when using a pretrained encoder (exp 29 ssl_29_large_pretrain, trained on 25G of OB007 data, 63% probe top-1) vs training from scratch (exp 33). Same grid, same output format — the two experiments produce directly comparable scaling curves.

## Design

- **Pretrained checkpoint**: `scripts/experiments/ssl_29_large_pretrain/checkpoints/final_model.pt`
- **Pretraining details**: SmrtAutoencoder, d=128, L=4, H=4, ctx=128, 3000 epochs on 25G OB007 subset
- **Training sizes**: Same 16 sizes as exp 33 (100 to 8M)
- **Steps per size**: 400k (same as exp 33)
- **Two-stage fine-tuning** (exp 27 style):
  - Stage 1 (steps 1-100k): Frozen encoder, head-only at lr=3e-3
  - Stage 2 (steps 100k-400k): Unfrozen, encoder lr=3e-4, head lr=3e-3
- **Eval schedule**: 20 log-spaced points from step 100 to 400k
- **Validation**: First 1M samples (deterministic, no shuffle)
- **GPUs**: 8 (parallelised, 2 sizes per GPU)

**Note**: PE buffer mismatch (ctx=128 pretrained → ctx=32 classifier) is handled automatically — PE is skipped during loading and randomly initialized.

## Comparison with Exp 33

| | Exp 33 (scratch) | Exp 35 (OB007 pretrained) |
|--|---|---|
| Encoder init | Random (seed 42) | Exp 29 autoencoder (25G OB007, 63% probe) |
| Optimizer | AdamW lr=3e-3 all params | Stage 1: head lr=3e-3; Stage 2: encoder 3e-4, head 3e-3 |
| Schedule | 1 cosine over 400k steps | 2 cosines (100k + 300k) |
| Everything else | Identical | Identical |

## Outputs

### CSV (`n{size}/results.csv` and merged `results.csv`)

Same as exp 33 plus a `stage` column (1 or 2):

| Column | Description |
|--------|-------------|
| train_size | Number of training samples |
| eval_point | Evaluation checkpoint (1-20) |
| step | Optimizer step at this eval |
| stage | Training stage (1=frozen, 2=unfrozen) |
| train_loss | Average training loss since last eval |
| val_loss | Average validation loss |
| val_f1, val_auroc, val_auprc, val_accuracy | Eval metrics |
| epochs_completed | Full passes through training data |

### Checkpoints (`n{size}/step{N}.pt`)

Same format as exp 33, plus `stage` field.

## Submission

```bash
bash run.sh scripts/experiments/supervised_35_data_scaling_ob007
```
