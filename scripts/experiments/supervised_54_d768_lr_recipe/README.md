# supervised_54_d768_lr_recipe

Structured LR x recipe ablation on a single SSL artifact (`ssl_58_autoencoder_grid/size_d768_L8_long`) to diagnose why `supervised_53_finetune_revamp` fails to lift over the frozen-encoder linear probe at d=768.

## Why this exists

`supervised_53_finetune_revamp/midlayer` at d=768/L=8/n=100k peaks at val_acc=0.747 at the very first eval (step 100, epoch 4) and then degrades to 0.725 over 200 epochs. The pilot trajectory shows:

| step | epoch | train_loss | val_loss | val_acc | val_auroc |
|---|---|---|---|---|---|
| 100 | 4 | 0.499 | 0.529 | **0.747** | 0.830 |
| 1500 | 60 | 0.017 | 1.72 | 0.721 | 0.796 |
| 5000 | 200 | 2e-6 | **3.16** | 0.725 | 0.796 |

Train loss collapses 5 orders of magnitude; val loss rises 6x; AUROC stays 0.79-0.83. The encoder is still producing rank-informative features but the decision boundary is being driven into confident-wrong territory. The 0.747 peak is barely above the SSL linear probe number (0.71), so fine-tuning is not extracting anything beyond LP and is actively degrading from there.

Two suspects, each tied to a known recipe choice supervised_53 inherits unchanged from supervised_50:

- **Encoder LR**: supervised_53 uses uniform 3e-3 for the entire encoder. The only fine-tune recipe that produced a clean +13pp lift in this project (supervised_27 on ssl_25) used encoder_lr=3e-4 with head_lr=3e-3.
- **Recipe (batch size, epoch count)**: supervised_53 uses bs=4096 / 200 epochs. supervised_20 / supervised_27 used bs=512 / 20 epochs. At n=100k with bs=4096 there are 25 steps/epoch, so each label gets reinforced ~200 times across the run -- no SGD-noise regularization, every batch is nearly the full dataset.

## Design

Single SSL artifact (`size_d768_L8_long`, resolved via `auto_best`). Single arch (`d768_L8`). Single treatment (`midlayer`, matching the existing supervised_53/midlayer entry so cross-experiment trajectories line up directly).

| Cell | head_lr | encoder_lr | batch_size | max_epochs | treatment | Tests |
|------|---------|------------|------------|------------|-----------|-------|
| **A0** | 3e-3 | 3e-3 | 4096 | 200 | midlayer | Reproduces existing supervised_53/midlayer (control). |
| **A1** | 3e-3 | 3e-4 | 4096 | 200 | midlayer | LR fix alone (supervised_27 ratio). |
| **A2** | 3e-3 | -- (frozen) | 4096 | 200 | linear_probe | In-harness anchor for the 0.71 LP number. |
| **B0** | 3e-3 | 3e-3 | 512 | 20 | midlayer | Recipe fix alone (current LR, sane batch / epochs). |
| **B1** | 3e-3 | 3e-4 | 512 | 20 | midlayer | Both fixes (sane LR + sane recipe). |

At every cell: 2 inits (`random`, `ssl_58_best`) x 3 train_sizes (`10000`, `100000`, `8000000`).
Total: 5 cells x 2 inits x 3 sizes = **30 runs across 5 sbatch jobs**.

The 5 cells fill a 2-LR x 2-recipe factorial plus the LP anchor:

| | current recipe (bs=4096, 200ep) | sane recipe (bs=512, 20ep) |
|---|---|---|
| current_lr (enc=3e-3) | **A0** | **B0** |
| sane_lr (enc=3e-4) | **A1** | **B1** |
| frozen LP (head only) | **A2** | (not run) |

## How to run

Phase A (independent submissions, run in parallel):

```bash
bash run.sh scripts/experiments/supervised_54_d768_lr_recipe/phase_a_lr_ablation/current_lr
bash run.sh scripts/experiments/supervised_54_d768_lr_recipe/phase_a_lr_ablation/sane_lr
bash run.sh scripts/experiments/supervised_54_d768_lr_recipe/phase_a_lr_ablation/frozen
```

Phase B (after Phase A finishes -- two independent submissions):

```bash
bash run.sh scripts/experiments/supervised_54_d768_lr_recipe/phase_b_recipe_ablation/current_lr
bash run.sh scripts/experiments/supervised_54_d768_lr_recipe/phase_b_recipe_ablation/sane_lr
```

## Harness dependencies

This study relies on two additions to `scripts/ds_grid_v3.py` introduced for supervised_54:

- `treatment: 'linear_probe'` -- encoder frozen for the whole run, optimizer only iterates over head parameters. Used by A2.
- `classifier.head_lr` / `classifier.encoder_lr` -- when both are set and differ, the non-LLDR optimizer builds two parameter groups (head and encoder) with a single shared cosine schedule. Used by A1 and B1. When they match (A0, B0) the old single-group codepath is preserved.

Both are gated by config keys; existing supervised_53 configs (which do not set these fields) continue to work unchanged. Tests in `tests/test_ds_grid_v3_lr_treatments.py`.

## Report

Analysis lives in `report/ft_ablations/supervised_54_d768_lr_recipe/`. After the runs finish:

```bash
# Unbiased val3 evaluation per phase (picks each cell's best-val_acc checkpoint).
python -m scripts.ft_eval evaluate \
  --ft_exp_dir scripts/experiments/supervised_54_d768_lr_recipe/phase_a_lr_ablation \
  --recipes current_lr,sane_lr,frozen \
  --out report/ft_ablations/supervised_54_d768_lr_recipe/val3_eval_phase_a.csv

python -m scripts.ft_eval evaluate \
  --ft_exp_dir scripts/experiments/supervised_54_d768_lr_recipe/phase_b_recipe_ablation \
  --recipes current_lr,sane_lr \
  --out report/ft_ablations/supervised_54_d768_lr_recipe/val3_eval_phase_b.csv

# Stitch all per-cell results.csv plus val3 eval into one CSV with phase / lr_strategy / recipe columns.
python report/ft_ablations/supervised_54_d768_lr_recipe/aggregate.py
```
