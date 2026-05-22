# supervised_54 d=768 LR x recipe ablation -- report

Analysis of `scripts/experiments/supervised_54_d768_lr_recipe`. Tests whether the absent LP -> FT lift at d=768 in supervised_53 is caused by an over-aggressive encoder LR (3e-3 uniform), the high-bs / 200-epoch recipe, or a true representation ceiling at this model size.

Anchor SSL artifact: `ssl_58_autoencoder_grid/size_d768_L8_long` (auto_best probe milestone).

## Cells

| Cell | head_lr | encoder_lr | batch_size | max_epochs | treatment |
|------|---------|------------|------------|------------|-----------|
| A0   | 3e-3 | 3e-3 | 4096 | 200 | midlayer (control) |
| A1   | 3e-3 | 3e-4 | 4096 | 200 | midlayer |
| A2   | 3e-3 | -- (frozen) | 4096 | 200 | linear_probe |
| B0   | 3e-3 | 3e-3 | 512  | 20  | midlayer |
| B1   | 3e-3 | 3e-4 | 512  | 20  | midlayer (both fixes) |

## How to refresh after a run completes

```bash
# Unbiased val3 eval per phase (picks each cell's best-val_acc checkpoint).
python -m scripts.ft_eval evaluate \
  --ft_exp_dir scripts/experiments/supervised_54_d768_lr_recipe/phase_a_lr_ablation \
  --recipes current_lr,sane_lr,frozen \
  --out report/ft_ablations/supervised_54_d768_lr_recipe/val3_eval_phase_a.csv

python -m scripts.ft_eval evaluate \
  --ft_exp_dir scripts/experiments/supervised_54_d768_lr_recipe/phase_b_recipe_ablation \
  --recipes current_lr,sane_lr \
  --out report/ft_ablations/supervised_54_d768_lr_recipe/val3_eval_phase_b.csv

# Stitch every per-leaf results.csv plus the two val3 outputs into one combined frame.
python report/ft_ablations/supervised_54_d768_lr_recipe/aggregate.py
```

`combined_results.csv` is the single source for downstream plots. Columns:

- `phase` (`phase_a` / `phase_b`), `cell` (`current_lr` / `sane_lr` / `frozen`)
- `head_lr`, `encoder_lr`, `train_bs`, `max_epochs` (recipe identifiers)
- `init_name` (`random` / `ssl_58_best`), `arch_name`, `train_size`
- `eval_point`, `step`, `epochs_completed`
- `train_loss`, `val_loss`, `val_f1`, `val_auroc`, `val_auprc`, `val_accuracy`
- `treatment` (`midlayer` / `linear_probe`)
- `test_top1`, `test_auroc`, `test_auprc`, `test_f1`, `test_loss` -- val3 numbers from `ft_eval` joined onto the best step per (cell, init, size).

## Acceptance criteria

1. **A2 (ssl_58_best, n=8M) val_accuracy approx 0.71** -- harness reproduces the externally-computed SSL probe number.
2. **A0 reproduces supervised_53/midlayer's trajectory** (peak val_acc approx 0.74 at step 100, decay to approx 0.72 by step 5000).
3. **A1 (sane LR) val_loss minimum past step 100; final val_acc > A0's peak** -- LR was a real bottleneck.
4. **B1 (sane LR + sane recipe) val_acc at n=100k > A0's peak (0.747)** -- both fixes together close the gap. If lift >= +5pp over A2 the recipe + LR explain the missing lift; if B1 still tops out at approx 0.75 the representation is the bottleneck.

## Plots

- `plots/val_acc_trajectories.png` -- val_acc vs step, faceted by (train_size, init), one line per cell.
- `plots/peak_val_acc_grid.png` -- best val_acc heatmap across the 5x2x3 cell x init x size grid.
- `plots/lp_to_ft_lift.png` -- `(best val_acc) - (A2 LP val_acc)` per (LR, recipe) cell x init x train_size.
- `plots/train_val_divergence.png` -- train_loss vs val_loss over steps to visualise the overfit pathology (compare A0 vs B1 trajectories side by side).

Each plot is produced by a small notebook / script reading `combined_results.csv`; add them once the runs land.
