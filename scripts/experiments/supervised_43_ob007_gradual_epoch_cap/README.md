# Experiment 43: OB007 gradual unfreeze with epoch-capped step budget

Rerun of exp 38 with epoch-capped step budgets. Two-stage gradual unfreezing with proportional scaling: 25% frozen (stage 1), then gradual encoder warmup over 1/3 of stage 2.

## Why

Exp 38's fixed 100k/300k stage split meant n=100 spent 100k steps in stage 1 (100k epochs frozen) and 300k in stage 2. With epoch capping, the same 25%/75% proportions apply to the per-size budget, so n=100 gets 2,500 frozen steps and 7,500 unfrozen steps.

## Submission

```bash
bash run.sh scripts/experiments/supervised_43_ob007_gradual_epoch_cap
```
