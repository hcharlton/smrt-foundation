# Experiment 40: Supervised baseline v2 (LR scheduler fix)

Clean rewrite of exp 31. Same model (`DirectClassifier d=128 n=4 h=4 ctx=32`), data (`cpg_pos/neg_v2`), optimizer (`AdamW lr=3e-3 wd=0.02`), normalization (`KineticsNorm log_transform=True`), seed (42), metrics, and per-epoch checkpointing.

## Why

Exp 31's LR scheduler has a bug: `accelerator.prepare(scheduler)` wraps the `LambdaLR` in `AcceleratedScheduler`, which modifies the stepping behaviour and causes `pct_start: 0.1` to not produce a peak at 10% of training steps. This version creates the scheduler directly on the prepared optimizer and steps it manually, matching the correct pattern used in exp 39.

The code is also restructured into smaller functions (`load_config`, `build_data`, `build_model`, `evaluate`, `save_checkpoint`) to improve readability.

## Submission

```bash
bash run.sh scripts/experiments/supervised_40_baseline_v2
```

## Expected outcome

Same ~82% top-1 as exp 31, with the LR schedule now peaking at 10% of total steps as configured.
