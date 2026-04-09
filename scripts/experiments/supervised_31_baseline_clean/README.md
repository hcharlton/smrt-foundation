# Experiment 31: Supervised baseline (clean rerun of exp 20)

Clean rerun of `supervised_20_full_v2`. Functionally identical — same data (`cpg_pos_v2` / `cpg_neg_v2` train/val), same `DirectClassifier(d_model=128, n_layers=4, n_head=4, max_len=32)`, same `ZNorm(log_transform=True)`, same AdamW at `lr=3e-3 / wd=0.02`, same cosine schedule with `pct_start=0.1`, same 20 epochs at `batch_size=512`, same `bf16` mixed precision, same `set_seed(42)`. The final top-1 should land within noise of exp 20's ~82%.

## Why

Exp 20's `train.py` only writes a checkpoint after the full 20-epoch loop finishes. If the job hits SLURM walltime or crashes mid-run, nothing reaches disk. A previous exp 20 run produced no artifacts for exactly this reason. This version writes `checkpoints/epoch_XX.pt` after every completed epoch, so any progress that survives an epoch is preserved.

## Bugs fixed vs exp 20

1. **Artifacts lost on early termination** — save block moved inside the epoch loop.
2. **No sync before save** — `accelerator.wait_for_everyone()` added between eval and save, matching the pattern in `ssl_30_smallrf_autoencoder/train.py`.
3. **Silent save failures** — `torch.save` wrapped in try/except with contextual logging.
4. **Dead config keys removed** — `vocab_size: 5` and `optimizer: 'AdamW'` were never read by the train script (`DirectClassifier` doesn't take `vocab_size`, and AdamW is hardcoded).
5. **Missing CNN receptive field logging at startup** — added per project convention (every other recent train.py does this).
6. **Unvalidated config keys** — the four data-path keys are now explicitly asserted at startup instead of silently defaulting to `None`.
7. **Fragile checkpoint-dir derivation** — derived from `__file__` instead of `os.path.dirname(config_path)`, so caller cwd can't misplace the output.

## What NOT to change

Anything that would break functional identity with exp 20:
- `ZNorm` (not `KineticsNorm` — they compute statistics differently)
- `find_unused_parameters=True` on DDP kwargs
- All tensorboard logging keys (`train_loss`, `learning_rate`, `epoch`, `epoch_avg_loss`, `eval_f1`, `eval_auroc`, `eval_auprc`, `eval_top1`)
- `balance=True` on the train dataset only
