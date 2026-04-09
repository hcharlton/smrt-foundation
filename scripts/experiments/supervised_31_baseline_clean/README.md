# Experiment 31: Supervised baseline (clean rerun of exp 20)

Clean rerun of `supervised_20_full_v2`. Functionally identical on the CpG data distribution — same data (`cpg_pos_v2` / `cpg_neg_v2` train/val), same `DirectClassifier(d_model=128, n_layers=4, n_head=4, max_len=32)`, same log1p + z-score normalization, same AdamW at `lr=3e-3 / wd=0.02`, same cosine schedule with `pct_start=0.1`, same 20 epochs at `batch_size=512`, same `bf16` mixed precision, same `set_seed(42)`. The final top-1 should land within noise of exp 20's ~82%.

The normalization class is `KineticsNorm` rather than exp 20's `ZNorm`. On CpG windows (context=32, no padding) the two are equivalent — `KineticsNorm`'s padding-exclusion branch is a no-op when `x[..., -1]` is all zero, and both apply the same `(log1p - mean) / std` transform on channels [1, 2]. The swap means saved stats can be loaded at inference via `KineticsNorm.from_stats(...)` with no cross-class translation.

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
8. **Normalization stats not persisted** — training-time `KineticsNorm` means/stds are now saved in every epoch checkpoint so inference code can reconstruct the exact transform via `KineticsNorm.from_stats(...)` instead of re-sampling (which would add sampling noise and also require the training data on disk).

## What NOT to change

Anything that would break functional identity with exp 20:
- `log_transform=True` on the normalization (matches exp 20's `ZNorm(log_transform=True)`; `KineticsNorm` applies the same log1p + z-score on channels [1, 2])
- `find_unused_parameters=True` on DDP kwargs
- All tensorboard logging keys (`train_loss`, `learning_rate`, `epoch`, `epoch_avg_loss`, `eval_f1`, `eval_auroc`, `eval_auprc`, `eval_top1`)
- `balance=True` on the train dataset only

If exp 31's final top-1 lands materially below exp 20's ~82%, the first suspect should be anything other than normalization — on CpG data the `ZNorm` → `KineticsNorm` swap is provably a no-op (both classes compute `(log1p(x) - mean) / std` over the same positions when the pad mask is all zero).

## Checkpoint contents and inference loading

Each `checkpoints/epoch_XX.pt` contains:

| Key | Value |
|---|---|
| `model_state_dict` | full `DirectClassifier` weights |
| `encoder_state_dict` | just the `SmrtEncoder` weights, for fine-tune reuse |
| `config` | the merged config dict (yaml + DEFAULT + git_hash) |
| `epoch` | 1-indexed epoch number |
| `metrics` | `{train_loss, eval_top1, eval_f1, eval_auroc, eval_auprc}` for this epoch |
| `norm_means` | training-time `KineticsNorm.means` tensor (shape `(4,)`) |
| `norm_stds` | training-time `KineticsNorm.stds` tensor (shape `(4,)`) |
| `norm_log_transform` | bool flag (always `True` for this script) |

At inference time, reconstruct the exact training-time normalization via `KineticsNorm.load_stats(ckpt)` — no need to re-sample statistics from the training data (and no sampling noise):

```python
import torch
from smrt_foundation.model import DirectClassifier
from smrt_foundation.normalization import KineticsNorm

ckpt = torch.load('scripts/experiments/supervised_31_baseline_clean/checkpoints/epoch_20.pt',
                  map_location='cpu')
c = ckpt['config']['classifier']

model = DirectClassifier(
    d_model=c['d_model'], n_layers=c['n_layers'],
    n_head=c['n_head'], max_len=c['context'],
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

norm_fn = KineticsNorm.load_stats(ckpt)

# x is a (B, context, 4) tensor in the raw memmap layout
with torch.no_grad():
    logits = model(norm_fn(x))
```

On the training side, the same symmetry: `torch.save({..., **norm_fn.save_stats()}, path)` merges the three `norm_*` keys into the checkpoint dict without having to reimplement the detach/cpu/key-naming boilerplate in every experiment's train.py.

The stats loaded via `from_stats` are exactly the ones the model saw during training (modulo float32 precision), so inference normalization is bit-for-bit reproducible — no sampling noise from re-deriving stats. For any other data distribution — do NOT recompute stats from the new data; always use the training-time stats via `from_stats`, otherwise you normalize out the exact distribution shift the model needs to see.
