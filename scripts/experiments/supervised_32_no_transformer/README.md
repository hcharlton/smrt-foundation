# Experiment 32: Transformer ablation (CNN-only)

One-variable ablation of exp 31 (`supervised_31_baseline_clean`). Uses `DirectClassifierNoTransformer` — `SmrtEmbedding` + the default `CNN` (11 ResBlocks, RF=107, 4x downsample) + the same classification head, **with no PositionalEncoding and no TransformerBlocks**. Everything else — data (`cpg_pos_v2` / `cpg_neg_v2` train/val), `KineticsNorm(log_transform=True)`, `AdamW(lr=3e-3, wd=0.02)`, cosine schedule with `pct_start=0.1`, 20 epochs at `batch_size=512`, `bf16` mixed precision, `set_seed(42)`, all four eval metrics, per-epoch checkpointing — is byte-for-byte identical to exp 31.

## Hypothesis

At ctx=32 the default CNN's receptive field (107) already exceeds the context. Every CNN output latent is a function of the entire 32-base input, so the transformer's bidirectional attention has less to contribute than it would at longer contexts: its job is just to re-mix latents that each already see everything. This experiment measures how much of exp 31's ~82% top-1 actually depends on that re-mixing.

## Comparison thresholds

| Exp 32 final top-1 | Interpretation |
|---|---|
| >= 80% | Transformer contributes <= 2pp at this context. It's cosmetic at ctx=32 and SSL pretraining efforts that rely on the transformer for downstream performance should be re-scoped toward longer contexts where the CNN can't cover the full input on its own. |
| 75-80% | Transformer contributes 2-7pp. Real but modest — worth keeping in the default architecture but not worth heavy optimization. |
| <= 75% | Transformer contributes >= 7pp even when the CNN already sees the full input. The "CNN covers everything" heuristic is wrong, probably because a single CNN latent can't cleanly integrate both left and right context at the center position. Keep the transformer. |

## What's the same as exp 31

- `cpg_pos_v2` / `cpg_neg_v2` train/val memmaps
- `KineticsNorm(log_transform=True)` — stats computed from ~2M sample cap, persisted in every epoch checkpoint via `save_stats()`
- `AdamW(lr=3e-3, weight_decay=0.02)` with `get_cosine_schedule_with_warmup(pct_start=0.1)`
- `BCEWithLogitsLoss`, all four eval metrics (`BinaryF1Score`, `BinaryAUROC`, `BinaryAveragePrecision`, `BinaryAccuracy`)
- `LabeledMemmapDataset` with `balance=True` on train, default balance on val
- 20 epochs, `batch_size=512`, `ds_limit=0` (full dataset)
- `bf16` mixed precision, `find_unused_parameters=True` on DDP, `set_seed(42)`
- Per-epoch `checkpoints/epoch_XX.pt` writes with `wait_for_everyone()` sync + try/except
- CNN receptive field logged at startup, config-key assertions, `checkpoint_dir` derived from `__file__`

## What's different from exp 31

Exactly one thing: the model class. Exp 31 uses `DirectClassifier` (`SmrtEmbedding -> CNN -> PE -> 4x TransformerBlock -> head`, ~2.2M params). Exp 32 uses `DirectClassifierNoTransformer` (`SmrtEmbedding -> CNN -> head`, ~1.2M params — roughly 45% smaller).

Two follow-on consequences of the model swap:
- The `classifier:` config block drops `n_head` and `n_layers` (no transformer to configure). Same cleanup principle as exp 20's dead `vocab_size` key.
- `save_epoch_checkpoint` drops the `encoder_state_dict` key from the saved dict. Exp 31 writes both `model_state_dict` and `encoder_state_dict` because `DirectClassifier.encoder` is the reusable backbone for SSL fine-tuning, but `DirectClassifierNoTransformer` has no `.encoder` attribute — the submodules sit directly on the module. `model_state_dict` remains the full backup.

## Checkpoint layout

Each `checkpoints/epoch_XX.pt` contains:

| Key | Value |
|---|---|
| `model_state_dict` | full `DirectClassifierNoTransformer` weights. Keys start with `embed.*`, `cnn.*`, `head.*` — no `encoder.*` prefix. |
| `config` | merged config dict (yaml + DEFAULT + git_hash) |
| `epoch` | 1-indexed epoch number |
| `metrics` | `{train_loss, eval_top1, eval_f1, eval_auroc, eval_auprc}` for this epoch |
| `norm_means` | training-time `KineticsNorm.means` tensor (shape `(4,)`) |
| `norm_stds` | training-time `KineticsNorm.stds` tensor (shape `(4,)`) |
| `norm_log_transform` | bool flag (always `True` for this script) |

**Not present** (intentionally): `encoder_state_dict`. This model has no `.encoder` attribute, so there is no reusable backbone to save under that key. Inference / fine-tuning code that wants the weights should read `model_state_dict`.

State dicts are NOT interchangeable with exp 20 / 28 / 31 checkpoints. The key prefixes differ (`embed.*` vs `encoder.embed.*`), so cross-loading would silently drop every parameter.

## Inference

Same pattern as exp 31, except the model class is different and there's no `encoder_state_dict` to read:

```python
import torch
from smrt_foundation.model import DirectClassifierNoTransformer
from smrt_foundation.normalization import KineticsNorm

ckpt = torch.load('scripts/experiments/supervised_32_no_transformer/checkpoints/epoch_20.pt',
                  map_location='cpu')
c = ckpt['config']['classifier']

model = DirectClassifierNoTransformer(d_model=c['d_model'], max_len=c['context'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

norm_fn = KineticsNorm.load_stats(ckpt)

with torch.no_grad():
    logits = model(norm_fn(x))
```
