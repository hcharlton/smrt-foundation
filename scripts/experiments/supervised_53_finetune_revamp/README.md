# supervised_53_finetune_revamp

Fine-tune revamp on the ssl_58 autoencoder grid. Four parallel
treatments testing different hypotheses for why ssl_58 fine-tunes barely
beat the linear probe at scale.

## Problem statement

ssl_25 (d=128, ctx=32, autoencoder) → fine-tune lifted probe accuracy
from 66% to 79% (+13pp). ssl_58 fine-tunes (supervised_51_finetune_ssl58_grid)
show much smaller LP→FT gains. This is structurally suspicious — the
recipe is the same; only the encoder is bigger and trained on a longer
context. Four candidate explanations:

| Treatment | Hypothesis | Implementation |
|---|---|---|
| **midlayer** (F1) | The autoencoder loss makes the *top* transformer layer reconstruction-specialised at scale. Classification-relevant features migrate to middle layers. | `DirectClassifierMidLayer` reads from a configurable `layer_idx` (set per-arch from a layer-sweep probe). `forward_to_layer(x, layer_idx)` on `SmrtEncoder`. |
| **lpft_lldr** (F2) | The fine-tune recipe is wrong: top-layer is shocked by full-LR unfreeze, and uniform LR across layers is too aggressive on the bottom of the encoder. | LP-FT (frozen-encoder warmup on the head, then unfreeze) + layer-wise LR decay (top layer at max_lr, deeper layers exponentially smaller). |
| **decoder_init** (F3) | "Don't throw the projection head away": the SSL decoder upsample stack carries useful structure that random-init reinvents. | `DirectClassifierWithDecoder` re-uses the autoencoder's `SmrtDecoder.upsample` (two ConvTranspose1d + GELU) as a feature mixer between encoder and classifier head. |
| **recipe_match** (F4) | The classifier head is undersized for d=512/d=768, and the LR schedule may need exact supervised_20 matching. | `DirectClassifierBigHead` (3-layer MLP head keeping full width before projection). Recipe locked to supervised_20's. |

Each treatment is a separate `bash run.sh ...` job. Internally each
treatment fans out to 5 archs × 2 inits × 3 train_sizes = 30 combos
(with the random-init baseline included for clean A/Bs against the
ssl_58-init treatment).

## Best-checkpoint resolution

All ssl_58-init treatments use `checkpoint: 'auto_best'` instead of
`final_model.pt`. The `scripts/utils/select_best_ssl_checkpoint`
resolver reads `probe_history.csv` for the SSL experiment's TB run dir,
picks the step with the highest `probe_top1`, and returns that step's
`step_<N>.pt`. This matters most for the smaller arches in the grid,
whose probe trajectory often peaks well before the final step.

Each arch declares the *specific* SSL experiment directory it's
fine-tuning from via the init spec's `ssl_exp_dirs:` block (keyed by
arch_name). All five archs currently point at their `size_d*_L8_long`
continuations rather than the base `size_d*_L8` directories, so
fine-tune always reads from the longer-trained (post-cosine-tail)
representation. To target a different SSL run for a given arch, edit
the dict entry — no code change.

## Layout

```
supervised_53_finetune_revamp/
├── README.md
├── midlayer/{config.yaml, train.py}      # F1
├── lpft_lldr/{config.yaml, train.py}     # F2
├── decoder_init/{config.yaml, train.py}  # F3
└── recipe_match/{config.yaml, train.py}  # F4
```

Submit each with `bash run.sh scripts/experiments/supervised_53_finetune_revamp/<treatment>`.
All four can be queued at once; Slurm stages them.

## Compute

30 combos per treatment × 4 treatments = 120 combos total. ds_grid_v3
load-balances combos across the 8 GPUs of one node, so each treatment
is one 8-GPU job that internally cycles through ~4 combos per GPU.
Expected walltime: 24h per treatment job, 4 treatments × 8 GPUs × 24h
= 768 GPU-hours (the d=1024 combos are the long-pole on each GPU but
fit within the 24h budget).

## Reading the F1 layer_idx

For F1 the `architectures` entries each carry an explicit `layer_idx`
selected from the offline layer-sweep probe
(`scripts/utils/midlayer_probe_sweep.py`). Defaults assume the
sweep landed somewhere mid-stack; update before submitting the job
once `midlayer_probe_results.csv` is available.
