# supervised_51_finetune_ssl58_grid

First fine-tune evaluation of the ssl_58 autoencoder grid. The linear
probe at random init plus 0 SSL training gives ~0.62 on CpG; ssl_58 at
d=128 already pushed that to ~0.67 by step 10k of pretraining. This
experiment tests whether the encoder gain translates to fine-tune
accuracy over random init at matching architecture.

## What's running

A 2 x 4 x 3 = 24-run matrix:

- **Inits**: `random`, `ssl_58_full` (the `final_model.pt` from each
  ssl_58 size's `checkpoints/` dir)
- **Architectures**: `d128_L4`, `d256_L8`, `d512_L8`, `d768_L8`,
  matching ssl_58's per-size shapes exactly (head_dim=64 throughout)
- **Train sizes**: 10000, 100000, 8000000

The protocol mirrors `supervised_50_finetune_grid_v2`: single-stage
AdamW, max_lr=3e-3, weight_decay=0.02, pct_start=0.1, bs=4096 (capped
at n_train), max_epochs=200 with `min_steps=100` so small-n runs are
governed by max_epochs (avoiding the supervised_49 silent-overtraining
trap).

## Why architectures match ssl_58 instead of supervised_50

ssl_58 sizes use head_dim=64 across all four (d=128/n_head=2,
d=256/n_head=4, d=512/n_head=8, d=768/n_head=12). supervised_50 used
n_head=4 at d=128 (head_dim=32). Loading an ssl_58 d=128 encoder
state-dict into a d=128/n_head=4 classifier fails because the
attention QKV projections have incompatible shapes (the encoder ships
2-headed projection matrices; the fresh classifier wants 4-headed).

We could not have it both ways. Matching ssl_58 means:

- Random-init baseline at d=128 is the right comparator for
  ssl_58_d128 (same architecture).
- Cross-experiment continuity at d=128 against supervised_50's
  random-init baseline is broken (different head count). Sizes
  d=256/d=512/d=768 are continuous across both experiments.

## Why two inits and not three

The current goal is "does ssl_58 beat random init at this
architecture?". Adding ssl_57 here would re-run the supervised_50
question with a different head count at d=128, doubling the run count
without resolving the ssl_58 question. A separate cross-SSL-family
matrix is the cleaner follow-up if the ssl_58 vs random comparison
shows a meaningful gap.

## The cnn_variant switch

ssl_58 uses `SmrtEncoderSmallRF` (CNN RF=27, 4 ResBlocks). The default
`SmrtEncoder` (CNN RF=107, 11 ResBlocks) has incompatible
`encoder.cnn.*` state-dict keys, so trying to load an ssl_58 encoder
into a default `DirectClassifier` partially-fails and silently leaves
half the CNN at random init.

To match, this experiment sets `classifier.cnn_variant: 'small_rf'`
in the config. `ds_grid_v2.py` reads that knob and instantiates
`DirectClassifierSmallRF` (`smrt_foundation/model.py`) instead of
`DirectClassifier`. The two classes share head, forward, and
encoder.{embed,pe,blocks,layer_norm_target}; only the encoder's CNN
submodule differs. State-dict load through the existing
`load_pretrained_encoder` (`scripts/ds_grid.py:196`) is encoder-class-
agnostic and works for both.

## Pass criterion framing

Per-size: `ssl_58_full` should beat `random` at each train_size, with
the gap larger at small n_train than at large n_train (the canonical
SSL-pretraining shape). Across architectures: cross-arch slope tells
us whether scaling the autoencoder helps. A flat or inverted slope
isn't a fail per se but is informative.

## Layout

- `config.yaml` declares the matrix.
- `train.py` is a 12-line wrapper delegating to
  `scripts/ds_grid_v2.py:main`.
- After launch the experiment dir gets `random/<arch>/n<size>/` and
  `ssl_58_full/<arch>/n<size>/` subdirs, each with per-step `step<N>.pt`
  checkpoints, a per-run `results.csv`, and TB logs under
  `training_logs/<init>/<arch>/n<size>/`. A merged `results.csv` at
  the experiment root carries all 24 runs.

## Submission

Single submission for the whole grid; ds_grid_v2 manages dispatch
across the 8 GPUs internally via `mp.spawn` with FLOPS-weighted
greedy assignment:

```
bash run.sh scripts/experiments/supervised_51_finetune_ssl58_grid
```

Gated on ssl_58 emitting `final_model.pt` per size. Until then,
dry-run combo expansion still works:

```
DS_GRID_DRY_RUN=1 DS_GRID_FORCE_WORLD_SIZE=8 \
  python scripts/experiments/supervised_51_finetune_ssl58_grid/train.py \
  scripts/experiments/supervised_51_finetune_ssl58_grid/config.yaml
```

Missing-checkpoint warnings under DS_GRID_DRY_RUN are expected and do
not abort.
