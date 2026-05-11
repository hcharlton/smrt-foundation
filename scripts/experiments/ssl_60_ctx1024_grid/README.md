# ssl_60_ctx1024_grid

Context-axis ablation on the ssl_58 autoencoder lineage. Holds model
size and every other hyperparameter fixed; sweeps only the model's
trained context window (512 → 1024) across the same four sizes as
ssl_58.

## Why a separate experiment number

ssl_58 sweeps capacity at fixed ctx=512. The earlier `_ctx4096`
variants nested inside ssl_58 conflated the two axes; that didn't
matter while ctx=4096 was dormant. With a renewed context sweep at
ctx=1024 — the natural smaller-jump experiment after ctx=4096 turned
out to be too big — the cleaner separation is one experiment number per
scientific question. ssl_58 owns the size axis at ctx=512; ssl_60 owns
the context axis at the sizes ssl_58 established.

## What is fixed vs swept

Fixed (matches the `ssl_58_autoencoder_grid/size_d*_L8_long`
baselines byte-identically):
- Model class: `SmrtAutoencoderSmallRF` (r0=27, 4x CNN downsample).
- Loss: `MaskedReconstructionLoss` on kinetics at masked positions.
- Masking: `p_mask=0.15`, `mask_size=10`.
- Optimizer: AdamW, `max_lr=3e-4`, `weight_decay=0.02`,
  `pct_start=0.03`, `grad_clip=5.0`.
- Schedule: `schedule_steps=1_000_000` cosine warmup+decay,
  `total_steps=10_000_000` so walltime is the binding exit.
- Per-arch batch size: same as `_long` (bs=512 / 256 / 128 / 64
  for d=128 / 256 / 512 / 768).
- Dataset and norms: `yoran_raw.memmap` shards, per-read MAD
  normalisation in `KineticsNorm`.
- Random crop: `NormedDataset(crop_len=context)` is the only point
  where context enters the data path. At ctx=1024 the wrapper picks
  a random 1024-position window from each 4096-position stored
  segment per fetch, matching the `_long` augmentation philosophy.

Swept:
- `smrt2vec.context`: 1024 (vs 512 in `_long`).

The cosine LR curve overlays exactly with `_long` for the first 1M
steps, so probe-top1 vs step plots between ssl_58 `_long` and ssl_60
are directly visually comparable on the cosine portion.

## Walltime caveat

48h walltime (vs 96h in `_long`) is a hard constraint from the
remaining grant window. Per-arch step output in 48h at ctx=1024:

| arch | expected steps in 48h | cosine status |
|---|---|---|
| d=128_L4 | ~600k | cuts ~60% through (no min_lr tail) |
| d=256_L8 | ~960k | just reaches min_lr at walltime |
| d=512_L8 | ~1.15M | min_lr reached + ~150k tail |
| d=768_L8 | ~1.25M | min_lr reached + ~250k tail |

So d=128 is the one weak point in the comparison: its trajectory at
ctx=1024 won't reach the cosine end inside 48h. Comparisons to
`ssl_58_autoencoder_grid/size_d128_L4_long` are valid only on the
first ~600k steps. The other three sizes complete their cosine
inside walltime and can be compared along the full LR curve.

## Harness reuse

`_shared_train.py` is a verbatim copy from `ssl_58_autoencoder_grid/`
at the time of branch (mirrors the ssl_58 → ssl_59 split). If a
bugfix lands in ssl_58's harness after this branch, port the diff
here too.

## Submit

```
bash run.sh scripts/experiments/ssl_60_ctx1024_grid/size_d128_L4
bash run.sh scripts/experiments/ssl_60_ctx1024_grid/size_d256_L8
bash run.sh scripts/experiments/ssl_60_ctx1024_grid/size_d512_L8
bash run.sh scripts/experiments/ssl_60_ctx1024_grid/size_d768_L8
```

4 jobs x 8 GPU x 48h = 1536 GPU-h.
