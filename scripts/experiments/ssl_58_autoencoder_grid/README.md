# ssl_58_autoencoder_grid

Scaled masked-autoencoder pretraining across the 4-size grid
(d128_L4 / d256_L8 / d512_L8 / d768_L8) at ctx=512 on the yoran SSL
training set, using the modern ssl_57 harness.

## Why

Every contrastive variant in this project (ssl_50/51/53/54/55/56/57) has
either failed to beat the random-init probe baseline, collapsed via
projection-head dimensional collapse, or both. The diagnosis from
ssl_57 v3/v4/v5 is structural: contrastive learning makes the encoder
invariant to whatever defines the positive pair, and every readily-
available pair definition (within-read, neighbor windows, same-tissue,
etc.) ends up suppressing the per-position kinetic variation that CpG
methylation lives in. Lower LR delays collapse without changing the
destination; smaller RF doesn't change the contrastive optimum.

The autoencoder lineage is the only family in this project that has
consistently produced positive transfer:

| Experiment | Setup | Probe top-1 |
|---|---|---|
| ssl_24 | Autoencoder, full read, ctx=128 | 62-63% |
| ssl_25 | Autoencoder, CpG data, ctx=32 | 66% |
| ssl_29 | Autoencoder, 25G OB007, ctx=128 | 63% |

MSE on masked kinetics directly penalizes losing kinetic signal —
including the kinetic patterns where methylation is encoded — so the
encoder is forced by the loss surface to preserve per-position
variation. There is no contrastive collapse pathway: the loss has a
non-degenerate minimum.

ssl_58 takes that working recipe and scales it up:

- **Larger context** (ctx=512 vs the 32-128 of prior autoencoder runs).
  Saturates the GPUs that ssl_56-style 32-bp contexts left dataloader-
  bound, while still being short enough that the small-RF CNN's locality
  assumption holds (RF=27 ≪ 512).
- **Small-RF CNN** (`SmrtEncoderSmallRF`, r0=27 vs the default r0=107).
  Preserves locality so per-latent reconstruction is genuinely
  context-from-context — ssl_30's structural fix, never tested at scale.
  4× downsampling preserved → probe head and pair-val adapter are
  unchanged.
- **Modern harness** (the entire ssl_57 fix stack — see checklist below).
  We don't regress on any of the bugs that took 4 versions of ssl_57
  to find.
- **Standard 4-size grid** mirroring ssl_55/56/57 axes for direct
  architecture comparison.

## Pass criterion

At d=128 (control / smallest size where ssl_25 trained):

- `probe_top1 ≥ 0.67` at the end (beats ssl_25's reported 0.66)
- AND `probe_top1` non-decreasing over the last 3 probe evaluations.

Stretch goal: any size reaches `probe_top1 ≥ 0.75` (closes half the gap
to the supervised ceiling of 0.82).

## Bug-fix inheritance from ssl_53 → ssl_57

Inherited verbatim or with the autoencoder-relevant adjustment noted:

| # | Fix | Inherited as |
|---|---|---|
| 1 | No `accelerator.prepare(scheduler)` — raw LambdaLR stepped manually | YES, identical |
| 2 | Non-finite-grad skip + `nonfinite_skip_count` TB scalar | YES, identical |
| 3 | LayerNorm in any projection head | N/A — autoencoder has no projection head; decoder ends in `Linear(d_model, 2)` directly into MSE |
| 4 | `AgInfoNCE` sync-min subsample | N/A — `MaskedReconstructionLoss` has no all-gather, no NCCL collectives in the loss path |
| 5 | `ChunkedRandomSampler` (chunk_size=2048) | YES, identical |
| 6 | `ProgressState` registered with Accelerate state | YES, identical |
| 7 | `run_metadata.yaml` sidecar + `_check_resume_compatible` arch_keys | YES, identical (arch_keys cover the autoencoder's parameters) |
| 8 | `skip_first_batches(dl, skip_n)` mid-epoch resume | YES, identical |
| 9 | Encoder-only portable milestones | YES — drops the decoder, saves only the encoder |
| 10 | Step-cap schedule (`global_step >= total_steps` exits the outer loop) | YES, identical |
| 11 | Step-based cadences (`probe/ckpt/resume_every_steps=10000`) | YES, identical |
| 12 | Step-0 baseline eval (probe + pair-val on random-init encoder) | YES, identical |
| 13 | Linear probe eval on labeled CpG data | YES, identical |
| 14 | SSL pair-val with dual-set (yoran in-dist, ob007 held-out) | YES, identical (guardrail diagnostic) |
| 15 | TB collapse diagnostics: `embed_z_std`, `embed_z_norm`, `grad_norm`, `nonfinite_skip_count` | YES — `embed_z_std`/`embed_z_norm` now log on the encoder transformer output `c` (captured via a forward hook) instead of the projection head output `c_proj`, since the autoencoder has no projection head. Same TB keys so trajectories overlay visually with ssl_57. |
| 16 | TB scalars at startup: `architecture/cnn_receptive_field`, `architecture/param_count` | YES, identical |

## What changes vs ssl_57

Three places in the harness:

1. **Model:** `SmrtAutoencoderSmallRF` (model.py:757) instead of
   `Smrt2VecInputMaskLN(SmallRF)`. Forward returns
   `(kin_recon, kin_target, mask)`. No projection head.
2. **Loss:** `MaskedReconstructionLoss()` (loss.py:57) instead of
   `AgInfoNCE`. Computes `F.mse_loss(kin_recon[mask], kin_target[mask])`.
3. **Diagnostic capture:** a forward hook on the encoder submodule
   captures `c` for `embed_z_std` / `embed_z_norm` logging without
   changing the autoencoder's forward contract (which is shared with
   ssl_30).

## The grid

Mirrors ssl_55/56/57 axes (`head_dim=64` constant):

| Size | d_model | n_layers | n_head | batch_size | global (8 GPU) |
|------|---------|----------|--------|------------|----------------|
| d128_L4 | 128 | 4 | 2 | 512 | 4096 |
| d256_L8 | 256 | 8 | 4 | 256 | 2048 |
| d512_L8 | 512 | 8 | 8 | 128 | 1024 |
| d768_L8 | 768 | 8 | 12 | 64 | 512 |

Per-rank batch sizes are 8× ssl_57's at the same architecture (since
ctx=512 is 8× smaller, activation memory budget is preserved).

## Schedule

- `total_steps=1,000,000` uniform across all four sizes.
- Cosine schedule with `pct_start=0.03` → 30k warmup, 970k cosine.
- `max_lr=3e-4` (autoencoder canonical from ssl_25/29/30).
- `weight_decay=0.02`, `grad_clip=5.0` with non-finite-grad skip.
- `p_mask=0.15`, `mask_size=10` (autoencoder lineage convention; with
  RF=27, mask_size=10 covers ~37% of each latent's RF, BERT-style ratio).

Walltime budget is **48h uniform** per size (4 × 8-GPU nodes total). At
d=128 with bs=512 ctx=512, roughly comparable throughput to ssl_57 at
d=128. d=768 expected to land mid-cosine ~150k steps, same pattern as
ssl_57 1m.

## Layout

- `_shared_train.py` — single training loop used by all four sizes.
- `size_<d>_L<L>/config.yaml` — per-size knobs (`d_model`, `n_layers`,
  `n_head`, `batch_size`); everything else identical across the four.
- `size_<d>_L<L>/train.py` — 12-line thin wrapper.

Submit a size with:
```
bash run.sh scripts/experiments/ssl_58_autoencoder_grid/size_<d>_L<L>
```
