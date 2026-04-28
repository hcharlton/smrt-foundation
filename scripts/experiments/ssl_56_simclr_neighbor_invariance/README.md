# ssl_56_simclr_neighbor_invariance

Pilot. Tests whether changing the SimCLR invariant from
"any-position-within-read" (ssl_54/55's `random_subcrop`) to **short-range
neighbor invariance** eliminates the probe oscillation and "best accuracy
at the start" pattern observed in ssl_54/55.

## The invariant

Positives are **two non-overlapping `target_len`-base windows separated by
`gap_bp` real (non-pad) bases on the same molecule**. The encoder is
asked to embed these two views similarly. Per-view augmentations
(channel dropout, Gaussian noise, temporal blur) are applied independently
to each view as in ssl_55.

This is implemented in `smrt_foundation/augment.py` as
`neighbor_pair_subcrop` + `NeighborPairAugmentationPolicy` (sibling of
`AugmentationPolicy`).

## Why it differs from ssl_54/55

ssl_54/55 paired *arbitrary distant 32-base windows from the same long
read* as positives. The contrastive loss minimizes the distance between
those distant windows, and the encoder learns "same-read membership" as
its discrimination signal — using global read-level properties
(polymerase noise, average kinetic intensity) rather than the *local*
kinetic patterns where methylation actually lives. The CNN's locality
bias gives the encoder useful CpG signal in epoch 1; the contrastive loss
progressively erodes it. d=768's higher capacity erodes it faster (full
collapse), d=512's slower (oscillation between 0.61 and 0.67).

Restricting positives to nearby windows aligns the contrastive task with
the locality the methylation signal lives at. Two windows 16–128 bases
apart often share a CpG-island context; their kinetic dynamics share local
polymerase behaviour; the encoder must use *local* features to match
them.

## The gap sweep

Four sub-experiments, varying only `augment.gap_bp`:

- **`gap_16/`** — tightest separation; views are nearly adjacent, share
  immediate kinetic neighbourhood. Tests whether very local positives
  produce monotone-improving probe trajectories.
- **`gap_32/`** — gap equals window length; total span 96 bases.
- **`gap_64/`** — ~average inter-CpG distance in a CpG island. Two
  windows at this separation often share methylation status if both are
  inside an island.
- **`gap_128/`** — total span 192 bases ≈ a typical CpG island length.
  Loosest; tests the upper limit of the locality-aware regime before the
  encoder can fall back to global read statistics.

## Pilot status

d=128 L=4 only at all four gaps. If a clear winner emerges with monotone
probe trajectory, the winning gap value gets scaled up to a full size
grid in a follow-up experiment. If all four gaps still oscillate with
best-at-start, the augmentation locality hypothesis is falsified and
focus pivots to masked-prediction (modernized Smrt2Vec lineage).

## Resources

Each sub-experiment runs on **4 GPUs (half a node)**, bs=1024 per rank →
8190 effective negatives per query, matching ssl_55_d128_L4's contrastive
task exactly. With four sub-experiments at half-node each, two full nodes
hold the entire sweep.

## What's preserved from ssl_55

- `SimCLRSmrtLN` encoder (LayerNorm projection head — keeps the
  magnitude-runaway fix from ssl_55).
- Non-finite-grad skip in `_shared_train.py` (defensive insurance,
  expected to never engage at d=128).
- All other hyperparameters: max_lr=3e-4, wd=1e-4, pct_start=0.05,
  grad_clip=5.0, temperature=0.1, ds_limit=0, epochs=50,
  ChunkedRandomSampler chunk_size=2048, probe/ckpt/resume cadences 10k.
- Per-view augmentation parameters (channel_dropout_p, gaussian_noise_*,
  blur_*) identical to ssl_55.

The only structural difference vs ssl_55_d128_L4 is the augmentation
policy class. The only varied parameter across the four sub-experiments
is `gap_bp`.
