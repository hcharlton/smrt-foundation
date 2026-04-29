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

## Diagnostics: per-gap SSL pair val probe (train + val)

Alongside the CpG linear probe, every `probe_every_steps` (10k) the
trainer evaluates the model on **two** frozen pair sets:

- **`ssl_pair_val_train`** → `ssl_pair_val_yoran_v1.memmap`.
  *In-distribution* probe. The same yoran reads were the training
  source, so the model has seen them with augmentation. The pairs
  themselves (specific anchors, no augmentation noise at eval) are
  novel, so this isn't quite a training-set re-evaluation, but it isn't
  proper held-out either — best read as "can the model match positives
  on its own training distribution at all?"
- **`ssl_pair_val_val`** → `ssl_pair_val_ob007_v1.memmap`.
  *Held-out* probe. ob007 is a different sample, never in the SSL
  training source. This is the genuine generalisation signal:
  same-task, OOD distribution.

Both probe sets are 19 fixed non-overlapping window-pair conditions at
gap separations of 0, 32, 64, …, 512, 1024, and 2048 bp, sampled
deterministically with seed 42. Per-gap stratification turns each
metric into a conditional estimate given a specific gap; the *shape*
of the curve across gaps and the *gap between train and val numbers*
are what the diagnostic is for.

The motivation is identification, not validation. The CpG linear probe
conflates several failure modes — framework broken, augmentation
misaligned, transfer broken. The dual-set pair probe disambiguates
them by exposing two axes the CpG probe collapses:

- **Across training (over `global_step`):** does the SSL task itself
  improve on either set? If both stay at chance, the framework isn't
  learning.
- **Train vs val at any given step:** if `train_top1` is high but
  `val_top1` is at chance, the encoder memorised yoran-specific
  read-level features and won't transfer. If both are high, real
  invariance was learned. If both are flat across `g`, the encoder
  is discriminating via global per-read statistics (the ssl_54/55
  failure mode this experiment is designed to break) regardless of
  read identity.

For each gap value `g`, the trainer logs the same four families of
metrics for **both** the train set and the val set, prefixed with
`train_` or `val_`:

- `val_ssl/{train,val}_top1_gap_{g}` — fraction of within-batch
  positives that rank first among `B-1` batch-mate negatives at the
  training temperature (0.1). Chance is `1/B`; default
  `B = val_pair_batch_size = 1024`. The classification reading of the
  contrastive task.
- `val_ssl/{train,val}_loss_gap_{g}` — per-batch NTXent cross-entropy.
  Same loss family as `train_loss` but with a smaller negative pool
  (B vs the all-gather pool of `2 · world_size · B − 2` used in
  training). Absolute magnitudes don't line up with `train_loss`
  exactly, but trends and per-gap ordering are directly comparable.
- `val_ssl/{train,val}_pos_cos_gap_{g}` — mean cosine similarity of
  the matched pair on L2-normalised projection-head outputs. The
  regression-flavoured signal: does the embedding space encode a
  monotone notion of pair similarity over the source-side gap, or has
  it collapsed to position-invariance?
- `val_ssl/{train,val}_spearman_cos_vs_gap` — Spearman rank correlation
  between per-gap mean cosine similarity and the gap value, computed
  across the 19 gaps. Non-parametric summary of the cosine-vs-gap
  curve's monotonicity.

Reading the curves:

- **Both `train_top1` and `val_top1` rise above chance, decay
  monotonically with `g`, and track each other closely** → encoder
  learned generalisable local-context discrimination. The desired
  outcome.
- **`train_top1` high, `val_top1` near chance** → the encoder
  memorised yoran-specific features (read identity, polymerase-quality
  statistics specific to that sample) and didn't generalise. The SSL
  framework is "working" in a narrow sense but the learned features
  aren't useful.
- **Both at chance across the entire sweep** → SSL task isn't being
  learned at all. Most likely a pipeline bug (norm mismatch,
  augmentation degenerate solution, projection-head collapse) rather
  than augmentation–task alignment. Stop tuning hyperparameters and go
  find the bug.
- **Both high and approximately flat across `g`** → encoder is
  discriminating via global per-read statistics (kinetic intensity,
  polymerase noise). This is the ssl_54/55 failure mode, *generalised
  to ob007*: it means the encoder learned a feature that's stable
  across both samples but uninformative about local context.
- **`{train,val}_spearman_cos_vs_gap` < 0** → cosine sim drops with
  gap, positional info preserved, encoder using local features.
- **`{train,val}_spearman_cos_vs_gap` > 0** → similarity *grows* with
  gap. Pathological.

Caveat on the val set: ob007 has known artifacts (the reason ssl_54
moved to yoran). A poor `val_top1` could reflect either (a) genuine
generalisation failure or (b) ob007's distribution-shift hits a
specific artifact pattern. For the diagnostic question this experiment
asks ("is the framework learning anything generalisable at all?"),
that conflation is mostly tolerable — both readings trigger
investigation, not a green light. A second held-out source from a
different sample without ob007's specific artifacts would be cleaner
long-term.

Eval cost is bounded at `limit_per_gap × 19 × 2 sets = 380k` pair
forwards per probe step (default; tunable via `val_pair_limit_per_gap`
in the config), well under the existing CpG probe's wall time.
Implementation matches `linear_probe_eval`'s convention: every rank
computes the metrics independently with `shuffle=False`, only the main
process logs.

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
