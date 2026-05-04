# ssl_57_inputmask_grid_lnhead

Combines ssl_21's input-masked-prediction learning task with ssl_55's
modernized harness (LayerNorm projection head, non-finite-grad skip,
ChunkedRandomSampler, step-based cadences, encoder-only portable
checkpoints, Accelerate-state resume) and ssl_56's per-gap SSL pair val
diagnostic (adapted for a non-pair-trained encoder).

The intent is to bring the original `Smrt2VecInputMask + AgInfoNCE`
objective forward into the harness the SimCLR lineage produced, scale
it across the same four sizes ssl_55 swept, and read both the per-size
scaling slope (does the masked-pred objective benefit from capacity
the way SimCLR did?) and the dual-set pair-val curves (does the
encoder produce locally-coherent representations even though that
isn't its training objective?).

## The learning task

Per ssl_21:

- **Mask kinetics at the input level** (before the CNN). Sample mask
  centers at probability `p_mask` per position, expand each center into
  a contiguous span of `mask_size` bases, and zero out the IPD/PW
  channels at masked positions. Sequence-token (channel 0) and pad
  (channel 3) channels are preserved.
- **Predict masked latents from context.** The encoder sees the
  partially-masked input and produces a per-position projection
  `c_proj`. Targets are the *unmasked* CNN latents at the same
  positions. `AgInfoNCE` minimizes per-position cross-entropy over a
  pool of subsampled masked positions (`max_negatives=8192`).

ssl_21 used `p_mask=0.15`. ssl_57 starts with `p_mask=0.05` (sparser
masking → more unmasked context per masked position → easier task).
Both are config knobs and can be retuned without code changes.

## What's new vs ssl_21

- **`Smrt2VecInputMaskLN`** (`smrt_foundation/model.py`): the masked-pred
  sibling of ssl_55's `SimCLRSmrtLN`. Same encoder; the project head
  becomes `Linear → LN → GELU → Linear → LN`. The final `LayerNorm`
  bounds the per-channel scale of `c_proj` going into `AgInfoNCE`'s
  `F.normalize`, eliminating the same magnitude-runaway pathway ssl_55
  fixed for the SimCLR head. The encoder's state-dict keys are
  identical to `DirectClassifier.encoder` and the original
  `Smrt2VecInputMask.encoder`, so portable encoder checkpoints stay
  drop-in for downstream fine-tuning.
- **Step-cap schedule**: `total_steps=300000` is set explicitly in
  config; the cosine schedule is keyed off this rather than off
  `len(dl) × epochs`. The training loop iterates the DataLoader as many
  epochs as fit and exits when `global_step >= total_steps`, regardless
  of epoch boundary. This makes the X-axis directly comparable across
  sizes despite different per-step throughput.
- **All ssl_55 fixes**: ChunkedRandomSampler (sequential shard reads),
  non-finite-grad skip with TB counter, encoder-only portable
  checkpoints at every `checkpoint_every_steps`, Accelerate-state
  resume to `checkpoints/latest/`.
- **ssl_56 dual-set SSL pair val** (adapted; see below).

## SSL pair val adapter

ssl_56's `ssl_pair_val_eval` calls `model(v1, v2) → (z1, z2)`, expecting
a SimCLR-style pair-encoder. `Smrt2VecInputMaskLN` doesn't have that
forward; it has `model(x) → (c_proj, targets, mask_idx)` and applies
input masking which is wrong at eval time.

The ssl_57 adapter goes through the encoder directly:

```python
c1 = encoder.forward(v1)              # no input mask at eval
c2 = encoder.forward(v2)
z1 = c1[:, c1.shape[1] // 2, :]       # center latent
z2 = c2[:, c2.shape[1] // 2, :]
sim = (z1n @ z2n.T) / temperature
# ... per-gap top1, pos_cos, spearman as in ssl_56
```

Two design choices in this adapter:

1. **Center latent, no projection.** The center-latent convention is
   what the CpG linear probe uses for this exact encoder, so the pair
   val and the CpG probe are evaluating the same underlying
   representation. The project head is *not* applied — it was trained
   for per-position context-from-context retrieval, not pair retrieval,
   so applying it would mix two different signals.
2. **Reuses ssl_56's frozen pair val sets verbatim.** No new build is
   needed; `ssl_pair_val_yoran_v1.memmap` (in-distribution train) and
   `ssl_pair_val_ob007_v1.memmap` (held-out val) are already on disk.
   They were built at `target_len=32`, and `SmrtEncoder.forward` works
   on shorter inputs than its `max_len` (the same path the CpG probe
   uses).

### Honest scoping: pair val ≠ SSL task for this experiment

For ssl_56's encoder, pair val *is* the training task — both training
and eval ask "do positives match?". For ssl_57 they're disjoint:

- **Training task** is per-position masked prediction: given context,
  predict the masked latent at position *i* from surrounding unmasked
  positions. The per-position projection is optimized to match the
  per-position CNN target.
- **Pair val** asks whether two whole windows from the same molecule
  produce similar pooled embeddings.

There is no direct mathematical relationship between these. A
masked-prediction encoder that's *excellent* at its training task could
in principle:

- Learn purely local context-conditional features (sequence + nearby
  kinetics)
- Have no same-molecule coherence beyond what local-context similarity
  coincidentally provides
- Show flat or noisy pair-val curves
- Still score well on the CpG probe

That outcome would be fine. **A flat pair-val curve does not mean the
encoder is broken.**

### Reading the curves (asymmetric)

The pair val for this encoder is a **one-sided detector**: it can
falsify, not confirm. Treat it as a guardrail, not a goalpost.

| Pattern | Reading |
|---|---|
| `pos_cos` decays monotonically with gap, `top1` above chance at small g, near chance at large g | Encoder learned local features. Expected behaviour, neutral. |
| `pos_cos` flat-and-noisy across all gaps, `top1` at chance | Inconclusive — the encoder may simply not have a same-molecule invariance, which is fine for masked prediction. |
| `pos_cos` high and **flat across gap** (Spearman ρ ≈ 0, large mean) | **Red flag** — encoder is using global per-read statistics as a shortcut. Same failure mode ssl_54/55 had. |
| `pos_cos` *rising* with gap (ρ > 0) | Pathological. |
| Big gap between `train_top1` (high) and `val_top1` (low) | **Red flag** — memorisation of yoran-specific features. |

The CpG linear probe remains the load-bearing transfer metric. Watch
`val_ssl/{train,val}_spearman_cos_vs_gap` and the train-vs-val gap at
gap=32 as guardrails. Don't try to optimise pair-val numbers directly.

### Caveats specific to this encoder

- **Distribution shift at eval.** The encoder is trained at ctx=4096
  where the CNN's 107-bp receptive field always sees real data. The
  pair-val views are 32 bp, so the CNN sees ~75 bp of zero-padding in
  its RF at every position. This is the same shift the CpG probe has
  had since ssl_21 — empirically tolerable, but **the absolute
  numbers are not directly comparable to ssl_56's** pair-val numbers
  (ssl_56's encoder was trained at ctx=32 with no such shift). Use
  ssl_57 numbers cross-step (does the trajectory improve?) and
  cross-size (does d=768 outperform d=128?), not against ssl_56.
- **No projection head applied.** ssl_56 applies its SimCLR projection
  during pair val because that head was trained on pair-pooled
  embeddings; we deliberately skip ours because it was trained per-
  position. The eval is therefore measuring raw encoder output, which
  is also what the CpG probe sees.

### What would be cleaner

The most direct SSL-task validation would be a held-out
masked-prediction val: held-out reads at ctx=4096, same masking, compute
`val_loss / val_top1 / val_pos_cos` via AgInfoNCE on the held-out batch.
That would directly measure the training objective on unseen data
(language-model perplexity analogue). This was Option C in the design
questions; we chose pair-val adaptation (Option A) because no new build
script was needed. If pair val proves uninformative in practice, that
held-out masked-pred val is the obvious follow-up.

See ssl_56's README for the metric-key reference; the keys are
identical (`val_ssl/{train,val}_{loss,top1,pos_cos}_gap_{g}` and
`val_ssl/{train,val}_spearman_cos_vs_gap`), only the encoder under
evaluation and the meaning of "good" differ.

## The grid

Mirrors ssl_55's axes (`head_dim=64` ratio):

| Size      | d_model | n_layers | n_head | batch_size | global (8 GPU) |
|-----------|---------|----------|--------|------------|----------------|
| d128_L4   | 128     | 4        | 2      | 64         | 512            |
| d256_L8   | 256     | 8        | 4      | 32         | 256            |
| d512_L8   | 512     | 8        | 8      | 16         | 128            |
| d768_L8   | 768     | 8        | 12     | 8          | 64             |

Per-size batch sizes are set by ctx=4096 activation memory. Smaller
batches at larger sizes mean smaller global negative pools — the
contrastive task gets harder at the d=768 end of the grid for that
reason in addition to the deeper attention. ssl_21's
`max_negatives=8192` cap remains load-bearing at d=128 (local pool
~4800 masked latents × bs=64 = 307k per-rank candidates, capped to
8192) and becomes a no-op at d=512+ (local pool < 8192 because batch
size shrank).

## Schedule

- `total_steps=300000` uniform across all four sizes.
- Cosine schedule with `pct_start=0.10` → 30k warmup, 270k cosine.
- `max_lr=3e-4`, `weight_decay=0.02` (matches ssl_21).
- `grad_clip=5.0` with non-finite-grad skip (matches ssl_55).

Walltime budget is **48h uniform**. d=128 expected to finish 300k steps
in ~12-15h with cosine bottoming at min_lr; d=768 expected to run to
walltime mid-cosine at ~150k steps. Step-based ckpts mean the latest
portable encoder checkpoint at the cutoff is the deliverable for
size-comparison even when cosine didn't complete.

## Pass criterion

At d=128 (control / smallest size where ssl_21 trained):

- `probe_top1 ≥ 0.58` at the end (matches ssl_21's reported end value)
- AND `probe_top1` non-decreasing over the last 3 probe evaluations.

Cross-size scaling slope reported separately. Scaling is monotone if
end-of-training (or end-of-walltime) `probe_top1` increases with
`d_model` across the four sizes.

## Layout

- `_shared_train.py` — single training loop used by all four sizes.
- `size_<d>_L<L>/config.yaml` — per-size knobs (`d_model`, `n_layers`,
  `n_head`, `batch_size`); everything else (resources, probe paths,
  pair val paths, masking, optimizer, schedule, cadences, dataset)
  identical across the four.
- `size_<d>_L<L>/train.py` — 12-line thin wrapper (verbatim ssl_55/56
  pattern).

Submit a size with `bash run.sh scripts/experiments/ssl_57_inputmask_grid_lnhead/size_<d>_L<L>`.

## What's preserved from ssl_55

- LayerNorm projection head (now on the masked-pred project MLP, but
  same Linear→LN→GELU→Linear→LN structure).
- Non-finite-grad skip + `nonfinite_skip_count` TB scalar (defensive
  insurance, expected to never engage with the LN head).
- ChunkedRandomSampler with `chunk_size=2048`.
- Step-based probe / portable encoder ckpt / Accelerate-state resume
  cadences (every 10k steps).
- Encoder-only portable milestone format with norm stats sidecar.
- yoran SSL training source (artifact-free).

## What's preserved from ssl_56

- Per-gap SSL pair val with dual-set (train: in-distribution yoran,
  val: held-out ob007).
- All four metric families per gap: `top1`, `loss`, `pos_cos`,
  `spearman_cos_vs_gap`.
- Frozen `PairedGapMemmapDataset` + the existing build under
  `data/01_processed/val_sets/ssl_pair_val_*_v1.memmap/`.


# v5

Smallest surface change from v3: swap the encoder's CNN for the
existing `SmrtEncoderSmallRF` (4 ResBlocks, r0=27, 4x downsampling
preserved) via a new `Smrt2VecInputMaskLNSmallRF` wrapper. All other
v3 hyperparameters are preserved verbatim — `max_lr=3e-4`,
`pct_start=0.03`, 1M steps, `mask_size=10`, `p_mask=0.05`. The LR
fix is the orthogonal v4 axis; combining the two changes would
confound attribution.

## Hypothesis

The default CNN's `r0=107` makes the masked-prediction objective
largely interpolatable at `mask_size=10`. An output latent centred on a
masked region sees 107 input bases of which only ~10 are masked, so
~97 unmasked context bases are available to interpolate the masked
positions. The contrastive signal is therefore partially degenerate at
every masked position, which plausibly contributes to both the
epoch-1-peak-then-decay pattern (the encoder learns a useful local
feature immediately, then has nothing left to learn) and the
projection-head dimensional collapse seen in v1/v2/v3.

`CNNSmallRF` reduces `r0` to 27 bases. With `mask_size=10` that means
~37% of each latent's RF is masked — a BERT-style ratio that demands
genuine context-from-context inference rather than local
interpolation. If the trivial-shortcut hypothesis is right, v5 should
either (a) train without collapsing at v3's `max_lr=3e-4` and reach a
non-trivial probe, or (b) collapse at a different point in the
trajectory than v3, isolating RF as a contributing factor.

## What changed and why nothing else did

| Component | v3 | v5 | Reason for the choice |
|---|---|---|---|
| Encoder CNN | `CNN` (11 ResBlocks, r0=107) | `CNNSmallRF` (4 ResBlocks, r0=27) | Test the trivial-interpolation hypothesis. |
| Encoder downsampling | 4x | 4x | Preserved so probe head, pair-val adapter, and `_downsample_mask` math are unchanged. |
| Projection head | `Linear → LN → GELU → Linear → LN` | identical | Same magnitude bound on the contrastive output. |
| `max_lr` | 3e-4 | 3e-4 | Smallest surface change; isolate the RF variable. |
| `pct_start` | 0.03 | 0.03 | Carried from v3. |
| `total_steps` | 1,000,000 | 1,000,000 | Carried from v3. |
| `mask_size`, `p_mask` | 10, 0.05 | 10, 0.05 | Carried from v3. With RF=27 the mask covers ~37% of each latent's RF, a load-bearing change in the contrastive signal without retuning. |
| Resume guard arch_keys | 6 keys | 6 keys | No new key added. State-dict mismatch at torch load is sufficient if `resume_from` is misconfigured across variants. |
| State-dict transferability | n/a | not transferable from default-RF runs | ResBlock counts differ; this is fresh training only. |

## Implementation footprint

- New class `Smrt2VecInputMaskLNSmallRF` in `smrt_foundation/model.py`
  (subclass of `Smrt2VecInputMaskLN`, swaps `SmrtEncoder` for
  `SmrtEncoderSmallRF`; forward path inherited unchanged).
- New config knob `cnn_variant` in `_shared_train.py` (default
  `'default'`, `'small_rf'` selects the new class). Existing v2/v3/v4
  configs omit the key and resolve to the original `Smrt2VecInputMaskLN`
  — fully backward compatible.
- Four new size dirs `size_<d>_L<L>_1m_v5/` whose configs differ from
  v3 only by `experiment_name` suffix and `cnn_variant: 'small_rf'`.

## Pass criterion

Same as v3/v4 at d=128 (control / smallest size where ssl_21 trained):

- `probe_top1 ≥ 0.58` at the end (matches ssl_21's reported end value)
- AND `probe_top1` non-decreasing over the last 3 probe evaluations.

A persistent collapse at v5 would falsify the trivial-shortcut
hypothesis (or at minimum demote it from sole explanation), at which
point combining v4's lower LR with v5's smaller RF becomes the next
move.
