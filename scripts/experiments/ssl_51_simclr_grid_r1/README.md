# ssl_51_simclr_grid_r1

Round-1 SimCLR scoping grid. Same NT-Xent objective, augmentation policy, data,
and step budget as `ssl_50_simclr_pilot`; only the encoder shape varies across
the four subdirectories below. Each subdir is an independent SLURM submission.

| subdir          | d_model | n_layers | n_head | measured params | est. H100-hours |
|-----------------|--------:|---------:|-------:|----------------:|----------------:|
| size_d128_L4    | 128     | 4        | 2      | 2.1M            | ~20             |
| size_d256_L8    | 256     | 8        | 4      | 11.4M           | ~80             |
| size_d512_L8    | 512     | 8        | 8      | 45.5M           | ~190            |
| size_d768_L8    | 768     | 8        | 12     | 102.2M          | ~380            |

The measured counts are higher than the plan's original table because the
CNN is a fixed 11-ResBlock stack whose param count scales with d_model²
(the plan assumed a near-linear scaling and undercounted at high d). The
top config landed at 102M rather than ~54M, which just fits under the
400 H100-hour per-request cap at 100 epochs; the original d=768 L=12
target (~130M) would have overshot and is deferred to Round 2.

## Submission

Each subdirectory is its own experiment from `run.sh`'s perspective. They can
be queued independently (per the current ~400 H100-hour-per-request cap):

```bash
bash run.sh scripts/experiments/ssl_51_simclr_grid_r1/size_d128_L4
bash run.sh scripts/experiments/ssl_51_simclr_grid_r1/size_d256_L8
bash run.sh scripts/experiments/ssl_51_simclr_grid_r1/size_d512_L8
bash run.sh scripts/experiments/ssl_51_simclr_grid_r1/size_d768_L8
```

## Data path and loss (per batch)

`B` = per-rank batch size from `simclr.batch_size`. `G = world_size · B`
is the global batch size after DDP all-gather. `T = 4096` (read length in
the OB007 memmap), `C = 32` (probe/crop context), `d = simclr.d_model`.

```
  PairedViewDataset + DataLoader (num_workers=8, pin_memory=True)
  ───────────────────────────────────────────────────────────────
  for each of B indices in the batch (in parallel across workers):
      x  ← ShardedMemmapDataset[idx]        [T=4096, 4]   (seq, fi, fp, pad)
      x  ← KineticsNorm(x)                  log1p + z-score on channels [fi, fp]
      v1, v2 ← AugmentationPolicy(x)        each view: random_subcrop→C,
                                            then Bernoulli-gated kinetics_noise /
                                            channel_dropout / temporal_blur
                                            (revcomp off by default — strand-
                                             asymmetric kinetics; see augment.py)
  default_collate stacks B items:
      view1 [B, C=32, 4]           view2 [B, C=32, 4]

                              │
                              ▼   torch.cat([view1, view2], dim=0) → [2B, C, 4]
  ╭───────────────────────────────────────────────╮
  │                 SmrtEncoder                   │   shared weights; one forward
  │                                               │   over the concatenated 2B batch
  │   SmrtEmbedding   [2B, C, d]                  │     seq_emb(d/2) ⊕ kin_linear(d/2)
  │   CNN: 11 ResBlocks [2B, d, C/4]              │     RF=107, 4× downsample; C=32 → T/4=8
  │   + PositionalEncoding [2B, C/4, d]           │
  │   × L TransformerBlocks [2B, C/4, d]          │     bidirectional attention,
  │                                               │     head_dim=64, pad-mask aware
  ╰─────────────────────┬─────────────────────────╯
                        │   c [2B, C/4, d]
                        ▼
              masked mean pool → h [2B, d]           mean over non-pad latent positions
                        │
                        ▼
             MLPProjectionHead → z [2B, 128]         2-layer: Linear(d→d) → GELU → Linear(d→128)
                        │
                        ▼
               split on dim 0                        z1, z2 ∈ [B, 128] on this rank
                        │
                        ▼
  ╭───────────────────────────────────────────────╮
  │                   NTXent                      │   symmetric, temp-scaled cross-entropy
  │                                               │   (SimCLR v1 §2.1, τ=0.1)
  │   F.normalize(z1), F.normalize(z2)            │   unit-norm rows → sim = cosine
  │                                               │
  │   ── DDP all_gather (differentiable) ──       │   grads only flow through the
  │   z1_all [G, 128]   z2_all [G, 128]           │   local B-slice; the rest are stop-grad
  │                                               │
  │   queries = cat(z1,     z2)    [2B,  128]     │   this rank's 2B views
  │   keys    = cat(z1_all, z2_all) [2G, 128]     │   global pool of candidate matches
  │                                               │
  │   sim = (queries · keysᵀ) / τ  [2B, 2G]       │
  │   sim[self_row, self_col] ← −∞                │   mask each query's own slot
  │                                               │
  │   positives (by construction):                │   z1_i ↔ corresponding z2 in z2_all;
  │     query z1_i  (row i, i<B)  ↔  z2-slot of i │   z2_i ↔ corresponding z1 in z1_all.
  │     query z2_i  (row B+i)     ↔  z1-slot of i │   All 2G−2 remaining slots per query
  │                                               │   are negatives.
  │   loss = CrossEntropy(sim, positive_indices)  │   per-batch scalar; accelerator.backward
  ╰─────────────────────┬─────────────────────────╯
                        │
                        ▼
                     loss (scalar)
```

Every `probe_every` epochs the encoder is handed off, frozen, to
`linear_probe_eval` (copied from `ssl_26_cpg_contrastive/train.py`),
which trains a single `Linear(d → 1)` on the center latent of labelled
CpG windows and reports `probe_top1` / `probe_auroc`. That metric — not
the NT-Xent training loss — is the pass criterion below.

## Layout

`_shared_train.py` holds the full training loop (SimCLRSmrt + NTXent + linear
probe every 5 epochs). Each `size_*/train.py` is a thin entry point that
delegates to `main()` in that shared module; only the per-size
`size_*/config.yaml` differs across runs. Keeping the loop in one file avoids
the copy-paste drift that plagues experiment trees, while respecting `run.sh`'s
convention of `train.py` + `config.yaml` per experiment.

## Pass criteria (go/no-go to Round 2)

- `size_d128_L4/`: probe_top1 must be ≥ 0.63 at end AND non-decreasing over the
  last 3 evaluations (matches pilot pass criterion; ensures the recipe works
  at all).
- Across the four sizes: probe_top1 should be monotone-increasing with model
  size. If probe_top1 is flat or inverted across sizes, model scaling is not
  paying off under this augmentation policy and Round 2 should not launch
  before revisiting the augmentation grid.
