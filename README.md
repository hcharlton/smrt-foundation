# smrt-foundation
A foundation model for pacbio SMRT reads, providing a native understanding of kinetics wrt nucleotide context. The success case is producing a downstream classifier that using the pretrained encoder that beats (ideally by a significant margin) a classifier that is trained purely by supervised learning. This is directly comparable to wav2vec 2.0 minus the quantization module.

The supervised baseline for single-strand CpG methylation classification is established at 82% top-1 accuracy using a CNN-transformer encoder trained end-to-end on labeled kinetics windows. The bulk of the work over the past six months has been on the self-supervised pretraining side, which has proven more difficult than anticipated. A wav2vec 2.0-style contrastive approach (InfoNCE on masked kinetics) was implemented first, but probe accuracy actively declined over pretraining epochs — a systematic investigation traced this to a combination of normalization mismatch, context length mismatch (4096 pretraining vs 32 downstream), and fundamental task misalignment where the contrastive objective discards the kinetics signal relevant to methylation. Switching to a masked autoencoder with direct kinetics reconstruction stabilized the probe at ~66% but did not close the gap. Fine-tuning the best pretrained encoder reached 79%, narrowing the deficit to 3pp, though the comparison is confounded by optimizer schedule differences and the fact that pretraining and fine-tuning used the same data distribution — violating the core SSL premise of leveraging additional unlabeled data. A large-scale autoencoder pretraining run on 25G of OB007 reads (exp 29, ~1000 GPU hours) reached 63% linear-probe accuracy — comparable to the smaller CpG-data autoencoder, so additional unlabeled data alone did not close the gap. Current work runs on two tracks: (1) **SimCLR** with kinetic-feature augmentations (subcrop, channel dropout, Gaussian noise, temporal blur) as an alternative to input-masking pretext tasks — scaling from 2M to 102M params in ssl_50/51/52; and (2) a **data scaling grid** for CpG supervised fine-tuning (exps 41–46) to characterize when pretrained initializations help versus hurt. A significant amount of infrastructure was built along the way: a BAM-to-memmap data pipeline, multi-GPU training with HuggingFace Accelerate, three pretraining architectures (input-masked contrastive, masked autoencoder, augmentation-based SimCLR), a linear probe evaluation framework, epoch-capped data scaling grids with per-size batch scaling, and a two-stage fine-tuning pipeline with differential learning rates.

## Data
PacBio SMRT HiFi reads. Unlike normal genetic sequencing data this comes with two extra features: Pulse width (pw or rp/fp) and interpulse duration (ipd or fi/ri). Since modifications to nucleuotides are also forced through the polymerase, they alter the kinetics of the molecule along with nucleotide context. This allows the inference of modifications on a per-site basis. This is natively done on a dual-strand basis for CpG sites by PacBio, but many other types of modifications are missing. 
## Plan
1. ~~Establish a supervised baseline for single strand CpG methylation classification on 32/64 basepair samples.~~ Done (exp 20, ~82% top-1).
2. ~~Implement the "smrt2vec" following the design principles of wav2vec 2.0.~~ Done (`Smrt2VecInputMask`, `SmrtAutoencoder`).
3. ~~Run pretraining experiments.~~ Contrastive (exp 21/23/26) and autoencoder (exp 24/25) completed; fine-tuning (exp 27) reached 79%.
4. ~~Large-scale autoencoder pretraining on OB007.~~ Done (exp 29, 63% probe) — additional unlabeled data alone did not close the gap.
5. Test SimCLR with kinetic-feature augmentations as an alternative to input-masking (ssl_50 pilot, ssl_51/52 scaling grid 2M → 102M params).
6. Characterize the CpG supervised data scaling curve with and without pretrained initialization (exps 33–46, spanning 100 to 8M labeled training samples).

## Self supervised task
Three pretext tasks have been attempted. **Input-masked InfoNCE** (exps 21/23, `Smrt2VecInputMask`): mask kinetics channels at random spans in a 4096-base read, encode, and use InfoNCE to match masked-position outputs to unmasked targets. **Masked autoencoder** (exps 24/25, `SmrtAutoencoder`): same input masking but reconstruct the original kinetics at masked positions via MSE — trades contrastive matching for direct regression. **Augmentation-based SimCLR** (exps 50–52, `SimCLRSmrt`): no input masking — generate two augmented views of each read (random subcrop + channel dropout + Gaussian noise + temporal blur) and pull views of the same read together under NT-Xent while pushing all other in-batch views apart. The bet is that kinetic-feature augmentations preserve methylation identity.
### Experimental Option and Thoughts
None of the existing methods take advantage of coverage, which refers to the number of reads that align to the same position in the genome. Currently the model (even the SimCLR method) is taking windows of a read, and then using the contrastive loss objective to pull those views together. In contrastive learning, what is held invariant across the views tends to be something should be aligned for the downstream task, and since it is only really advantageous to use that in a unabeled context, that is based on some assumptions. For example, in the SimCLR paper, there are two views of the same image, and the task assumes that each image fundamentally represents one object, therefore the invariant of the two views is the class of the entity they represent, even though it is not explicitly labeled. So then the contrastive task is aligned with a classification task because the views that are pulled together are of the same object, and so the model learns to collapse the noise of the images toward the entity class invariant. In the context of this project, the generalized downstream task is modification at a single position in a read. We could use the coverage at a position to generate an tensor that is say of dimensions B,C,H,W = B,4,64,256 where there are 4 feature channels of ipd, pw, sm (agreements in the read-specific ccs position), sx (disagreements in the ccs read-specific position), 64 different reads all aligned to the exact genome position, and 256 base pairs of those reads. If we took multiple crops of this and used a SimCLR setup to pull their encoded representations together, what would the model learn? What's invariant in this section of data analagous to the original SimCLR setup with dogs, cats, etc.? The problem is that methylation or other modification is not necessarily the same for a given site in each read because each read is more likely to have come from a different cell than the other haplotype. 

Another idea: Roughly half of the reads come from one haplotype, and the other half from the other haplotype. And modification is MOSTLY symmetric across the haplotypes at one position. 
## Downstream task
Binary classification task of a fixed context window around CpG sites for a single strand of the sequence. So we take say 32 base pairs of data and feed that into the encoder and then attach a shallow classification head to make a logit for classification. 

## Status

### Supervised baseline: RESOLVED
Direct supervised training on v2 memmap shards reaches ~82% top-1 (exp 20). The 70% regression from the original memmap pipeline was caused by a reverse kinetics misalignment bug in `zarr_to_methyl_memmap.py` — see "Reverse kinetics misalignment" below.

### Self-supervised pretraining: IN PROGRESS
Earlier blockers (all resolved): information leakage from latent masking (fixed by input masking in `Smrt2VecInputMask`), normalization mismatch (fixed by shared `KineticsNorm`), missing fine-tuning infrastructure (built in exp 22), sparse masking (increased to p_mask=0.15).

**Input-masked objectives capped at ~66% probe.** Contrastive pretraining (exp 21/23) produced probe accuracy of ~58% that declined over epochs. Autoencoder pretraining (exp 24) stabilized at ~62%. Training on CpG data directly (exp 25/26) improved both by +4-5pp (autoencoder ~66%, contrastive ~63%). Fine-tuning the best autoencoder encoder (exp 27) recovered 13pp over the linear probe, reaching 79% — still 3pp below the supervised baseline.

**Scaling the input-masked autoencoder to 25G of OB007 (exp 29, ~1000 GPU hours) reached 63% probe** — comparable to exp 25 on CpG data, so additional unlabeled data alone did not close the gap. A reduced-receptive-field variant (exp 30, r0=27 vs 107) aimed to restore per-latent locality at ctx=32 and is still being evaluated.

**Current SSL direction: SimCLR with kinetic-feature augmentations (exps 50–52).** The augmentation-based positive pairing replaces input masking. ssl_50 is the pilot; ssl_51/52 is a scaling grid spanning 2M → 102M params. The d768_L8 model in ssl_51 showed gradient explosion post-warmup (norms of 1e9–1e10); ssl_52 adds gradient clipping (max_norm=5.0) and extra dataloader workers/prefetch to address it.

### Supervised data scaling: IN PROGRESS
Exps 33–46 characterize how CpG classification performance scales with labeled training data (100 to 8M samples) under different initializations and schedules. Exps 41–43 established epoch-capped step budgets so tiny datasets don't train for 400k epochs at bs=4096. Exps 44–45 added per-size batch scaling to restore stochastic gradient noise at small n, where runs with bs ≥ n_train were deterministically memorizing the training set. Exp 46 isolates the batch-scaling fix from the gradual-unfreeze schedule after 43/44/45 regressed at large n relative to exp 42.

TLDR/Summary: The pretraining, even when it's performing at its best (79 percent) is still worse than 1 epoch of training directly.  


### Shared encoder (`SmrtEncoder`)

All pretraining and supervised models share the same encoder. The forward path:

```
Input [B, T, 4]  (seq token, IPD, PW, pad mask)
  │
  ├─ SmrtEmbedding
  │    nuc: Embedding(5, d/2)  →  [B, T, d/2]     (scaled by √d)
  │    kin: Linear(2, d/2)     →  [B, T, d/2]     (scaled by √d)
  │    concat + LayerNorm      →  [B, T, d]
  │
  ├─ CNN (11 ResBlocks, two stride-2 for 4× downsampling)
  │    [B, T, d] → permute → [B, d, T] → conv stack → [B, d, T/4] → permute → [B, T/4, d]
  │    Also downsamples pad mask: [B, T] → max_pool → [B, T/4]
  │
  ├─ Sinusoidal PE: [B, T/4, d] += pe[:, :T/4]
  │
  └─ Transformer (n_layers × {LayerNorm → MultiHeadAttn → LayerNorm → MLP(d→4d→d)})
       [B, T/4, d] → [B, T/4, d]
```

Default: d_model=128, n_layers=4, n_head=4. CNN receptive field ≈ 107 bases.

### Contrastive pretraining (`Smrt2VecInputMask`)

Experiments 21, 23. Wav2vec 2.0 style: mask input kinetics, encode, match to unmasked targets via InfoNCE.

```
Input x [B, T, 4]
  │
  ├─ Targets branch (no grad):
  │    encoder.get_latents(x)  →  z_clean [B, T/4, d]
  │    LayerNorm(z_clean)      →  targets [B, T/4, d]
  │
  ├─ Masked branch:
  │    apply_input_mask(x)     →  x_masked [B, T, 4]  (kinetics ch 1,2 zeroed at random spans)
  │    encoder.get_latents     →  z [B, T/4, d]
  │    add_pe + transformer    →  c [B, T/4, d]
  │    MLP(d→d, GELU, d→d)    →  c_proj [B, T/4, d]
  │
  └─ Loss (AgInfoNCE):
       Select masked positions: c_proj[mask] [N, d], targets[mask] [N, d]
       L2-normalize both, cosine similarity matrix [N, N_gathered] / τ
       Cross-entropy against diagonal (each prediction matches its own target)
       Distributed: all_gather targets across ranks for more negatives
```

**Transfer results on full-read SSL data** (experiments 21, 23): InfoNCE loss converges, but the linear probe accuracy on CpG classification *declines* over epochs (~58%). Three compounding issues identified:

1. **Normalization mismatch** (fixed in exp 21 rerun): Probe computed separate `KineticsNorm` from CpG data instead of reusing SSL norm.
2. **Length mismatch** (tested in exp 23): Encoder trained on T=4096 (1024 after CNN) but probed on T=32 (8 after CNN). Reducing to T=128 didn't help.
3. **Task misalignment**: Targets are layer-normed CNN features that shift as the encoder trains (moving target). The loss operates in cosine-similarity space disconnected from actual kinetics values.

Experiment 26 re-tests contrastive on CpG data directly (labels discarded, context=32) to isolate whether the objective or the data regime/handling is the bottleneck.

### Contrastive pretraining, mask-token variant (`Smrt2VecInputMaskToken`)

Not yet wired to an experiment. Same contrastive objective as `Smrt2VecInputMask`, but the masked branch injects a learnable d_model mask token at the embedding output instead of zeroing kinetics channels at the raw input. Motivation: after `KineticsNorm` (log1p + z-score), zeros are indistinguishable from real kinetics that happen to sit at the mean of the normalized distribution, so the encoder has no explicit signal that a position is masked. A learnable token gives the encoder a distinct "this position is missing" representation, matching how wav2vec 2.0 and BERT handle masking.

```
Input x [B, T, 4]
  │
  ├─ Targets branch (no grad):
  │    encoder.get_latents(x)              →  z_clean [B, T/4, d]
  │    LayerNorm(z_clean)                  →  targets [B, T/4, d]
  │
  ├─ Masked branch:
  │    apply_input_mask(x)                 →  input_mask [B, T]   (contiguous random spans, padding excluded)
  │    encoder.embed(x_nuc, x_kin, x_pad)  →  x_emb [B, T, d]
  │    x_emb[input_mask] = mask_vec        →  learnable d_model mask token at masked positions
  │    encoder.cnn → add_pe → transformer  →  c [B, T/4, d]
  │    MLP(d→d, GELU, d→d)                 →  c_proj [B, T/4, d]
  │
  └─ Loss (AgInfoNCE):
       Same as Smrt2VecInputMask — select masked positions, L2-normalize,
       cosine similarity / τ, cross-entropy against diagonal, all_gather
       targets across ranks for more negatives.
```

The masked branch cannot reuse `encoder.get_latents()` because there is no mid-call hook to overwrite the embedding output before the CNN; the forward pass runs `encoder.embed → cnn → add_pe → forward_transformer` manually instead. Encoder weights remain interchangeable with `DirectClassifier` for fine-tuning since the underlying `SmrtEncoder` is unchanged.

### Masked autoencoder pretraining (`SmrtAutoencoder`)

Experiment 24. Replaces contrastive matching with direct kinetics reconstruction.

```
Input x [B, T, 4]
  │
  ├─ Save target: x_orig[..., 1:3]  →  kin_target [B, T, 2]
  │
  ├─ apply_input_mask(x)             →  x_masked [B, T, 4]
  │
  ├─ Encoder: SmrtEncoder(x_masked)  →  c [B, T/4, d]
  │
  ├─ Decoder (SmrtDecoder):
  │    permute                        →  [B, d, T/4]
  │    ConvTranspose1d(d, d, k=4, s=2, p=1) + GELU  →  [B, d, T/2]
  │    ConvTranspose1d(d, d, k=4, s=2, p=1) + GELU  →  [B, d, T]
  │    permute + Linear(d, 2)        →  kin_recon [B, T, 2]
  │
  └─ Loss (MaskedReconstructionLoss):
       MSE(kin_recon[mask], kin_target[mask])
       Targets are the original normalized kinetics (fixed, not learned)
       No all_gather — standard DDP gradient averaging
```

The decoder is intentionally shallow (two transpose convolutions + linear) so the encoder must learn informative representations; a deep decoder could reconstruct from minimal encoder features. Normalization (`KineticsNorm`: log1p + z-score) is computed once from SSL data and reused for both training and probe evaluation.

**Transfer results on full-read SSL data** (experiment 24): Probe accuracy stabilized at ~62% (no decline), a 4pp improvement over contrastive. Still 20pp below the supervised baseline (82%).

**CpG data regime** (experiments 25, 26): Both architectures retrained on CpG data directly (pos+neg combined, labels discarded, context=32). Autoencoder probe improved to ~66%, contrastive to ~63%. Confirms data regime matters (+4-5pp), but the gap to supervised 82% remains large.

**Fine-tuning** (experiment 27): Fine-tunes the exp 25 autoencoder encoder on labeled CpG data with a two-stage schedule (5 epochs frozen, 15 unfrozen with differential LR). Reached 79% top-1, recovering 13pp over the linear probe but landing 3pp below the supervised baseline. The deficit is confounded by optimizer schedule differences and matching data budget (see exp 28/29 in `docs/status.md`).

**Fine-tuning infrastructure** (experiment 22, `scripts/experiments/supervised_22_finetune/`): The original fine-tuning experiment, loading contrastive (exp 21) encoder weights. Two-stage training: frozen encoder + classification head (5 epochs), then unfreeze all with differential LR (encoder 3e-4, head 3e-3, 15 epochs). Exp 27 reuses this infrastructure with the autoencoder encoder.

## Process
### 1. Data Pipeline (`workflow.py`)
gwf only manages the data pipeline: BAM → Zarr → memmap → validation. The `CONFIG` dictionary at the top is a static registry that maps raw BAM files to intermediate Zarr stores and training-ready memmap tensors. Modify `CONFIG` when ingesting new datasets, then `gwf run` to process them.

### 2. Experiments (`run.sh`)
Each experiment lives in its own directory under `scripts/experiments/<name>/` with a `config.yaml` and `train.py`. Resource specs (cores, memory, walltime, GPUs) go in the config's `resources:` section so they stay with the experiment. Submit with:
```bash
bash run.sh scripts/experiments/ssl_21_pretrain              # uses config resources
bash run.sh scripts/experiments/supervised_20_full_v2 --mem=512gb  # sbatch override
```
Auto-detects environment (local/Gefion/GenomeDK). Job output lands in the experiment directory as `<jobid>.out`. Avoids hunting through `.gwf/logs/`.

### 3. EDA Plots (`plot.sh`)
Plot scripts live in `report/eda/<name>/plot.py`. Run any of them with:
```bash
bash plot.sh report/eda/model_input_heatmaps              # local: runs directly
bash plot.sh report/eda/fi_vs_ri_distributions --mem=128gb # HPC: submits sbatch with override
```
Auto-detects environment (local/Gefion/GenomeDK). Output goes to `report/eda/<name>/plot.svg`.

### 4. Tests (`test.sh`)
```bash
bash test.sh tests/test_kinetics_norm.py               # local: runs pytest
bash test.sh tests/                                      # local: all tests
bash test.sh tests/test_cpg_pipeline_fidelity.py --mem=64gb  # HPC: sbatch
```

### 5. Model Logic & Codebase Safety Net
The training scripts have internal `DEFAULT_SMRT2VEC` / `DEFAULT` dictionaries that establish baseline architectural parameters. This is a fail-safe so that older `config.yaml` files that don't have newly introduced variables still run without crashing.

### Standard Execution Flow
To launch a new experiment:
1. Make a directory under `scripts/experiments/` with a `config.yaml` (hyperparameters + `resources:` section) and `train.py`.
2. Make sure the dataset paths in `config.yaml` exist (check the `workflow.py` CONFIG registry).
3. `bash run.sh scripts/experiments/<name>`.


## Resolved Issues

### Reverse kinetics misalignment in zarr_to_methyl_memmap.py (RESOLVED)

The new memmap dataset class (original script now at `archive/zarr_to_methyl_memmap.py`) regressed to ~70% top-1 compared to the legacy parquet pipeline's ~80%. Root cause: kinetics/sequence misalignment in reverse strand windows. Experiments 17 and 18 confirmed the fix, both peaking at ~80%.

PacBio stores forward and reverse kinetics in different orders: `fi[i]` = forward kinetics at position `i` (forward order), but `ri[i]` = reverse kinetics at position `L-1-i` (reverse order). The v1 script (`zarr_to_methyl_memmap.py`) processes reverse windows with `np.flip(read_rev, axis=0)`, which flips all columns together. After the flip, the sequence at position `j` represents original position `L-1-j` (correct after RC), but ri at position `j` gives reverse kinetics at position `j` (because ri was already reversed, flipping puts it in forward order). The sequence and kinetics point to different genomic positions — every reverse window has kinetics from the mirror-image positions.

`np.flip` works correctly for fi/fp (forward order → flip → reverse order, matching the reversed sequence) but incorrectly for ri/rp (already reverse order → flip → forward order, opposite to the reversed sequence). Since ~50% of training windows are reverse strand, half the training data had misaligned features.

The v2 script (`zarr_to_methyl_memmap_v2.py`) fixes this by using explicit reverse indexing (`ri[L-end:L-start]`) matching the legacy script, with no `np.flip` on the whole read. Experiment 17 also works because using fi/fp for both strands sidesteps the issue entirely (fi is in forward order, so flipping aligns correctly).

#### What was ruled out along the way
1. **Normalization method + hyperparameters** (Experiment 16): log1p ZNorm + matched LR/ds_limit. Same ~70%.
2. **Train/val leakage** (`tests/test_legacy_leakage.py`): PASSED, no read_name overlap.
3. **CpG site finding differences**: both pipelines find the same sites.
4. **UInt16 vs uint8 kinetics truncation**: PacBio CCS tags are uint8, no truncation.

### Tests

See `tests/readme.md` for full documentation of all test modules. Summary:

#### `tests/test_cpg_pipeline_fidelity.py`
End-to-end fidelity: BAM -> Zarr -> CpG memmap. Verifies byte-exact kinetics transfer at every pipeline stage.

#### `tests/test_legacy_vs_new_pipeline.py`
Direct comparison between legacy parquet and new memmap pipelines. Traces the same BAM reads through both paths.

#### `tests/test_zarr_to_methyl_memmap_v2.py`
Integration tests for the v2 CpG memmap pipeline: Zarr -> shards -> LabeledMemmapDataset -> DataLoader.

#### `tests/test_kinetics_norm.py`
Equivalence tests verifying KineticsNorm matches ZNorm and SSLNorm on their respective dataset types.

#### `tests/test_autoencoder.py`
Unit tests for SmrtAutoencoder, SmrtDecoder, and MaskedReconstructionLoss. Verifies shapes, masking, gradients, and encoder weight transfer.

#### `tests/test_large_pretrain.py`
Tests for exp 29 features: random cropping, cosine LR schedule over 3000 epochs, periodic probe/checkpoint scheduling.

#### `tests/test_ssl_21_train.py` -- PASSED
Static AST analysis of ssl_21_pretrain/train.py. Verifies linear probe runs on all ranks (not gated behind `is_main_process`) and `wait_for_everyone()` synchronizes before next epoch.

#### `tests/test_kinetics_distributions.py`
Compares forward vs reverse kinetics distributions (fi vs ri, fp vs rp) at CpG sites via KS tests.

#### `tests/test_legacy_leakage.py` -- PASSED
Checks legacy parquet train/test split for read-level leakage. No overlap found.