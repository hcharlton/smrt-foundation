# smrt-foundation
A foundation model for pacbio SMRT reads, providing a native understanding of kinetics wrt nucleotide context. The success case is producing a downstream classifier that using the pretrained encoder that beats (ideally by a significant margin) a classifier that is trained purely by supervised learning. This is directly comparable to wav2vec 2.0 minus the quantization module. 
## Data
PacBio SMRT HiFi reads. Unlike normal genetic sequencing data this comes with two extra features: Pulse width (pw or rp/fp) and interpulse duration (ipd or fi/ri). Since modifications to nucleuotides are also forced through the polymerase, they alter the kinetics of the molecule along with nucleotide context. This allows the inference of modifications on a per-site basis. This is natively done on a dual-strand basis for CpG sites by PacBio, but many other types of modifications are missing. 
## Plan
1. Establish a supervised baseline for single strand CpG methylation classification on 32/64 basepair samples. 
2. Implement the "smrt2vec" following the design principles of wav2vec 2.0. 
3. Run pretraining experiments 
4. Align pretraining down downstream tasks (???)
## Self supervised task
Use an info-NCE loss on masked latents in a 4096 sequence of SMRT. This involves masking a large percentages of the indices and then teaching the model how to make a projection that is more similar to the label (positive) and dissimilar to the in-batch negatives. 
## Downstream task
Binary classification task of a fixed context window around CpG sites for a single strand of the sequence. So we take say 32 base pairs of data and feed that into the encoder and then attach a shallow classification head to make a logit for classification. 

## Status

### Supervised baseline: RESOLVED
Direct supervised training on v2 memmap shards reaches ~82% top1 (experiment 20 running on full dataset). The 70% regression from the original memmap pipeline was caused by a reverse kinetics misalignment bug in `zarr_to_methyl_memmap.py` — see "Debug new labeled dataset" below.

### Self-supervised pretraining: IN PROGRESS
Earlier blockers (all resolved): information leakage from latent masking (fixed by input masking in `Smrt2VecInputMask`), normalization mismatch (fixed by shared `KineticsNorm`), missing fine-tuning infrastructure (built in exp 22), sparse masking (increased to p_mask=0.15).

Current status: both contrastive (~58% probe) and autoencoder (~62% probe) pretraining on full-read SSL data produce representations far below the supervised baseline (82%). Experiments 25/26 test whether training on CpG data directly (labels discarded, context=32) closes the gap — isolating data regime vs pretraining objective as the bottleneck.

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

Experiment 26 re-tests contrastive on CpG data directly (labels discarded, context=32) to isolate whether the objective or the data regime is the bottleneck.

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

**Current experiments** (25, 26): Both architectures retrained on CpG data directly (pos+neg combined, labels discarded, context=32). Eliminates the data distribution, context length, and normalization mismatches. If probe accuracy improves significantly, the bottleneck was data regime. If not, the pretraining objectives themselves are insufficient for this task.

Experiment 22 (`scripts/experiments/supervised_22_finetune/`): Fine-tune pretrained encoder:
- Load encoder weights from experiment 21 checkpoint
- Stage 1: frozen encoder, train classification head only (5 epochs)
- Stage 2: unfreeze all, differential LR (encoder 3e-4, head 3e-3, 15 epochs)
- Compare against experiment 20 direct training baseline

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

### 5. Model Logic & Codebase Safety Net (`scripts/train.py`)
The training scripts have internal `DEFAULT_SMRT2VEC` / `DEFAULT` dictionaries that establish baseline architectural parameters. This is a fail-safe so that older `config.yaml` files that don't have newly introduced variables still run without crashing.

### Standard Execution Flow
To launch a new experiment:
1. Make a directory under `scripts/experiments/` with a `config.yaml` (hyperparameters + `resources:` section) and `train.py`.
2. Make sure the dataset paths in `config.yaml` exist (check the `workflow.py` CONFIG registry).
3. `bash run.sh scripts/experiments/<name>`.


## TODO + Problems
### Debug new labeled dataset
The legacy dataset class (script at archive/make_legacy_labeled_dataset, dataset at data/01_processed/legacy*.parquet) was performing much better with direct downstream training (using all the model components) than the new dataset class (original script now at archive/zarr_to_methyl_memmap.py). The reason I originally made the new dataset class was to upgrade the dataset to online normalization to run experiments, since the contrastively pretrained encoder (the bulk of the project) was not generalizing to the downstream methylation classification task

#### Root cause: reverse kinetics misalignment in zarr_to_methyl_memmap.py

**RESOLVED.** Experiments 17 and 18 both peak at ~80% top1, matching the legacy baseline. The bug was a kinetics/sequence misalignment in reverse strand windows.

PacBio stores forward and reverse kinetics in different orders: `fi[i]` = forward kinetics at position `i` (forward order), but `ri[i]` = reverse kinetics at position `L-1-i` (reverse order). The v1 script (`zarr_to_methyl_memmap.py`) processes reverse windows with `np.flip(read_rev, axis=0)`, which flips all columns together. After the flip, the sequence at position `j` represents original position `L-1-j` (correct after RC), but ri at position `j` gives reverse kinetics at position `j` (because ri was already reversed, flipping puts it in forward order). The sequence and kinetics point to different genomic positions — every reverse window has kinetics from the mirror-image positions.

`np.flip` works correctly for fi/fp (forward order → flip → reverse order, matching the reversed sequence) but incorrectly for ri/rp (already reverse order → flip → forward order, opposite to the reversed sequence). Since ~50% of training windows are reverse strand, half the training data had misaligned features.

The v2 script (`zarr_to_methyl_memmap_v2.py`) fixes this by using explicit reverse indexing (`ri[L-end:L-start]`) matching the legacy script, with no `np.flip` on the whole read. Experiment 17 also works because using fi/fp for both strands sidesteps the issue entirely (fi is in forward order, so flipping aligns correctly).

#### What was ruled out along the way
1. **Normalization method + hyperparameters** (Experiment 16): log1p ZNorm + matched LR/ds_limit. Same ~70%.
2. **Train/val leakage** (`tests/test_legacy_leakage.py`): PASSED, no read_name overlap.
3. **CpG site finding differences**: both pipelines find the same sites.
4. **UInt16 vs uint8 kinetics truncation**: PacBio CCS tags are uint8, no truncation.

### Tests

#### `tests/test_ssl_21_train.py` — PASSED
Static analysis tests for the ssl_21_pretrain training script. Verifies that the linear probe evaluation runs on all ranks (not just `is_main_process`) and that `wait_for_everyone()` is called between the probe and the next epoch. These guards prevent the NCCL timeout / "Invalid mt19937 state" crash that occurs when ranks desynchronize at epoch boundaries.