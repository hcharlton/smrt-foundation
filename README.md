# smrt-foundation
A foundation model for pacbio SMRT reads, providing a native understanding of kinetics wrt nucleotide context. The success case is producing a downstream classifier that using the pretrained encoder that beats (ideally by a significant margin) a classifier that is trained purely by supervised learning. This is directly comparable to wav2vec 2.0 minus the quantization module. 
## Data
PacBio SMRT HiFi reads. Unlike normal genetic sequencing data this comes with two extra features: Pulse width (pw or rp/fp) and interpulse duration (ipd or fi/ri). These respectively 
## Self supervised task
Use an info-NCE loss on masked latents in a 4096 sequence of SMRT. This involves masking a large percentages of the indices and then teaching the model how to make a projection that is more similar to the label (positive) and dissimilar to the in-batch negatives. 
## Downstream task
Binary classification task of a fixed context window around CpG sites for a single strand of the sequence. So we take say 32 base pairs of data and feed that into the encoder and then attach a shallow classification head to make a logit for classification. 

## Status

### Supervised baseline: RESOLVED
Direct supervised training on v2 memmap shards reaches ~80% top1 (experiment 20 running on full dataset). The 70% regression from the original memmap pipeline was caused by a reverse kinetics misalignment bug in `zarr_to_methyl_memmap.py` — see "Debug new labeled dataset" below.

### Self-supervised pretraining: IN PROGRESS
Contrastive encoder learns (InfoNCE loss converges), but does not generalize to downstream CpG classification. Analysis identified four blockers:
1. **Information leakage through CNN receptive field.** The CNN has a 107-base receptive field. With latent masking (masking AFTER the CNN), adjacent unmasked latents share 96% of their input bases with masked positions. The transformer can trivially reconstruct masked positions by interpolating from neighbors — the pretraining task is too easy and doesn't force learning meaningful representations.
2. **Normalization mismatch**: SSL trains on raw uint8 kinetics (0-255), supervised uses log1p + z-score (~-3 to +3). The `kin_embed` linear layer learns weights calibrated for the wrong input scale, making weight transfer fail.
3. **No fine-tuning infrastructure**: No code exists to load pretrained weights, freeze/unfreeze layers, or use differential learning rates.
4. **Sparse masking**: p_mask=0.05 is too easy for the same reason as #1.

### Pretraining strategy
Experiment 21 (`scripts/experiments/ssl_21_pretrain/`): SSL pretraining with:
- **Input masking** via `Smrt2VecInputMask` (`model.py`): masks raw kinetics BEFORE the CNN (wav2vec 2.0 style), forcing the encoder to learn without information at masked positions. CNN runs twice per step (masked input for context, unmasked for targets). Replaces the old `Smrt2Vec` latent masking approach (which remains available for comparison).
- ZNorm log1p normalization matching the supervised pipeline
- Higher masking: p_mask=0.15, mask_size=10
- Linear probe evaluation after each epoch (frozen encoder + linear head on labeled data) to track whether representations are becoming useful for classification
- Model checkpointing for downstream transfer

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