# smrt-foundation
A foundation model for pacbio SMRT reads, providing a native understanding of kinetics wrt nucleotide context. The success case is producing a downstream classifier that using the pretrained encoder that beats (ideally by a significant margin) a classifier that is trained purely by supervised learning. This is directly comparable to wav2vec 2.0 minus the quantization module. 
## Data
PacBio SMRT HiFi reads. Unlike normal genetic sequencing data this comes with two extra features: Pulse width (pw or rp/fp) and interpulse duration (ipd or fi/ri). These respectively 
## Self supervised task
Use an info-NCE loss on masked latents in a 4096 sequence of SMRT. This involves masking a large percentages of the indices and then teaching the model how to make a projection that is more similar to the label (positive) and dissimilar to the in-batch negatives. 
## Downstream task
Binary classification task of a fixed context window around CpG sites for a single strand of the sequence. So we take say 32 base pairs of data and feed that into the encoder and then attach a shallow classification head to make a logit for classification. 

## Status
Contrastive encoder learns, but does not generalize to downstream task.
### Strategy
1. Add online normalization.
    1.1 Use the new dataset class (numpy shards) to do this.
    1.2. ISSUE: now the direct training is significantly worse (max around 70 percent top1 wheras the legacy version was 81). Need to identify the bug (this is a priority)
2. Switch to input masking instead of latent masking (For pretraining)

### Exploratory analysis of problem
- analyze the encoder outputs (histogram of activations?) for both the legacy and new datasets
- plot by-index means of the two datast

## Process
### 1. Infrastructure Registry (`workflow.py`)
This script defines the Directed Acyclic Graph (DAG) for data processing and compute allocation. The internal `CONFIG` dictionary acts as a static registry for data provenance, mapping raw BAM files to intermediate Zarr stores and final training-ready Memmap tensors.
* **Usage:** Modify the `CONFIG` dictionary only when ingesting new raw sequencing datasets, adjusting genome chunking logic, or altering cluster resource allocation (e.g., GPU requests or walltimes). 

### 2. Experiment Control (`config.yaml`)
This is the single source of truth for individual training runs. It defines the hyperparameter topology, target dataset selection, and experiment metadata. During execution, an immutable snapshot of this configuration is embedded into the TensorBoard event logs alongside the Git commit hash to guarantee run reproducibility.
* **Usage:** Modify this file for every new experiment. Update `project_name`, `run_message`, and specific model or optimizer hyperparameters prior to triggering the workflow.

### 3. Model Logic & Codebase Safety Net (`scripts/train.py`)
The training script utilizes an internal `DEFAULT_SMRT2VEC` dictionary to establish baseline architectural parameters. This implements a fail-safe mechanism, ensuring backward compatibility by providing default values if a legacy `config.yaml` is executed that lacks newly introduced variables. 
* **Usage:** Modify these defaults only when introducing structural changes to the `Smrt2Vec` architecture (e.g., new projection heads for the single strand methylation classifier, updated kinetics masking ratios) to prevent execution failures on older configurations.

### Standard Execution Flow
To launch a new experimental iteration:
1. Define the run metadata, target dataset, and hyperparameter configuration in `config.yaml`.
2. Verify the dataset requested in `config.yaml` exists within the `workflow.py` registry.
3. Evaluate and execute the graph via `gwf run`. The engine will parse the updated `project_name` from the YAML, identify the unfulfilled `run.sentinel` artifact, and dynamically submit the requisite data pipeline and distributed training jobs.


## TODO + Problems
### Debug new labeled dataset
The legacy dataset class (script at archive/make_legacy_labeled_dataset, dataset at data/01_processed/legacy*.parquet) is performing much better with direct downstream training (using all the model components) than the new dataset class (script at scripts/zarr_to_methyl_memmap.py). I have not been able to figure out why. The reason that I originally made the new dataset class was to upgrade the datset to online normalization to run experiments, since the contrastively pretrained encoder (the bulk of the project) was not generalizing to the downstraem methylation classification task

#### What's been ruled out
1. **Normalization method + hyperparameters** (Experiment 16): Added log1p transform to ZNorm to match legacy's log(x+1) z-score normalization. Also aligned max_lr (3e-3) and ds_limit (2M) with legacy config. Result: same ~70% top1.
2. **Train/val leakage** (`tests/test_legacy_leakage.py`): Checked whether read_names overlap between legacy train/test parquets. Result: PASSED. The original legacy script (`archive/legacy_dataset_script_unmodified.py`) splits by read_name with an explicit disjointness assert (line 249).
3. **CpG site finding differences**: `re.finditer("CG", seq)` (legacy) and `sliding_window_view(seq, 2)` (new) find the same CpG sites — "CG" is 2 chars and can't self-overlap. Both also handle "CGCG" identically (two matches at positions 0 and 2).
4. **UInt16 vs uint8 kinetics truncation**: The legacy parquet stores kinetics as UInt16, but PacBio CCS fi/fp/ri/rp are uint8 arrays in the BAM spec, so all values are in [0, 255]. No truncation in the Zarr path.

#### Pending experiments
1. **Experiment 17 (fwd kinetics only)**: Use fi/fp for both strands (matching legacy behavior). Failed on first attempt — data not generated before training was scheduled (workflow dependency issue, now fixed). Needs rerun.
2. **Experiment 18 (v2 clean rewrite)**: `scripts/zarr_to_methyl_memmap_v2.py` mirrors the unmodified legacy extraction logic exactly (forward-strand CpGs only, explicit reverse indexing for ri/rp, writes paired fwd+rev views per CpG). End-to-end tests at `tests/test_zarr_to_methyl_memmap_v2.py`. Not yet run.

#### Remaining hypotheses
1. **Per-feature normalization stats contamination.** Legacy normalizes fi, fp, ri, rp each with their OWN mean/std (via `compute_log_normalization_stats`). ZNorm computes SHARED stats over mixed fi+ri in column 1 and fp+rp in column 2. If fi and ri have different distributions, neither is correctly centered under ZNorm. Experiment 16 tested log1p but still used mixed stats, so this was never actually tested. Experiment 17 (pure fi/fp, no mixing) would test this indirectly.
2. **Kinetics mixing in model input columns.** Legacy always feeds fi/fp into model columns 1-2 for forward views and ri/rp for reverse views — but each is normalized with its own feature-specific stats. New pipeline mixes fi/fp and ri/rp into the same columns with shared normalization, giving the kin_embed layer a bimodal distribution to handle. Experiment 17 tests this directly.
3. **Training data composition.** Legacy extracts CpGs from the forward strand only, then yields two views of each CpG (fwd: fi/fp, rev: flip(ri)/flip(rp)) via single_strand=True. The new v1 pipeline extracts CpGs from BOTH forward and reverse strands independently. Same total window count, but legacy has paired views of the same CpG sites while v1 has single views of different sites. Paired views are a form of data augmentation where the model sees the same genomic position from both kinetics perspectives.
4. **Subtle preprocessing bugs in zarr_to_methyl_memmap.py.** The original script uses np.flip on the entire read, stride tricks, and complex index mapping (`list(set(...))`). The v2 rewrite avoids all of this. Experiment 18 tests whether matching the legacy logic exactly closes the gap.
5. **fi vs ri distribution mismatch.** If fi and ri have meaningfully different distributions at CpG sites, the mixed normalization and mixed model inputs compound the problem. Diagnostic test at `tests/test_kinetics_distributions.py` (requires BAM files, can't run on Gefion where BAMs are mocked).