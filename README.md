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
    1.2. ISSUE: now the direct training is significantly worse (max around 70 percent top1 wheras the legacy version was 81/82). Need to identify the bug (this is a priority)
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