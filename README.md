# smrt-foundation
A foundation model for pacbio SMRT reads, providing a native understanding of kinetics wrt nucleotide context.


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