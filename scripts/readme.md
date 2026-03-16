# Scripts

Entry points for data processing and training. All scripts are invoked as modules (`python -m scripts.<name>`) from the project root or via gwf targets defined in `workflow.py`.

Configuration lives in `configs/`. The data pipeline reads from `configs/data.yaml`; training scripts read from `configs/ssl.yaml` or `configs/supervised.yaml`.

## Pipeline overview

```
BAM  ──►  Zarr  ──►  memmap shards  ──►  DataLoader  ──►  Training
       ①         ②                                      ④
                  ├── ②a (SSL, full reads)
                  └── ②b (CpG, windowed)
```

---

## Data processing

### `bam_to_zarr.py`

Converts a PacBio BAM into a Zarr store. Extracts the nucleotide sequence (tokenised via `config.data.token_map`), base quality, and per-base kinetics tags (`fi`, `fp`, `ri`, `rp`, plus any optional tags). Data is stored in a CSR-like layout: `data` is `(total_bases, n_features)` uint8, `indptr` marks read boundaries.

Validates that requested tags exist in the first N reads before committing to the full conversion. Supports line-profiling via the `TimeLINE_PROFILE` env var.

```
--input_path   BAM file
--output_path  Zarr directory
--config       configs/data.yaml
--n_reads      Number of reads (0 = all)
--optional_tags  Extra per-base tags beyond the kinetics set
```

### `zarr_to_memmap_instanceNorm.py`

Converts a Zarr store into sharded `.npy` memmap files for **self-supervised pretraining**. Splits each read into fixed-length segments of `context` bases, writing forward and reverse strand segments as separate samples. Each sample has a mask channel (1.0 = padding, 0.0 = real data).

Supports per-read MAD normalisation (`--normalize`), reverse-complement mapping (`--reverse_complement`), and quality filtering (`--filter_qual`). Output consumed by `ShardedMemmapDataset`.

```
--input_path    Zarr store
--output_path   Output directory for .npy shards
--config_path   configs/data.yaml
--context       Segment length (default 4096)
--shard_size    Samples per shard (default 16384)
--fwd_features / --rev_features  Feature columns per strand
```

### `zarr_to_methyl_memmap.py`

Converts a Zarr store into sharded `.npy` files for **CpG methylation classification**. Instead of fixed segments, extracts windows centred on CpG dinucleotide sites using `extract_pattern_windows_2d` (sliding-window pattern match). Produces one sample per CpG site per strand direction. Output is split into `train/` and `val/` directories and consumed by `LabeledMemmapDataset`.

Pipeline parameters (context, features) should be kept in sync with `configs/data.yaml` under `cpg_pipeline`.

Same CLI interface as `zarr_to_memmap_instanceNorm.py`. Key differences: CpG-centred windowing, train/val split at the read level.

**Known issues:**
- CLI default `--shard_size` is `int(2e20)` (will OOM) — always override via workflow or CLI.
- Train/val split is non-deterministic (no seed).
- Schema labels all samples with forward feature names, but reverse-strand samples contain reverse features.

### `inject_norm_stats.py`

Computes global log-space mean/std statistics over a Zarr store by sampling chunks in parallel (uses JAX for the computation). Writes the stats into the Zarr's `attrs['log_norm']` so downstream consumers can access them without a separate file.

```
--input_path     Zarr store (modified in place)
--chunk_stride   Skip every N chunks
--idx_stride     Skip every N rows within a chunk
--num_threads    Thread pool size
```

### `validate_memmap_instanceNorm.py`

Sanity checks a memmap shard directory. Loads `shard_00000.npy` and verifies:
1. The `seq` column contains discrete integer tokens (not normalised floats).
2. Forward/reverse sample pairs are valid reverse-complements of each other.
3. Padded regions contain only zeros in feature columns.

Writes a `validation.log` to the memmap directory.

```
--input_path   Memmap directory (must contain schema.json and shard_00000.npy)
--config_path  configs/data.yaml
```

### `validate_zarr.py`

Pytest-based tests for `bam_to_zarr`. Verifies that a round-trip through `bam_to_zarr` preserves sequences, quality scores, and kinetics tags by comparing the Zarr output back to the source BAM read-by-read.

Run with: `python -m pytest scripts/validate_zarr.py`

---

## Training

### `train_ssl.py`

Self-supervised pretraining using masked prediction (InfoNCE / AgInfoNCE loss). Loads data via `ShardedMemmapDataset` and trains `Smrt2Vec`. Uses HuggingFace Accelerate for multi-GPU, TensorBoard for logging. Config from `configs/ssl.yaml`.

```
python -m scripts.train_ssl configs/ssl.yaml
# or via accelerate:
accelerate launch --num_processes=8 scripts/train_ssl.py configs/ssl.yaml
```

### `train_supervised.py`

Binary CpG methylation classifier. Loads pos/neg memmap shards via `LabeledMemmapDataset`, trains `DirectClassifier` with BCE loss. Evaluates F1, AUROC, AUPRC, accuracy on a validation split each epoch. Config from `configs/supervised.yaml`.

```
accelerate launch --num_processes=1 scripts/train_supervised.py configs/supervised.yaml
```

### `train_supervised_legacy.py`

Same task as `train_supervised.py` but uses `LegacyMethylDataset` (parquet-based, log-normalised) instead of the memmap pipeline. Kept for comparison against the legacy data format. Uses the `legacy_train` / `legacy_val` paths from the supervised config.
