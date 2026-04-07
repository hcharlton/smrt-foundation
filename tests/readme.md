# Tests

Automated tests for the smrt-foundation data processing and training pipelines. All tests use [pytest](https://docs.pytest.org/) and can be run from the project root.

## Running tests

```bash
# all tests
python -m pytest tests/ -v

# a single module
python -m pytest tests/test_cpg_pipeline_fidelity.py -v

# a single class or test
python -m pytest tests/test_cpg_pipeline_fidelity.py::TestBamToZarr -v
python -m pytest tests/test_cpg_pipeline_fidelity.py::TestEndToEndFidelity::test_bam_cpg_features_in_shards -v
```

Tests can also be submitted as cluster jobs via gwf. Each `test_*.py` file gets its own target automatically (see the Testing section in `workflow.py`).

## Test data

Tests look for BAM files in `data/00_raw/labeled/`, trying subset files first for speed and falling back to full files:

| Positive (methylated) | Negative (unmethylated) |
|---|---|
| `methylated_subset.bam` | `unmethylated_subset.bam` |
| `methylated_hifi_reads.bam` | `unmethylated_hifi_reads.bam` |

The first file found (in the order above) is used. If neither exists the test session is skipped. Intermediate artifacts (Zarr stores, memmap shards) are created in temporary directories managed by pytest and cleaned up automatically.

## Configuration

All pipeline parameters are read from `configs/data.yaml` so that tests stay in sync with the production pipeline:

- **`data`** section &mdash; `token_map`, `rc_map`, `kinetics_features` (used by `bam_to_zarr` and the tests)
- **`cpg_pipeline`** section &mdash; `context`, `fwd_features`, `rev_features`, `shard_size` (used by `zarr_to_methyl_memmap` and the tests)

If you change a pipeline parameter (e.g. window context, feature set), update `configs/data.yaml` and the tests will automatically use the new values.

---

## Test modules

### `test_cpg_pipeline_fidelity.py`

End-to-end fidelity tests for the BAM &rarr; Zarr &rarr; CpG memmap pipeline. The core question: are the kinetics features surrounding every CpG site in the original BAM faithfully transferred into the final sharded numpy output?

#### Session fixtures

The pipeline stages are expensive relative to individual assertions, so they run once per session and are shared across all test classes:

| Fixture | What it does |
|---|---|
| `config` | Loads `configs/data.yaml` |
| `cpg_params` | The `cpg_pipeline` section from config (context, features, shard_size) |
| `context` | Window size from config |
| `fwd_features` / `rev_features` | Feature lists from config |
| `n_output_features` | Computed as `len(fwd_features) + 1` (features + mask channel) |
| `sample_bam` / `sample_bam_neg` | Paths to methylated / unmethylated BAMs (auto-discovered) |
| `zarr_dir` / `zarr_dir_neg` | Runs `bam_to_zarr` on the first 50 reads &rarr; temp Zarr stores |
| `memmap_dir_raw` | Runs `zarr_to_sharded_memmap` without normalization &rarr; temp shards |
| `memmap_dir_norm` | Runs `zarr_to_sharded_memmap` with MAD normalization &rarr; temp shards |
| `memmap_dir_neg_raw` | Same as `memmap_dir_raw` for the negative BAM |

#### Test classes

**`TestBamToZarr`** &mdash; Stage 1 (BAM &rarr; Zarr)

| Test | Verifies |
|---|---|
| `test_zarr_has_expected_arrays` | Zarr store contains `data`, `indptr`, and `features` attr |
| `test_read_count_matches` | Number of reads in Zarr equals number of valid reads in BAM |
| `test_sequence_fidelity` | Every base token in the Zarr `seq` column matches the BAM query sequence |
| `test_kinetics_fidelity` | Every kinetics tag value (`fi`, `fp`, `ri`, `rp`) matches the BAM per-base tag arrays exactly |

**`TestZarrToMemmapRaw`** &mdash; Stage 2 (Zarr &rarr; sharded memmap, no normalization)

| Test | Verifies |
|---|---|
| `test_shards_exist` | At least one shard was written across train/val |
| `test_schema_written` | Each split directory has a `schema.json` with a `features` list ending in `mask` |
| `test_shard_shape` | Every shard is 3-D with context and feature count matching config |
| `test_mask_channel_is_zero_for_data` | The mask/padding channel is `0.0` at window centres |
| `test_cpg_at_window_centre` | Forward-strand windows have the CG dinucleotide centred at the expected position |
| `test_raw_values_match_zarr` | Reconstructs all CpG windows (forward + reverse) from the Zarr and asserts byte-exact equality with the shard contents |

**`TestZarrToMemmapNormalized`** &mdash; Stage 2 with MAD normalization

| Test | Verifies |
|---|---|
| `test_shards_exist` | Shards produced with normalization enabled |
| `test_normalized_kinetics_differ_from_raw` | Kinetics columns actually change after normalization (guards against no-op normalization) |
| `test_sequence_tokens_unchanged_by_normalization` | Sequence tokens are categorical and must not be altered by normalization |

**`TestEndToEndFidelity`** &mdash; Stage 3 (BAM &rarr; memmap, skipping Zarr as intermediary)

| Test | Verifies |
|---|---|
| `test_bam_cpg_features_in_shards` | Extracts CpG windows directly from BAM tag data for both strands and verifies byte-exact match against shard output. This is the highest-confidence fidelity check &mdash; it bypasses the Zarr entirely and compares source-of-truth BAM values to final output. |

**`TestExtractPatternWindows`** &mdash; Unit tests for `extract_pattern_windows_2d`

| Test | Verifies |
|---|---|
| `test_basic_extraction` | Single CpG in a minimal read produces one correctly-shaped window |
| `test_no_match` | A read with no CpG sites returns an empty array |
| `test_boundary_exclusion` | CpG sites too close to read edges are correctly excluded |
| `test_multiple_matches` | Multiple CpG sites each produce a window with CG centred |

**`TestFullPipelineDataLoader`** &mdash; Stage 4 (BAM &rarr; Zarr &rarr; memmap &rarr; `LabeledMemmapDataset` &rarr; `DataLoader`)

Runs the entire pipeline end-to-end for both the methylated (positive) and unmethylated (negative) BAMs, loads the resulting shards through `LabeledMemmapDataset` and `DataLoader` &mdash; the same interface used by training &mdash; and verifies that the data retrieved matches the source BAMs.

| Test | Verifies |
|---|---|
| `test_dataloader_labels` | Positive samples have label `1.0`, negative samples have label `0.0`, matching the `LabeledMemmapDataset` contract |
| `test_dataloader_shape` | Batches from the DataLoader have shape `(batch, context, n_output_features)` and `float32` dtype, where dimensions come from config |
| `test_positive_samples_match_bam` | Every positive sample pulled from the DataLoader exists in the set of CpG windows extracted directly from the methylated BAM |
| `test_negative_samples_match_bam` | Every negative sample pulled from the DataLoader exists in the set of CpG windows extracted directly from the unmethylated BAM |
| `test_full_batch_combined_counts` | Total samples across train + val splits (both classes) equals the total number of CpG windows from both BAMs &mdash; no windows are lost or duplicated by the pipeline |

---

### `test_legacy_vs_new_pipeline.py`

Direct comparison tests between the legacy parquet pipeline (`archive/make_legacy_labeled_dataset.py` &rarr; `LegacyMethylDataset`) and the new zarr-to-memmap pipeline (`bam_to_zarr` &rarr; `zarr_to_methyl_memmap` &rarr; `LabeledMemmapDataset`). Traces the same BAM reads through both paths and compares what the model actually receives as input tensors.

#### Session fixtures

| Fixture | What it does |
|---|---|
| `legacy_pos_parquet` / `legacy_neg_parquet` | Runs `bam_to_legacy_parquet` on the first 20 reads &rarr; temp parquet files |
| `zarr_pos` / `zarr_neg` | Runs `bam_to_zarr` on the first 20 reads &rarr; temp Zarr stores |
| `memmap_pos_raw` / `memmap_neg_raw` | New pipeline shards (raw, with RC) |
| `memmap_pos_norm` | New pipeline shards (MAD-normalized, with RC) |

#### Test classes

**`TestSingleSampleProvenance`** &mdash; Traces a single CpG site from one BAM read through both pipelines

| Test | Verifies |
|---|---|
| `test_forward_strand_raw_values_match` | Raw kinetics at the same CpG site match between legacy parquet and BAM source |
| `test_reverse_strand_sequence_handling` | Legacy reverse windows have correctly reverse-complemented sequence |

**`TestNormalizationComparison`** &mdash; Compares the three normalization strategies

| Test | Verifies |
|---|---|
| `test_raw_kinetics_distribution` | Raw pipeline values are in uint8 range (0&ndash;255) |
| `test_mad_normalized_distribution` | MAD-normalized values are centered near 0 |
| `test_legacy_log_z_normalization` | Legacy log-Z values are approximately standard-normal |
| `test_normalization_strategy_changes_discrimination` | Prints class separation metrics for each normalization (diagnostic) |

**`TestFeatureColumnAlignment`** &mdash; How features map to model input positions

| Test | Verifies |
|---|---|
| `test_new_pipeline_mixes_fwd_rev_kinetics` | New pipeline puts fwd/rev kinetics in the same columns |
| `test_legacy_separates_fwd_rev_kinetics` | Legacy parquet stores all four kinetics columns (fi, fp, ri, rp) separately |

**`TestModelInputTensorComparison`** &mdash; Actual tensors the model receives

| Test | Verifies |
|---|---|
| `test_new_pipeline_tensor_shape` | Shape, dtype, and feature layout from `LabeledMemmapDataset` |
| `test_legacy_tensor_shape` | Shape and dtype from `LegacyMethylDataset` |
| `test_kinetics_scale_comparison` | Prints and asserts the dramatic scale difference between raw (0&ndash;255) and legacy log-Z (~standard normal) kinetics |

**`TestReverseStrandConsistency`** &mdash; Reverse strand handling

| Test | Verifies |
|---|---|
| `test_legacy_reverse_has_rc_sequence` | Legacy reverse windows have CG at center after RC |
| `test_new_pipeline_reverse_with_rc` | New pipeline (use_rc=True) also has CG at center for all windows |
| `test_new_pipeline_reverse_kinetics_are_flipped` | Reverse kinetics are correctly flipped |

**`TestWindowCountComparison`** &mdash; Window extraction consistency

| Test | Verifies |
|---|---|
| `test_same_total_cpg_windows` | Both pipelines produce the same number of CpG windows |
| `test_same_reads_contribute` | Same set of reads contribute windows in both pipelines |

**`TestDiagnosticSummary`** &mdash; Prints a full diagnostic comparison (always passes)

| Test | What it prints |
|---|---|
| `test_print_pipeline_comparison` | Sample counts, feature layouts, kinetics distributions, normalization strategies, and reverse-strand handling for all three paths (raw, MAD-norm, legacy log-Z) |

---

### `test_kinetics_norm.py`

Equivalence tests verifying that `KineticsNorm` produces identical statistics and normalized outputs as the original `ZNorm` (on labeled data) and `SSLNorm` (on SSL data). Ensures the unified normalization class is a drop-in replacement.

#### Session fixtures

| Fixture | What it does |
|---|---|
| `labeled_ds` | `LabeledMemmapDataset` with 10,000 samples from CpG subset memmaps |
| `ssl_ds` | `ShardedMemmapDataset` with 10,000 samples from SSL subset memmaps |

#### Test classes

**`TestZNormEquivalence`** &mdash; KineticsNorm matches ZNorm on LabeledMemmapDataset

| Test | Verifies |
|---|---|
| `test_statistics_match` | KineticsNorm means/stds for kinetics channels (cols 1, 2) match ZNorm within atol=1e-4 |
| `test_batch_output_identical` | Normalized tensor output is identical to ZNorm output within atol=1e-5 |
| `test_multiple_samples` | Equivalence holds across 50 different samples |

**`TestSSLNormEquivalence`** &mdash; KineticsNorm matches SSLNorm on ShardedMemmapDataset

| Test | Verifies |
|---|---|
| `test_statistics_match` | KineticsNorm means/stds match SSLNorm within atol=1e-4 |
| `test_batch_output_identical` | Normalized output matches SSLNorm within atol=1e-5 |
| `test_multiple_samples` | Equivalence holds across 50 samples |

**`TestKineticsNormProperties`** &mdash; Internal behavior checks

| Test | Verifies |
|---|---|
| `test_seq_and_mask_unchanged` | Columns 0 (sequence) and 3 (mask) are not modified by normalization |
| `test_kinetics_are_modified` | Columns 1 and 2 (IPD, PW) are modified |
| `test_no_log_transform_mode` | z-score-only mode (log_transform=False) still modifies kinetics |

---

### `test_ssl_21_train.py`

Static AST-based analysis of `ssl_21_pretrain/train.py`. Parses the training script source to verify that the linear probe evaluation runs on all ranks (not gated behind `is_main_process`) and that `wait_for_everyone()` synchronizes ranks between probe and next epoch. These guards prevent NCCL timeouts from rank desynchronization.

#### Module fixtures

| Fixture | What it does |
|---|---|
| `train_source` | Raw source code of train.py as string |
| `train_ast` | Parsed AST of train.py |

#### Test classes

**`TestLinearProbeNotRankGated`**

| Test | Verifies |
|---|---|
| `test_probe_call_not_inside_is_main_process_guard` | `linear_probe_eval()` is not inside an `if is_main_process` block (single-rank probe causes RNG divergence and NCCL timeout at next epoch) |

**`TestBarrierAfterProbe`**

| Test | Verifies |
|---|---|
| `test_wait_for_everyone_after_probe` | `wait_for_everyone()` exists between `linear_probe_eval()` and the next `model.train()` call |

---

### `test_autoencoder.py`

Unit tests for `SmrtAutoencoder`, `SmrtDecoder`, and `MaskedReconstructionLoss`. Uses synthetic inputs (no data files required).

#### Test classes

**`TestSmrtDecoder`** &mdash; Decoder output dimensions

| Test | Verifies |
|---|---|
| `test_output_shape` | Input `(B, T/4, d)` produces output `(B, T, 2)` |
| `test_different_lengths` | Correct shapes across context lengths (32, 64, 128, 256) |

**`TestSmrtAutoencoder`** &mdash; Forward pass, masking, gradients

| Test | Verifies |
|---|---|
| `test_forward_shapes` | Returns kin_recon `(B, T, 2)`, kin_target `(B, T, 2)`, mask `(B, T)` |
| `test_masking_zeros_kinetics` | `apply_input_mask()` zeros kinetics at masked positions |
| `test_masking_preserves_sequence` | Sequence tokens (col 0) and pad mask (col 3) unchanged after masking |
| `test_masking_produces_masked_positions` | Masking actually creates True values in the mask tensor |
| `test_gradient_flows` | Loss backward propagates gradients to both encoder and decoder |

**`TestMaskedReconstructionLoss`** &mdash; Loss function correctness

| Test | Verifies |
|---|---|
| `test_basic` | Produces scalar > 0 on random input |
| `test_perfect_reconstruction` | Loss < 1e-6 when recon equals target |
| `test_only_masked_positions` | Loss computed only at masked positions, not everywhere |

**`TestEncoderCompatibility`** &mdash; Weight transfer between models

| Test | Verifies |
|---|---|
| `test_encoder_weights_transfer` | Autoencoder encoder weights load into DirectClassifier via `load_state_dict()` (only positional encoding buffer size may differ) |

---

### `test_large_pretrain.py`

Tests for experiment 29 features: random cropping augmentation, cosine LR scheduling over 3000 epochs, and periodic probe/checkpoint logic. Uses synthetic data and scheduler objects (no cluster or GPU required).

#### Test classes

**`TestRandomCropping`** &mdash; Random crop augmentation

| Test | Verifies |
|---|---|
| `test_returns_correct_shape` | Cropping `(4096, 4)` with context=128 produces `(128, 4)` |
| `test_crops_differ_across_calls` | 10 crops from same sample have at least 2 different start positions |
| `test_crop_stays_in_bounds` | 100 crops never exceed input length |
| `test_short_input_fallback` | Input shorter than context falls back gracefully |

**`TestLRSchedulerLongTraining`** &mdash; Cosine schedule with low warmup

| Test | Verifies |
|---|---|
| `test_lr_starts_near_zero` | LR < 1e-6 at step 1 |
| `test_lr_peaks_after_warmup` | LR = max_lr at end of warmup phase (1% of total steps) |
| `test_lr_decays_at_midpoint` | LR between floor and max at midpoint |
| `test_lr_reaches_floor_at_end` | Final LR = max_lr * 0.05 |
| `test_lr_never_negative` | LR >= 0 at sampled checkpoints throughout training |

**`TestProbeFrequency`** &mdash; Periodic evaluation and checkpointing

| Test | Verifies |
|---|---|
| `test_probe_runs_at_correct_intervals` | Probe fires at correct epoch multiples |
| `test_checkpoint_runs_at_correct_intervals` | Checkpoint saves at correct epoch multiples |

---

### `test_kinetics_distributions.py`

Compares forward vs reverse kinetics distributions (fi vs ri, fp vs rp) at CpG sites using Kolmogorov-Smirnov tests. Identifies whether mixing forward and reverse kinetics in the same model input columns forces the embedding to handle two different distributions. Requires BAM files on disk; skips if not found.

Can also run standalone: `python tests/test_kinetics_distributions.py [max_reads]`

| Test | Verifies |
|---|---|
| `test_kinetics_distributions` | Runs distribution comparison for methylated and unmethylated BAMs, prints KS statistics and summary |

---

### `test_zarr_to_methyl_memmap_v2.py`

End-to-end integration tests for the v2 CpG memmap pipeline: Zarr &rarr; shards &rarr; `LabeledMemmapDataset` &rarr; `DataLoader`. Requires Zarr stores on disk; skips if not found.

#### Session fixtures

| Fixture | What it does |
|---|---|
| `config` | Loads `configs/data.yaml` |
| `pos_zarr` / `neg_zarr` | Paths to positive/negative Zarr stores (auto-discovered) |
| `pos_shards` / `neg_shards` | Runs `zarr_to_methyl_memmap_v2` on the Zarr stores into temp directories |

#### Test classes

**`TestCpgExtraction`** &mdash; CpG window identification

| Test | Verifies |
|---|---|
| `test_all_windows_have_cg_at_center` | All windows have C at center, G at center+1 |
| `test_window_count_matches_zarr` | Total windows = 2x CpG sites in Zarr (forward + reverse) |
| `test_forward_and_reverse_alternate` | Windows written in fwd/rev pairs; both have CG at center |

**`TestKineticsAlignment`** &mdash; Kinetics values match Zarr source

| Test | Verifies |
|---|---|
| `test_forward_kinetics_match_zarr` | Forward window fi/fp match Zarr values at corresponding positions |
| `test_reverse_kinetics_match_zarr` | Reverse window ri/rp match Zarr with correct reverse indexing |
| `test_reverse_seq_is_rc_of_forward` | Reverse sequence is reverse complement of forward |

**`TestShardIntegrity`** &mdash; Shard format and metadata

| Test | Verifies |
|---|---|
| `test_shard_shape` | All shards are `(N, 32, 4)` |
| `test_mask_is_zero` | Mask channel is all 0.0 (no padding in CpG windows) |
| `test_no_nan_inf` | All values are finite |
| `test_seq_tokens_valid` | Sequence tokens in range [0, 4] |
| `test_schema_written` | `schema.json` exists in train/ and val/ |
| `test_train_val_no_read_overlap` | Train + val counts sum to total (no duplication) |

**`TestDataLoaderIntegration`** &mdash; Dataset and DataLoader loading

| Test | Verifies |
|---|---|
| `test_dataset_loads` | `LabeledMemmapDataset` instantiates, len > 0 |
| `test_sample_shape` | Samples are `(32, 4)` float32; labels are scalar in {0.0, 1.0} |
| `test_model_input_channels` | Channel layout: col 0 = seq tokens, cols 1-2 = kinetics, col 3 = mask |
| `test_positive_label_is_one` | Positive samples have label 1.0 |
| `test_negative_label_is_zero` | Negative samples have label 0.0 |

---

### `test_legacy_leakage.py`

Checks whether the legacy parquet train/test split leaks windows from the same reads across splits. Read-level overlap would inflate evaluation metrics because CpG sites on the same read share kinetics context. Requires legacy parquet files on disk; skips if not found.

Can also run standalone: `python tests/test_legacy_leakage.py [train_path] [test_path]`

| Test | Verifies |
|---|---|
| `test_no_leakage` | Zero read_name overlap between train and test parquet files |

---

## Adding new tests

- Place new test modules in this directory, named `test_*.py`. They are automatically picked up by the gwf workflow &mdash; no changes to `workflow.py` needed.
- Reuse session fixtures from `conftest.py` if one is added, or define scoped fixtures in the module as done in `test_cpg_pipeline_fidelity.py`.
- Draw pipeline parameters from `configs/data.yaml` rather than hardcoding them, so tests stay correct when configuration changes.
- Prefer testing against the real sample BAMs rather than synthetic data where feasible &mdash; the point is to catch pipeline bugs that only surface with real PacBio data.
