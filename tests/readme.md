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

## Adding new tests

- Place new test modules in this directory, named `test_*.py`. They are automatically picked up by the gwf workflow &mdash; no changes to `workflow.py` needed.
- Reuse session fixtures from `conftest.py` if one is added, or define scoped fixtures in the module as done in `test_cpg_pipeline_fidelity.py`.
- Draw pipeline parameters from `configs/data.yaml` rather than hardcoding them, so tests stay correct when configuration changes.
- Prefer testing against the real sample BAMs rather than synthetic data where feasible &mdash; the point is to catch pipeline bugs that only surface with real PacBio data.
