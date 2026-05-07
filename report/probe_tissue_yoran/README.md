# Tissue probe sweep — yoran ctx4096

Model-independent diagnostics for the `supervised_52_tissue` experiment. The
deep classifier drives train cross-entropy to ~0 on every model size
(d128/d256/d512/d768) but val loss never moves. This sweep tests whether
there is any extractable tissue signal in the kinetics features at all,
whether the cell-batch effect dominates, and whether the preprocessing
itself is to blame.

## Running

Each script is standalone and runs from inside the project `.venv` on a
GenomeDK interactive node. They share a small loader (`_shared.py`); each
probe materialises a downsampled subset of `train`, `val_s1`, and `val_s2`
(default 10k / 2k / 2k after the centre-crop to 2048), applies the same
`KineticsNorm(n_continuous=4)` the deep model uses, and writes its own
results to `results/` and `figures/`.

```bash
# Data integrity / EDA
python -m report.probe_tissue_yoran.eda_partition_sanity
python -m report.probe_tissue_yoran.eda_reverse_kinetics_alignment
python -m report.probe_tissue_yoran.eda_kinetics_distributions
python -m report.probe_tissue_yoran.eda_pca_umap

# Tissue probes — main matrix
python -m report.probe_tissue_yoran.probe_tissue_pooled
python -m report.probe_tissue_yoran.probe_tissue_binned
python -m report.probe_tissue_yoran.probe_tissue_flat
python -m report.probe_tissue_yoran.probe_per_tissue_one_vs_rest

# Diagnostics
python -m report.probe_tissue_yoran.probe_cell_s1_vs_s2
python -m report.probe_tissue_yoran.probe_readlength_only
python -m report.probe_tissue_yoran.probe_seq_only
python -m report.probe_tissue_yoran.probe_raw_vs_normed
python -m report.probe_tissue_yoran.probe_context_4096
```

A pytest smoke test at `tests/test_probe_tissue_shared.py` exercises the
loader against the live dataset; it auto-skips when the data dir is absent.

## What each script tests

| Script | Hypothesis tested | Output |
|---|---|---|
| `eda_partition_sanity` | partition.csv is disjoint, stratified, cell-mapped | `results/partition_sanity.csv` |
| `eda_reverse_kinetics_alignment` | `ri[::-1]` alignment in `bam_to_labeled_memmap.py` is correct (no v1-style misalignment) | `figures/reverse_kinetics_alignment_*.svg`, `results/reverse_kinetics_alignment_summary.csv` |
| `eda_kinetics_distributions` | per-channel densities are tissue- vs cell-shaped | `figures/kinetics_density_by_*.svg`, `figures/kinetics_mean_heatmap.svg` |
| `eda_pca_umap` | 2D projection clusters by tissue or by cell | `figures/{pca,umap}_pooled_by_*.svg` |
| `probe_tissue_pooled` | 25 pooled summary features → 8-way LogReg/RF/HGB | `results/probe_tissue_pooled.csv` |
| `probe_tissue_binned` | 128 16-bin features → same classifiers | `results/probe_tissue_binned.csv` |
| `probe_tissue_flat` | 8192 flattened raw kinetics → LogReg L2 saga | `results/probe_tissue_flat.csv` |
| `probe_per_tissue_one_vs_rest` | any single tissue separable in 1-vs-rest? | `results/probe_per_tissue_one_vs_rest.csv`, `figures/per_tissue_auroc.svg` |
| `probe_cell_s1_vs_s2` | cell s1 vs s2 separable from pooled features (batch-effect headline) | `results/probe_cell_s1_vs_s2.csv` |
| `probe_readlength_only` | tissue label leaks through read length | `results/probe_readlength_only.csv` |
| `probe_seq_only` | tissue label separable from base composition only | `results/probe_seq_only.csv` |
| `probe_raw_vs_normed` | KineticsNorm destroys signal | `results/probe_raw_vs_normed.csv` |
| `probe_context_4096` | centre-crop to 2048 throws away signal | `results/probe_context_4096.csv` |

The combined probe matrix from the three main probes lands in
`results/probe_matrix.csv`.

## Headline findings

Run on a 10k train / 2k val_s1 / 2k val_s2 subsample. Chance accuracy is
1/8 = 0.125. Cross-entropy floor is `ln(8) ≈ 2.08`.

### Tissue probe matrix (`results/probe_matrix.csv`)

| Representation | Classifier | train top1 | val_s1 top1 | val_s2 top1 |
|----------------|------------|-----------:|------------:|------------:|
| pooled_summary (25-d)     | LogReg | 0.147 | 0.129 | 0.136 |
| pooled_summary            | RandomForest | **1.000** | 0.137 | 0.130 |
| pooled_summary            | HistGB | 0.998 | 0.132 | 0.128 |
| binned_summary (128-d)    | LogReg | 0.193 | 0.132 | 0.131 |
| binned_summary            | RandomForest | **1.000** | 0.137 | 0.132 |
| binned_summary            | HistGB | **1.000** | 0.129 | 0.130 |
| flat_kinetics (8192-d)    | LogReg L2 saga | **1.000** | 0.134 | 0.123 |

Every classifier on every kinetics-summary representation collapses to
chance on validation. RF and HGB perfectly memorise the training set
without learning anything that transfers. This reproduces the deep-model
pathology one-for-one: train cross-entropy → 0, val cross-entropy stuck at
the chance floor.

### Diagnostics

- **Read length only**: train 0.141, val_s1 0.142, val_s2 0.148. Length
  carries 1.4 - 1.5 percentage points above chance. Negligible
  length-driven selection bias.
- **Sequence composition only** ({A,C,G,T,N} frequencies): LogReg train
  0.132, val_s1 0.142; RandomForest memorises (1.000) but val_s1 = 0.114
  (below chance). No tissue signal in base composition either.
- **Cell s1 vs s2 (pooled features)**: LogReg val accuracy 0.549 (AUROC
  0.555); RandomForest 0.541 / 0.545. The cell-batch effect is real but
  weak — and not strong enough on its own to explain why the deep model
  fails to generalise. The implication: the kinetics summaries we use here
  carry roughly *equally weak* tissue and cell signal.
- **Per-tissue 1-vs-rest** (pooled features): all 8 binary AUROCs lie in
  [0.50, 0.57] on val_s1 and val_s2. No single tissue is meaningfully
  separable.
- **Raw vs normalised pooled features** (HistGB): val accuracies are
  identical to two decimals (raw val_s1 0.131, normed val_s1 0.132).
  `KineticsNorm` is *not* destroying signal — there is no signal to
  destroy at this representation.
- **ctx=2048 (cropped) vs ctx=4096 (full window)** (HistGB on pooled
  summary): val_s1 0.132 (cropped) vs 0.121 (full); val_s2 0.128 vs
  0.116. The full window is *worse*, not better. The deep model's
  centre-crop is not throwing away useful signal — if anything, the
  dropped ends carry per-window noise.

### Where the signal sits in the means

`results/kinetics_means_by_tissue_cell.csv`: per-(tissue, cell, channel)
mean of normalised features. The cell-shift in `fi` is ~0.06 z-score units
(cell s2 mean ≈ 0.08 vs cell s1 mean ≈ 0.01). Within each cell, tissues
span ≈ 0.06 - 0.07 z-score units. The cell shift and the within-cell
between-tissue variation are the same order of magnitude, so a global
z-score (which is what `KineticsNorm` does) leaves the cell shift in the
data and the per-tissue signal is buried beneath comparable-magnitude
batch noise.

### Read this as

1. The deep model's overfitting is not a bug in training — it is a
   property of the dataset: pooled / binned kinetics + base composition
   carry no extractable tissue signal at this subset size.
2. The reverse-kinetics alignment is correct (per-base symmetry test
   confirms `fp at fwd-C ≈ rp at fwd-G` etc.); the build script's
   `ri[::-1]` step is doing what the docstring claims.
3. The cell-batch effect is real but small. It is not the dominant
   blocker; raw kinetics alone don't carry tissue information either.
4. Things still to try if we keep this task alive: per-cell normalisation
   (subtract per-cell channel means before the global z-score), much
   larger labelled sets, sequence-aware features (k-mer composition
   beyond base frequency), tracking *whether the read mapped to a
   tissue-marker region* rather than relying on whole-read kinetics.

### Anomalies surfaced by the probes (so far)

These were not the goal of the sweep but came out of running it:

1. **`partition.csv` has one row with `'val_s1 '` (trailing space)**.
   `_shared.load_partition()` strips whitespace by default and warns. Fix at
   the source by re-running `scripts/make_tissue_partition.py` once it
   filters trailing whitespace; the on-disk file currently silently drops
   one read from any string-equality filter on `split == 'val_s1'`.

2. **The manifest emits multiple windows for some reads.** 580 read_names
   have >1 manifest row (max 4). When the deep model filters via
   `pl.col('read_name').is_in(train_names)`, it picks up all matching rows,
   so the train split is +18 rows above the declared 50,000 (val_s1 +3,
   val_s2 +6). Probably negligible for training but means the on-disk
   "50,000 train" claim is approximate. Detected by
   `eda_partition_sanity.py` and reported in `results/manifest_inflation.csv`.

3. **Reverse-kinetics alignment is correct** despite a counter-intuitive
   Pearson signal. `r(fi[j], ri[j])` averaged over j is 0.148 and
   `r(fi[j], ri[T-1-j])` is 0.155 — nearly identical. This initially looks
   like alignment is wrong, but the per-base mean test
   (`figures/reverse_kinetics_alignment_per_base.svg`) confirms alignment
   is correct: `fp at fwd-C ≈ rp at fwd-G` and `fp at fwd-A ≈ rp at fwd-T`
   to within 0.1 uint8 unit. The Pearson floor comes from per-read overall
   rate effects (some reads are kinetically slower than others),
   independent of position. Position-aware tests would need to subtract
   per-read mean before computing correlations.
