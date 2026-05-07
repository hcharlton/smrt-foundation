# smrt-foundation

A foundation model for PacBio SMRT reads. The ambition is a downstream classifier built on a pretrained encoder that beats one trained from random init by some meaningful margin, especially in the label-scarce regime where SSL is supposed to help. Direct analog of wav2vec 2.0 for kinetics, minus the quantization module. The supervised baseline at full data sits at about 82% top-1 on single-strand CpG methylation in a 32 bp window (`supervised_20`). The autoencoder lineage is the SSL family that has produced positive transfer over random init; the active comparison is whether that lift translates into a beat in the small-n regime, where pretraining is meant to earn its keep.

## What we're working with

PacBio HiFi reads come with two channels per base that a normal sequencing read does not have: pulse width (`pw`, split into `fp`/`rp` for forward/reverse strands) and interpulse duration (`ipd`, `fi`/`ri`). The polymerase is sensitive to base modifications, so the kinetics carry a per-site footprint of methylation and other damage processes. PacBio resolves dual-strand CpG methylation natively, but most other modifications do not have a public model. That is the gap.

The current downstream task is binary CpG methylation classification on a single strand: take a 32 bp window centered on a CpG, encode it, attach a shallow head, predict methylated vs unmethylated.

## Where things stand

**Supervised baseline.** `supervised_20` reaches about 82% top-1 on full data with a CNN-transformer at d=128, L=4. The v2 of that recipe (`supervised_40_baseline_v2`) was still climbing at epoch 20 (0.808 to 0.818). A capacity sweep across d=128/256/512/768 (`supervised_51_baseline_size_grid`) is in flight on Gefion to test whether v40's plateau is capacity-bound.

**Autoencoder SSL is the working thread.** Two pretext tasks have produced positive transfer over random init, both autoencoders. `ssl_25` (CpG-data autoencoder, ctx=32) reaches 66% linear-probe top-1, a few pp above random-init probe. `ssl_29` (25G OB007 reads, ctx=128) reaches 63%. Fine-tuning the `ssl_25` encoder (`supervised_27`) brings it to 79% top-1, which is a 13pp recovery over the linear probe and within a few pp of the full-data supervised baseline. The full-data comparison is confounded by schedule differences and by pretraining and fine-tuning sharing a distribution, but the more interesting question is the small-n regime, where SSL is supposed to earn its keep. `supervised_47` (random-init data scaling), `supervised_50` (random vs `ssl_55` vs `ssl_57` finetune at 10k / 100k / 8M), and `supervised_51_finetune_ssl58_grid` (random vs `ssl_58` finetune at the same train sizes) are the matrixed comparisons that pin this down. Results pending on Gefion.

`ssl_58_autoencoder_grid` is the current scale-up of the working recipe. Same 4-size grid (d128/d256/d512/d768) at ctx=512 with `SmrtAutoencoderSmallRF` (small-RF CNN, r0=27) and `MaskedReconstructionLoss`. The structural argument for it is concrete: the autoencoder loss directly penalizes losing per-position kinetic variation (the signal methylation lives in), it has no projection head and no all-gather in the loss path so none of the dimensional-collapse or NCCL-deadlock pathways from `ssl_57` apply, and the small-RF CNN keeps each latent's receptive field local at ctx=512 rather than smearing it across the whole window. Modern post-`ssl_57` harness: no `accelerator.prepare(scheduler)`, `ChunkedRandomSampler` for sequential shard reads, step-based cadences for probe and resume, step-0 baseline on the random-init encoder, encoder-only `step_<N>.pt` milestones, dual-set pair-val (yoran in-distribution and ob007 held-out). Pass criterion at d=128: probe top-1 at least 0.67 and non-decreasing over the last three evals. The `size_*_long` variants of the same grid extend the cosine tail to a 96h walltime without reshaping the curve, so smaller models that finish the 1M-step schedule get to settle at min-lr instead of being clipped at walltime.

**Contrastive is paused.** Variants `ssl_50` through `ssl_57` either failed to beat the random-init probe or collapsed dimensionally during training. The diagnosis from the `ssl_57` post-mortem is structural: contrastive learning enforces an invariance, and every readily available pair definition we tried (within-read crops, neighbor-window pairs, same-tissue pairs) ends up suppressing the per-position kinetic variation that methylation lives in. Lower LR delays collapse without changing the destination. Smaller receptive field does not change the contrastive optimum. Worth revisiting if a fundamentally different positive-pair definition shows up (coverage-based pairs are the most interesting candidate), but no contrastive run is queued.

Compute cliff: GPU walltime on Gefion expires 2026-05-13. After that the project is on much smaller capacity, so anything that needs the full grid has to be queued before then.

## Tissue provenance (paused)

We have read-level tissue labels for several individuals across multiple tissues and species, so it is reasonable to ask whether kinetics over a 2 to 4 kb read can predict the tissue a read came from. The bet is that mutation signatures (oxidative damage in colon, UV in skin) leave a kinetic fingerprint on top of the contextual mutation distribution that COSMIC spectra describe.

`supervised_52_tissue` (8 tissues, 2 PacBio cells, 50k labeled reads, partition keyed on `read_name`) drove train cross-entropy to zero on all four sizes while val never moved. The follow-up sklearn probe sweep at `report/probe_tissue_yoran/` (2026-05-07) ran 13 models against the same partition and `KineticsNorm` the deep model uses. Pooled (25-d), binned (128-d), and flat (8192-d) kinetics features all collapse to chance on val (top1 between 0.121 and 0.137 against chance 0.125), while RF, HGB, and saga-LogReg memorize train to 1.0. Per-tissue 1-vs-rest AUROC sits between 0.50 and 0.57. `KineticsNorm` is not destroying signal, the reverse-kinetics alignment is correct, and ctx=2048 vs ctx=4096 changes nothing. The cell-batch effect (s1 vs s2) is real but small (val 0.549). The signal is not there at this dataset size, in the sense that the supervised model is trying to extract it.

Plausible next moves: scale labeled data well past 50k, A/B per-cell normalization (subtract cell mean before the global z-score), sequence-aware features beyond base composition, or condition on the genomic region the read maps to. None queued.

## Shared encoder

All SSL and supervised models share `SmrtEncoder` (`smrt_foundation/model.py`). Forward path:

```
Input [B, T, 4]   (seq token, IPD, PW, pad mask)
  │
  ├─ SmrtEmbedding
  │    nuc: Embedding(5, d/2)            (scaled by sqrt(d))
  │    kin: Linear(2, d/2)               (scaled by sqrt(d))
  │    concat + LayerNorm                →  [B, T, d]
  │
  ├─ CNN: 11 ResBlocks, two stride-2 (4x downsample on T and on the pad mask)
  │    [B, T, d]                         →  [B, T/4, d]
  │
  ├─ Sinusoidal positional encoding
  │
  └─ Transformer (n_layers x Pre-LN attn + 4x MLP)
       [B, T/4, d]                       →  [B, T/4, d]
```

Default d=128, L=4, H=4. Receptive field of the standard CNN is about 107 bp; `SmrtEncoderSmallRF` is about 27 bp, used by `ssl_58` to keep the per-latent receptive field local relative to ctx=512.

## Repo

```
smrt_foundation/    package: model, dataset, normalization, loss, optim
scripts/            data pipeline scripts and per-experiment dirs under scripts/experiments/
configs/            shared configs (data.yaml is the source of truth for token map and CpG pipeline params)
docs/               status.md, experiment_log.md, methodology.md, negative_results.md
report/             EDA and probe scripts (each in its own subdir with a plot.py / build.py)
tests/              pipeline-fidelity and unit tests (see tests/readme.md)
workflow.py         gwf DAG for the data pipeline only (BAM to Zarr to memmap to validation)
run.sh / test.sh / plot.sh   environment-detecting submission scripts (Gefion / GenomeDK / local)
```

The data pipeline is run via `gwf`. The `CONFIG` dict at the top of `workflow.py` registers BAMs and the Zarr / memmap targets they produce. Add a row, then `gwf run`. Datasets whose name starts with `cpg` go through the CpG windowing pipeline (v2); everything else goes through the SSL segmentation pipeline.

Each experiment lives in its own directory under `scripts/experiments/<name>/` with a `config.yaml` (resources plus hyperparameters) and a `train.py`, often a thin wrapper into a shared loop (e.g. `_shared_train.py` for the size-grid experiments). Submit with `bash run.sh scripts/experiments/<name>` and the script picks the right backend automatically. Job output lands as `<jobid>.out` in the experiment directory rather than `.gwf/logs/`. EDA plots and tests use the same pattern (`bash plot.sh report/eda/<name>`, `bash test.sh tests/<file>`). Each training script has an internal `DEFAULT` dict that fills in any keys missing from `config.yaml`, so configs from older experiments still load when the harness gains a new knob.

## The blocker worth remembering

Reverse kinetics misalignment in `zarr_to_methyl_memmap.py` (now archived). The original v1 script regressed the supervised baseline from about 80% to about 70% by `np.flip`-ing whole reverse-strand reads. PacBio stores `fi`/`fp` in forward order and `ri`/`rp` already in reverse order, so flipping the read flipped `fi` correctly into reverse order but un-flipped `ri` into forward order, leaving the kinetics on the mirror-image position of the sequence. Half the training windows had misaligned features. `zarr_to_methyl_memmap_v2.py` fixes this with explicit reverse indexing (`ri[L-end:L-start]`) instead of a whole-read flip, and matches the legacy parquet pipeline byte-for-byte. `supervised_18` confirmed the fix at about 80%, `supervised_20` extended it to about 82% on full data.

Things ruled out along the way (any of which would have been a more pleasant root cause): normalization method (no effect on the regression), train/val read-level leakage (none, by `tests/test_legacy_leakage.py`), CpG site finding (both pipelines agree), uint8 truncation (PacBio CCS tags are uint8, no truncation possible in the first place).

## Tests

Full list in `tests/readme.md`. The ones that matter most:

- `test_cpg_pipeline_fidelity.py`: BAM to Zarr to CpG memmap, byte-exact at every stage. Pipeline parameters are read from `configs/data.yaml`, not hardcoded.
- `test_zarr_to_methyl_memmap_v2.py`: integration tests for the v2 CpG pipeline (Zarr to shards to `LabeledMemmapDataset` to DataLoader).
- `test_legacy_vs_new_pipeline.py`: end-to-end comparison of the legacy parquet path and the current memmap path on the same BAM reads. The test that originally caught the reverse-kinetics misalignment.
- `test_kinetics_norm.py`: `KineticsNorm` parity with the legacy `ZNorm` and `SSLNorm`, plus the `n_continuous=4` synthetic-data tests added for the tissue task.
- `test_autoencoder.py`: shape, masking, gradient, and weight-transfer tests for `SmrtAutoencoder`.

`bash test.sh tests/` runs the lot.
