# Research Status

*Last updated: 2026-05-06*

## Active objectives

Two parallel threads are running:

1. **Tissue provenance (new task)**: Can a 4-channel-kinetics encoder + 8-way head learn to classify the tissue of origin of a PacBio HiFi read, and how large is the within-cell vs across-cell generalisation gap?
2. **CpG methylation foundation model**: Can self-supervised pretraining on PacBio kinetics produce representations that beat the supervised baseline (~82% top-1)? Open across two SSL families: contrastive (currently failing) and masked autoencoder (currently the only family with positive transfer).

## Established CpG results

| Result | Experiment | Notes |
|--------|-----------|-------|
| Supervised baseline ~82% top-1 | supervised_20 | DirectClassifier d=128/L=4, 20ep, full ds |
| Supervised v40 still climbing at ep20: 0.808 → 0.818 | supervised_40_baseline_v2 | Scheduler-fix preserved (no AcceleratedScheduler wrap) |
| Autoencoder probe 66% top-1 | ssl_25 | First ssl variant with positive transfer |
| Fine-tune ssl_25 encoder → 79% | supervised_27 | 3pp below supervised, schedule-confounded |
| Every contrastive variant ssl_50–57 either fails to beat random init or collapses | — | See `docs/negative_results.md` for the dimensional-collapse / augmentation-mismatch chain |

## Current state of CpG work

**Contrastive thread is paused after ssl_57's collapse chain.** The diagnosis is structural: contrastive learning makes the encoder invariant to whatever defines the positive pair. Both `random_subcrop` (any-position-within-read) and `neighbor_pair_subcrop` failed to produce monotone-improving probe trajectories without collapse. ssl_55 added LN to the projection head to fix the magnitude-runaway pathway; ssl_56 changed the invariant; ssl_57 moved to input-masked prediction with AgInfoNCE; v3 fixed the AcceleratedScheduler step-multiplier bug; v5 cut RF from 107 → 27. All eventually collapse. Conclusion: contrastive objectives in this domain suppress the per-position kinetic variation methylation lives in.

**Autoencoder thread is the active SSL bet.** ssl_58 launched 2026-05-05 across the same 4-size grid (d128_L4 / d256_L8 / d512_L8 / d768_L8) with `SmrtAutoencoderSmallRF` + `MaskedReconstructionLoss` at ctx=512, 1M steps, 8 GPUs × 4 nodes × 48h walltime. Modernised harness inherits all bug fixes (no AcceleratedScheduler wrap, non-finite-grad skip, ProgressState resume, ChunkedRandomSampler, encoder-only step_<N>.pt milestones, step-0 baseline, dual-set pair-val on yoran in-dist + ob007 held-out). Diagnostic `embed_z_std`/`embed_z_norm` log on encoder transformer output `c` via forward hook. Pass at d=128: probe_top1 ≥ 0.67 (beats ssl_25's 66%) and non-decreasing last 3 evals. Status: pending Gefion results.

**Supervised capacity sweep (supervised_51) is in flight on Gefion.** Same 4-size grid at the v40 recipe. Tests whether v40's 0.818 plateau is capacity-bound. 60-ep cosine, 24h walltime per size. Pass: any size beats v40's 0.8179 by ≥ 0.5pp. Status: pending.

## New thread: tissue provenance (supervised_52)

The yoran tissue dataset (`data/01_processed/tissue_sets/yoran_ctx4096`) is built and partitioned. 8 tissues × 2 PacBio cells; the partition.csv at `<data_dir>/partition.csv` carries `read_name → split` mapping for `train` (50k, cell s1) / `val_s1` (10k, cell s1, disjoint with train) / `val_s2` (10k, cell s2, never seen in train), stratified at 6250/1250/1250 reads per tissue, seeded 42, validated by `tests/test_make_tissue_partition.py` (21 tests pass).

The shakedown experiment supervised_52_tissue is implemented and ready to submit on Gefion. 4 size grid (d=128/256/512/768, head_dim=64). 4h walltime per size. Step-cadence harness forked from ssl_58 (per-eval CSV, per-eval `step_<N>.pt` full checkpoint, deterministic centre-crop 4096 → 2048, no augmentation). `dataset_on_gpu=true` materialises each split as a GPU TensorDataset at startup (~2.4 GB train per rank), drops the dataloader IO path entirely.

Library additions to support the new task (additive, default args preserve all existing call sites):
- `smrt_foundation.normalization.KineticsNorm`: `n_continuous=2|4` arg parameterises the kinetics channel list. Default 2 reproduces the SSL/CpG layout bit-for-bit. `save_stats` records `norm_n_continuous`; `load_stats` defaults missing field to 2 for legacy checkpoints.
- `smrt_foundation.model.SmrtEncoderTissue`: subclass of `SmrtEncoder` with `n_continuous=4` embedding and 6-channel slicing `[seq, fi, fp, ri, rp, mask]`. `layer_norm_target` and the SSL-only `targets` return value are dropped (streamlined for supervised use).
- `smrt_foundation.model.TissueClassifier`: encoder + center-latent multiclass head over 8 tissues; pair with `nn.CrossEntropyLoss`.

Test surface for the additions: 7 new synthetic tests in `tests/test_kinetics_norm.py::TestKineticsNormNContinuous4` cover the n_continuous=4 path and the legacy round-trip; all pass on GenomeDK (synthetic data; no cluster artefacts required). Existing data-bound tests skip locally because the CpG subset memmap isn't present here — they run on Gefion.

Pass criterion (all 4 sizes): train loss decreases monotonically; val_s1 top1 substantially above 1/8 = 0.125 (overfitting expected on 50k); measurable val_s1 vs val_s2 gap.

## Open questions

1. **CpG capacity scaling (supervised_51)**: does any size beat v40's 0.8179?
2. **CpG SSL with masked autoencoder (ssl_58)**: does any size beat ssl_25's 66% probe?
3. **Tissue task feasibility (supervised_52)**: does a 6-channel encoder learn the 8-way classification at 50k labelled reads, and how large is the cross-cell gap?

## Pending experiments

| Experiment | Status |
|-----------|--------|
| supervised_51 (4 sizes) | Submitted to Gefion |
| ssl_58 (4 sizes) | Submitted to Gefion |
| supervised_52_tissue (4 sizes) | Implemented; ready to submit on next Gefion sync |
