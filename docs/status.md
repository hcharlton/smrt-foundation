# Research Status

*Last updated: 2026-05-07*

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

## Tissue provenance (supervised_52) — paused after the probe sweep falsified the premise

`supervised_52_tissue` was the first run on the new yoran ctx4096 dataset (8 tissues × 2 PacBio cells, partition keyed by `read_name`, 50k train / 10k val_s1 / 10k val_s2). Across all four sizes (d128/d256/d512/d768) the deep classifier drives train cross-entropy to ~0 within walltime but val cross-entropy never moves. We didn't know whether this was a preprocessing bug, an alignment bug (the v1 `zarr_to_methyl_memmap.py` style), a cell-batch artefact, or simply absence of signal. **Result: the signal isn't there at this dataset size**, in the sense the supervised model is trying to extract it.

`report/probe_tissue_yoran/` is the diagnostic sweep — 13 standalone sklearn / EDA scripts run against the same partition and `KineticsNorm` the deep model uses (10k train / 2k val_s1 / 2k val_s2 subsample). Findings (full numbers in `docs/experiment_log.md` 2026-05-07 and `docs/negative_results.md` 2026-05-07):

- Pooled (25-d), binned (128-d), and flat (8192-d) kinetics features all collapse to chance on val (top1 ∈ [0.121, 0.137], chance = 0.125) while RF / HGB / saga-LogReg memorise to train top1 = 1.0. Per-tissue 1-vs-rest AUROC ∈ [0.50, 0.57]. The deep model is doing what these probes do: memorise.
- Cell s1 vs s2 from the same pooled features: val 0.549 — a real but small batch effect, not the dominant blocker.
- Read-length only and base-composition only: chance.
- Raw vs `KineticsNorm`-normed: identical val accuracies. KineticsNorm is not destroying signal.
- ctx=2048 vs ctx=4096: the full window is *worse*. The centre-crop is not throwing away tissue signal.
- Reverse-kinetics alignment is correct (per-base symmetry test: `fp@fwd-C ≈ rp@fwd-G` to within 0.04 uint8 units). The build script's `ri[::-1].copy()` step does what `docs/methodology.md:243-249` claims; the v1-style misalignment is not happening.

Two side-effect anomalies surfaced for the tissue dataset (not blockers, worth fixing at the source):
- `data/01_processed/tissue_sets/yoran_ctx4096/partition.csv` has one row with `'val_s1 '` (trailing space). `report.probe_tissue_yoran._shared.load_partition()` strips on load and warns; `scripts/make_tissue_partition.py` should `.str.strip_chars()` before write.
- 580 read_names appear on multiple manifest rows (max 4 windows per read), so `pl.col('read_name').is_in(train_names)` returns 50,018 train rows rather than 50,000 (val_s1 +3, val_s2 +6). The deep model trains on the inflated set; not a correctness bug, but the on-disk "50k" claim is approximate.

**No further architecture iteration on supervised_52 at this dataset size**. The next viable moves on the tissue task are: scale labelled data well past 50k; per-cell normalisation A/B (subtract cell-mean before the global z-score); sequence-aware features beyond base composition; or exploit which genomic regions the read maps to rather than relying on whole-read kinetics. None of those is queued yet.

Library additions still in place (additive, default args preserve all existing call sites):
- `smrt_foundation.normalization.KineticsNorm`: `n_continuous=2|4` arg.
- `smrt_foundation.model.SmrtEncoderTissue`, `TissueClassifier`.

Test surface: 7 synthetic tests in `tests/test_kinetics_norm.py::TestKineticsNormNContinuous4` and 5 live-data tests in `tests/test_probe_tissue_shared.py` (auto-skip without the data dir). All pass.

## Open questions

1. **CpG capacity scaling (supervised_51)**: does any size beat v40's 0.8179?
2. **CpG SSL with masked autoencoder (ssl_58)**: does any size beat ssl_25's 66% probe?
3. **Tissue task feasibility (supervised_52)**: ~~does a 6-channel encoder learn the 8-way classification at 50k labelled reads~~ — answered no by the sklearn probe sweep on 2026-05-07. New question: can per-cell normalisation, more labelled data, or sequence-aware features open the task back up?

## Pending experiments

| Experiment | Status |
|-----------|--------|
| supervised_51 (4 sizes) | Submitted to Gefion |
| ssl_58 (4 sizes) | Submitted to Gefion |
| supervised_52_tissue (4 sizes) | Paused — sklearn probe sweep falsified the premise; no signal at 50k labelled reads |
