# Negative Results

## ~2026-02: Normalization Method Does Not Explain Memmap Regression

**Hypothesis:** The accuracy gap between legacy parquet (~80%) and new memmap pipeline (~70%) is caused by different normalization (legacy uses log-Z, new uses ZNorm without log1p).

**Experimental Setup:** Experiment 16 (supervised_16_log1p_znorm). Added log1p to ZNorm to match legacy normalization. Same DirectClassifier architecture, matched LR and ds_limit.

**Observation:** ~70% top-1 accuracy, same as without log1p. No improvement.

**Conclusion:** Normalization is not the cause. The regression was later traced to reverse kinetics misalignment in v1 `zarr_to_methyl_memmap.py`.

## ~2026-03: Contrastive Pretraining (InfoNCE) Does Not Transfer to CpG Classification

**Hypothesis:** Wav2vec 2.0-style contrastive pretraining on masked kinetics produces representations useful for downstream CpG methylation classification.

**Experimental Setup:** Experiment 21 (ssl_21_pretrain). Smrt2VecInputMask with AgInfoNCE loss, d=128, L=4, ctx=4096, p_mask=0.15, 12 epochs on 2M samples. Linear probe evaluation (frozen encoder + shallow head) on labeled CpG data every epoch.

**Observation:** InfoNCE training loss converges normally. However, linear probe accuracy on CpG classification *declines* over training epochs. The more the encoder trains on the contrastive objective, the worse it gets at the downstream task.

**Conclusion:** Three compounding issues: (1) normalization mismatch between SSL and probe data (partially fixable), (2) 128× length mismatch between pretraining context (4096→1024 latents) and probe context (32→8 latents), (3) fundamental task misalignment — InfoNCE teaches position discrimination in cosine-similarity space, which doesn't preserve the actual kinetics signal needed for methylation classification. The targets are layer-normed CNN features that shift during training (moving target problem).

## 2026-03-28: Masked Autoencoder Only Marginally Improves Over Contrastive Pretraining

**Hypothesis:** Direct kinetics reconstruction (MSE on masked positions) forces the encoder to preserve methylation-relevant signal, unlike InfoNCE which operates in abstract cosine-similarity space. This should produce representations that transfer better to CpG classification.

**Experimental Setup:** Experiment 24 (ssl_24_autoencoder). SmrtAutoencoder with shallow decoder (2× ConvTranspose1d + Linear), d=128, L=4, H=4, ctx=128, bs=512, lr=3e-4, p_mask=0.15, mask_size=10, MaskedReconstructionLoss (MSE), 12 epochs. Linear probe evaluation on CpG classification (frozen encoder + shallow head).

**Observation:** Probe accuracy ~62-63% top-1 across epochs. Contrastive model (exp 21/23) achieved ~58%. Supervised baseline is ~80%.

**Conclusion:** Switching from contrastive to reconstruction objective yields only ~4-5 percentage points of improvement. The pretraining objective alone is not the dominant bottleneck. Possible remaining factors: (1) the encoder architecture itself may not learn transferable features at this scale/depth, (2) the unlabeled SSL data (full reads) may not contain enough methylation-discriminative signal for the probe to extract, (3) the linear probe evaluation may be too shallow to exploit the learned representations — full fine-tuning (exp 22) could reveal more. The gap between 63% and 80% suggests pretrained features carry *some* methylation signal but far less than direct supervision provides.

## 2026-03-30: Training on CpG Data Improves but Does Not Close Gap to Supervised

**Hypothesis:** The 16-20pp gap between SSL pretraining and supervised baseline is primarily caused by data distribution mismatch (full reads vs CpG-centered windows). Training SSL directly on CpG data (labels discarded, context=32) should close the gap.

**Experimental Setup:** Experiment 25 (autoencoder on CpG data, ctx=32, 12 epochs) and experiment 26 (contrastive on CpG data, ctx=32, 12 epochs). Both use CpG pos+neg memmap data with labels discarded, KineticsNorm from CpG data, same architecture (d=128, L=4).

**Observation:** Autoencoder probe: ~66% (up from 62% on full reads). Contrastive probe: ~63% (up from 58% on full reads). Both stable across epochs, not improving. 30 minutes training each.

**Conclusion:** Data regime mismatch accounts for ~4-5pp of the gap but not the remaining ~16pp. The improvement is consistent across both architectures, suggesting it's a data effect, not architecture-specific. The SSL objectives themselves (reconstruction and contrastive) produce representations that plateau at ~63-66% linear probe accuracy regardless of data regime. The linear probe may be too strict a test — fine-tuning (exp 22) should be run next to determine if the representations contain non-linearly-separable signal that end-to-end training can unlock.

## ~2026-03: Reducing Pretraining Context Length Does Not Fix Transfer

**Hypothesis:** The transfer failure in experiment 21 is primarily caused by the length mismatch between pretraining (4096 bases) and downstream evaluation (32 bases). Training at context=128 (32 latents after CNN, matching probe) will fix probe accuracy.

**Experimental Setup:** Experiment 23 (ssl_23_shortctx). Same as experiment 21 but with context=128, batch_size=512, max_negatives=1024.

**Observation:** Probe accuracy still declines over training epochs, same pattern as experiment 21.

**Conclusion:** Length mismatch is not the dominant blocker. Task misalignment (contrastive objective vs. classification signal) is likely the primary cause. This motivates the switch to autoencoder pretraining (experiment 24), which directly targets kinetics reconstruction rather than position discrimination.

## 2026-04-27: SimCLR Projection-Head Magnitude Runaway at d_model >= 768

**Hypothesis:** ssl_54's d=768 SimCLR run scales monotonically from d=512. Same architecture family, same hyperparameters, same data — only `d_model` and `n_head` change (head_dim=64 fixed) and `batch_size` halves to 256 for activation memory.

**Experimental Setup:** ssl_54_simclr_grid_yoran/size_d768_L8. SimCLRSmrt with `MLPProjectionHead` (Linear → GELU → Linear, no internal normalization), d_model=768, n_head=12, n_layers=8, ctx=32, batch_size=256, max_lr=3e-4, weight_decay=1e-4, pct_start=0.05, grad_clip=5.0, temperature=0.1, ds_limit=0, epochs=50, yoran dataset. Trained ~970k steps observed.

**Observation:** Two-phase failure.

- **Phase 1 (steps 0 → ~150k): slow magnitude drift.** `embed_z_std` (per-channel std of pre-`F.normalize` projection output) climbed from ~0 to ~50. `grad_norm` stayed bounded at ~0.4 throughout — the `clip_grad_norm_=5.0` threshold never engaged. `probe_top1` peaked at **0.65 in epoch 1** (matching d=512's plateau) then began drifting down.
- **Phase 2 (~step 150k–200k): single-step catastrophic explosion.** `grad_norm` spiked to ~6.5×10¹⁷ then transitioned to **Infinity** for the remaining ~770k steps. `embed_z_std` jumped to ~3.6M and plateaued. `probe_top1` collapsed to **0.50 (random)** and stayed there.

`d=512_L8` with the identical architecture family (only width differs) bumped to z_std ≈ 60 around step 100k and decayed back to ~50, with probe stable at ~0.65. Phase 2 never triggered at d=512.

**Conclusion:** `MLPProjectionHead` without internal normalization is unsafe at `d_model >= 768`. Mechanism: `F.normalize`'s Jacobian scales with `1/||z||`, so as the projection head drifts to high-magnitude outputs the gradient flowing back through the contrastive loss is damped proportionally. The model can't pull magnitudes back down because the very thing that would supply that gradient is what's being damped. Eventually a single batch produces non-finite gradients; `clip_grad_norm_` cannot recover from this (when `total_norm = Inf`, `clip_coef = 5/Inf = 0`, but `0 × Inf = NaN`, so the clip itself can introduce NaNs into the parameters). After that point every forward produces non-finite values and `grad_norm` is pinned at `Inf` permanently — a zombie run.

The failure is **width-dependent and tied to projection-head capacity**: the head has `O(d_model²)` parameters, so d=768's head has ~21× the capacity of d=128's head. Smaller heads can't drift as easily into the high-magnitude attractor. The encoder is briefly capable at d=768 (epoch 1 probe = 0.65 = d=512's plateau), so this is a head-dynamics problem rather than a fundamental encoder-capacity limit.

**Fix tested in ssl_55:**
1. Cause-side: `MLPProjectionHeadLN` with LayerNorm between Linear→GELU on hidden layers and a final LayerNorm after the last Linear (immediately before `F.normalize`). Bounds the per-channel scale of the projection output. LN placement mirrors SimCLR v1 §4.2's BN placement (Chen et al. 2020).
2. Defensive: `_shared_train.py` skips both `optimizer.step()` and `scheduler.step()` when `clip_grad_norm_` returns a non-finite value, with a `nonfinite_skip_count` TB counter. Converts a single bad batch from "permanently corrupts the run" into "one lost step." Should not engage if the LN fix is doing its job.

**Reading:** Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" (ICML 2020). Jing et al., "Understanding Dimensional Collapse in Contrastive Self-Supervised Learning" (ICLR 2022). Chen et al., SimCLR v1 §4.2 and Appendix B (NeurIPS 2020) for the original BN-in-projection-head empirical justification.

## 2026-05-04: All Accelerate-Driven SSL Runs Trained at 1/num_processes the Specified Schedule Horizon

**Hypothesis (implicit, never explicitly tested):** `total_steps=N` in an SSL config produces a cosine schedule that bottoms at global step N.

**Experimental Setup:** Investigation triggered by the ssl_57 1m_v2 collapse (`docs/experiment_log.md` 2026-05-01). 1m_v2 was a single-variable warmup-duration fix (`pct_start: 0.10 → 0.03`) over 1m_v1, motivated by the hypothesis that the long warmup destabilized the moving-target AgInfoNCE training. v2 also collapsed. The shared early-trajectory shape between v2 and the 300k run prompted plotting z_std vs cumulative LR (`report/training_dynamics/ssl57_zstd_attractor/`) to test a shared-attractor reading, which led to inspecting the actual LR schedule. Source-checked `accelerate.scheduler.AcceleratedScheduler.step` (accelerate 1.13.0, default `split_batches=False` path).

**Observation:** `accelerator.prepare(scheduler)` wraps the underlying `LambdaLR` as `AcceleratedScheduler`, whose `step()` method advances the wrapped scheduler by `num_processes` per call:

```python
# accelerate/scheduler.py: AcceleratedScheduler.step()
else:
    # the training dataloader batch size was multiplied by `num_processes`,
    # so we need to do num_processes steps per training step
    num_processes = AcceleratorState().num_processes
    for _ in range(num_processes):
        ...
        self.scheduler.step(*args, **kwargs)
```

With 8 GPUs and `total_steps=1000000`, the cosine bottoms at global step **125,000** (= 1M / 8). The remaining 87% of training runs at `min_lr_ratio × max_lr = 0.05 × 3e-4 = 1.5e-5` — effectively frozen. **Every Accelerate-driven SSL run in the project carries this bug**: ssl_53, ssl_54, ssl_55, ssl_56, ssl_57 (300k, 1m, 1m_v2). The supervised harness already had the fix in supervised_40 (`docs/experiment_log.md` 2026-04-16); the SSL harness never inherited it.

**Conclusion:** Single-line patch — remove `scheduler = accelerator.prepare(scheduler)` from `ssl_57_inputmask_grid_lnhead/_shared_train.py:463`. The raw `LambdaLR` from `get_cosine_schedule_with_warmup` (`smrt_foundation/optim.py:5`) is correct; the training loop's per-step `scheduler.step()` advances it by 1 per global step. Applied to ssl_57 in the 1m_v3 row (`docs/experiment_log.md` 2026-05-04). `ssl_55_simclr_grid_lnhead/_shared_train.py` and `ssl_56_simclr_neighbor_invariance/_shared_train.py` carry the same bug; not patched here because those experiments are not being rerun, but the same one-line removal is needed if either is ever resubmitted.

**Caveat for past SSL interpretations:** ssl_53/54/55/56/57 probe trajectories should be read with the schedule-compression confound. The 300k SSL runs effectively trained at full LR for ~37,500 global steps then idled at 1.5e-5 for the remaining 262,500. Patterns previously interpreted as "stable plateau" (300k probe trajectories), "oscillating without monotone improvement" (ssl_55 d=512), or "still climbing at 300k" (ssl_57 300k probe reading from the user-reported screenshots) were partially the encoder not actually training in the late phase. ssl_57 1m_v3 is the first Accelerate-driven SSL run to test the actual schedule horizon. Cross-experiment "did extending help?" questions across the historical SSL log are confounded by the bug.

## 2026-05-07: yoran Tissue Kinetics Carry No Generalisable Tissue Signal at 50k Labelled Reads

**Hypothesis:** `supervised_52_tissue` overfits because the architecture or preprocessing is suppressing tissue signal in the kinetics. A model-independent sklearn probe with comparable features should at least beat chance on val_s1 if the signal is there.

**Experimental Setup:** `report/probe_tissue_yoran/` — 13 standalone scripts running on a GenomeDK interactive node (`.venv`) against the same `data/01_processed/tissue_sets/yoran_ctx4096/partition.csv` the deep model uses. 10k train / 2k val_s1 / 2k val_s2 subsample, `KineticsNorm(n_continuous=4)` + deterministic centre-crop 4096 → 2048 (matches `_shared_train.py:355-370`). Probe matrix: pooled_summary (25-d: per-channel mean/std/p10/p50/p90 + 5 base frequencies) × binned_summary (16 bins × 4 channels × 2 stats = 128-d) × flat_kinetics (2048 × 4 = 8192-d) crossed with LogisticRegression / RandomForest / HistGradientBoostingClassifier (saga LogReg L2 only on the 8192-d rep). Plus diagnostics: cell-classifier from same features, read-length only, seq-only, raw vs normed, ctx=2048 vs 4096, 8 binary 1-vs-rest, PCA scatter, per-channel densities by tissue and by cell, and a reverse-kinetics alignment EDA via per-base symmetry.

**Observation:**

- Every kinetics-summary representation collapses to chance on validation despite RF / HGB / saga LogReg achieving train top1 = 1.0:
  - pooled (25-d): val_s1 ∈ {0.129, 0.137, 0.132}, val_s2 ∈ {0.136, 0.130, 0.128}.
  - binned (128-d): val_s1 ∈ {0.132, 0.137, 0.129}, val_s2 ∈ {0.131, 0.132, 0.130}.
  - flat (8192-d): val_s1 = 0.134, val_s2 = 0.123.
  Chance is 1/8 = 0.125. Macro-F1 tracks accuracy.
- All 8 binary 1-vs-rest LogReg AUROCs lie in [0.50, 0.57] on both val_s1 and val_s2. No single tissue is meaningfully separable.
- Cell s1 vs s2 from pooled features: LogReg val 0.549 / AUROC 0.555. A weak but real batch effect, not strong enough on its own to explain the failure.
- Read-length only: val 0.142. Seq composition only: val 0.142. Both at chance.
- Raw vs `KineticsNorm`-normed pooled features (HistGB): val_s1 0.131 vs 0.132. Identical to two decimals — normalisation is not destroying signal.
- ctx=2048 (centre-cropped) vs ctx=4096 (full window) (HistGB on pooled summary): val_s1 0.132 vs 0.121, val_s2 0.128 vs 0.116. The full window is *worse*; the centre-crop is not throwing away tissue signal.
- Reverse-kinetics alignment is correct: per-base symmetry test (`results/reverse_kinetics_alignment_per_base.csv`) shows `fp@fwd-C ≈ rp@fwd-G` to within 0.04 raw uint8 units across all four base pairs. The position-level Pearson floor at ~0.15 between `fi[j]` and `ri[j]` (vs ~0.15 for the reversed pairing) is a per-read overall-rate effect, not misalignment.
- Two on-disk anomalies surfaced as side-effects: `partition.csv` has one row with `'val_s1 '` (trailing space, byte 0x20) which silently drops one read from any string-equality filter on `split == 'val_s1'`; and 580 read_names have multiple manifest rows (max 4) so `pl.col('read_name').is_in(train_names)` produces 50,018 rather than the declared 50,000 train rows (val_s1 +3, val_s2 +6). Both detected by `report/probe_tissue_yoran/eda_partition_sanity.py`.

**Conclusion:** The supervised_52_tissue val plateau reflects the data, not the architecture, the preprocessing, or any alignment bug. With a 50k-labelled-read subsample on yoran, the kinetics features at every level of granularity (per-window summary, per-bin summary, per-position) carry no extractable tissue signal that generalises across reads. The deep model is doing exactly what an sklearn probe with the same features does: memorise the training set and predict at chance on validation.

The cell-batch effect exists but is small (cell-classifier val 0.55 vs chance 0.5) and not the dominant blocker. The kinetics shift between the two cells (per-channel mean shift ~0.06 z-score units, see `results/kinetics_means_by_tissue_cell.csv`) is comparable in magnitude to the within-cell between-tissue mean shift, so a global z-score (`KineticsNorm`) leaves the cell shift in the data. A per-cell normalisation step (subtract per-cell channel mean before global z-score) would test whether removing the batch shift exposes more tissue signal, but the probe results suggest the within-cell signal is also weak.

Practical implications:
1. Stop iterating supervised_52 architecture variants on this dataset size — there is no signal to fit.
2. The next viable moves on the tissue task are: scale labelled data (50k → much larger), per-cell normalisation as an A/B, sequence-aware features beyond base composition (k-mer composition, mappability, repeat content), or exploiting which genomic regions the read maps to (the deep encoder sees raw kinetics, not where the read comes from).
3. Two `partition.csv` anomalies should be fixed at the source (`scripts/make_tissue_partition.py` should `.str.strip_chars()` before write; the upstream BAM has duplicate primary records causing the 580 manifest duplicates and either the build script or the partition script should de-duplicate or document the inflation explicitly).
