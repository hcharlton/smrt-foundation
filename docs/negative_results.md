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
