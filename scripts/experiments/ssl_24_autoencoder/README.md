# Experiment 24: Masked autoencoder pretraining

Replaces the contrastive objective (experiments 21, 23) with direct kinetics reconstruction. Motivated by the observation that InfoNCE probe accuracy declines over training epochs — the contrastive loss operates in abstract embedding space and discards the actual kinetics signal. See the main README "Why contrastive pretraining failed to transfer" section for the full analysis.

Uses `SmrtAutoencoder` (encoder + lightweight decoder) with `MaskedReconstructionLoss` (MSE on masked positions). Same encoder architecture, context=128, and probe evaluation as experiment 23.
