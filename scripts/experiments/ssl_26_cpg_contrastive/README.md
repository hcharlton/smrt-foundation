# Experiment 26: Contrastive pretraining on CpG data

Head-to-head comparison with experiment 25. Uses `Smrt2VecInputMask` + `AgInfoNCE` instead of `SmrtAutoencoder` + `MaskedReconstructionLoss`, but on the same CpG data (labels discarded). Isolates the effect of the pretraining objective from the data regime — if both architectures improve similarly, the bottleneck was data mismatch. If the autoencoder wins, reconstruction is genuinely better than contrastive for this task.
