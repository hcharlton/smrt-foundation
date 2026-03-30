# Experiment 25: Autoencoder pretraining on CpG data

Trains `SmrtAutoencoder` directly on CpG-centered windows (labels discarded) instead of full-read SSL data. Eliminates the triple mismatch that plagued experiments 21-24: same data distribution, same context length (32), same normalization statistics. If the probe improves significantly, the bottleneck was data regime, not the reconstruction objective.
