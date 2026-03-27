# Experiment 23: Short-context SSL pretraining

Tests whether the declining linear probe accuracy in experiment 21 was caused by the sequence length mismatch between SSL training (context=4096) and CpG probe evaluation (context=32).

## Hypothesis

In experiment 21, the encoder trains on 4096-position sequences (1024 positions after CNN downsampling) but is probed on 32-position CpG windows (8 positions after CNN). The transformer learns attention patterns calibrated for 1024-position sequences that don't transfer to 8 positions, and the CNN operates in a different receptive field regime (local window vs entire input visible).

This experiment uses context=128 (32 positions after CNN), which is much closer to the probe's 8 positions. If the length mismatch was a significant factor, probe accuracy should improve or at least stabilize.

## Changes from experiment 21

- `context: 128` instead of `4096` — SSL sequences are 128 bases instead of 4096
- The normalization fix from experiment 21 is included: the probe uses the same `ssl_norm` as training instead of computing separate statistics
- Everything else is identical (d_model, n_layers, masking, loss, etc.)

## What to look for

1. Does the probe accuracy stabilize instead of declining?
2. Does the probe accuracy start higher than experiment 21?
3. Does the SSL loss still converge?

If probe accuracy still declines, the issue is more likely fundamental task misalignment (the SSL reconstruction objective doesn't produce features useful for methylation classification) rather than an architectural mismatch.
