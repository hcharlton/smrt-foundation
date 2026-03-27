# Experiment 24: Masked autoencoder pretraining

Tests whether a reconstruction-based SSL objective produces representations more useful for downstream CpG methylation classification than the contrastive approach (experiments 21, 23).

## Hypothesis

The contrastive loss (InfoNCE) encourages the encoder to match masked positions to targets in an abstract embedding space. As training progresses, the encoder specializes in contrastive features and discards the actual kinetics signal — the probe accuracy declines over epochs.

A masked autoencoder directly reconstructs the masked kinetics values (MSE loss in the original data space). This forces the encoder to preserve the actual kinetics information in its representations, including the subtle shifts caused by methylation. The reconstruction targets are fixed (the original input values), avoiding the moving-target problem of the contrastive approach where targets shift as the encoder's CNN evolves.

## Architecture

- **Encoder**: `SmrtEncoder` (identical to experiments 21/23 — CNN + transformer)
- **Decoder**: `SmrtDecoder` (new) — two transposed convolutions for 4x upsampling, then a linear projection to 2 kinetics channels. Intentionally lightweight so the encoder must do the heavy lifting
- **Masking**: Same input-level masking as experiment 23 (p_mask=0.15, mask_size=10, kinetics channels zeroed)
- **Loss**: `MaskedReconstructionLoss` — MSE only on masked positions

## Key differences from experiments 21/23

| | Contrastive (21/23) | Autoencoder (24) |
|---|---|---|
| Loss | InfoNCE (cosine similarity in embedding space) | MSE (reconstruction in data space) |
| Targets | Layer-normed CNN features (shift over training) | Original input kinetics (fixed) |
| Distributed | Requires all_gather for negatives | Standard DDP gradient averaging |
| What encoder learns | Features that distinguish positions from each other | Features that preserve kinetics values |

## What to look for

1. Does the probe accuracy improve or stabilize over epochs (vs declining in 21/23)?
2. Does the reconstruction loss converge?
3. How does the final probe accuracy compare to supervised baseline (82%)?
