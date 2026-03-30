# Experiment 27: Fine-tune autoencoder pretrained encoder

Fine-tunes the experiment 25 checkpoint (autoencoder pretrained on CpG data, best probe accuracy at ~66%). Supersedes experiment 22, which targeted the exp 21 checkpoint (contrastive on full reads, ~58% probe — not worth fine-tuning).

The linear probe is a strict test: single `Linear(128,1)`, 3 epochs, frozen encoder. This experiment uses the full `DirectClassifier` head (`Linear(128,64) → GELU → Linear(64,1)`) with two-stage training:

- Stage 1 (5 epochs): Frozen encoder, train head only at lr=3e-3
- Stage 2 (15 epochs): Unfreeze all, differential LR (encoder 3e-4, head 3e-3)

If this beats the 82% supervised baseline (exp 20), pretraining is working and the probe was the bottleneck. If not, the pretrained representations genuinely lack discriminative signal.
