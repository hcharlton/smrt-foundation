# Research Status

*Last updated: 2026-03-30*

## Active Objective

**Can self-supervised pretraining on PacBio kinetics produce representations that beat the supervised baseline on CpG methylation classification?**

The supervised baseline (experiment 20) achieves ~82% top-1 accuracy. The goal is to pretrain an encoder on unlabeled kinetics data and fine-tune it to surpass this baseline.

## Established Results

| Result | Experiment | Date | Notes |
|--------|-----------|------|-------|
| Supervised baseline: ~82% top-1 | exp 20 (supervised_20_full_v2) | ~2026-02 | Full v2 memmap dataset, DirectClassifier, 20 epochs |
| Contrastive SSL converges but doesn't transfer | exp 21/23 | ~2026-03 | InfoNCE loss decreases; probe accuracy declines (~58%) |
| Autoencoder marginally better than contrastive | exp 24 (ssl_24_autoencoder) | 2026-03-28 | Probe ~62% on full-read data (ctx=128) |
| CpG data regime improves both architectures by ~4-5pp | exp 25, 26 | 2026-03-30 | Autoencoder ~66%, contrastive ~63% on CpG data (ctx=32). Stable across epochs |
| Autoencoder consistently outperforms contrastive by ~3pp | exp 24-26 | 2026-03-30 | Holds across both full-read and CpG data regimes |

## Current Gap

Best SSL probe: **66%** (exp 25, autoencoder on CpG data). Supervised baseline: **82%**. Gap: **16pp**.

The data regime mismatch accounted for ~4-5pp. The remaining 16pp gap is either:
1. An artifact of the linear probe being too strict (single Linear(128,1), 3 epochs, frozen encoder), or
2. A genuine limitation of the reconstruction/contrastive objectives

## Next Step

**Fine-tuning evaluation (exp 22) with the experiment 25 checkpoint.** The linear probe is a strict test — single `Linear(128,1)`, 3 epochs, frozen encoder. The supervised model uses a deeper head and end-to-end training. Fine-tuning is the proper comparison:
- Two-stage: frozen encoder + head warmup (5 ep) → unfreeze all + differential LR (15 ep)
- If fine-tuning reaches 82%+: pretraining works, the probe was the bottleneck
- If fine-tuning doesn't help: the representations genuinely lack discriminative signal

Not pursuing more pretraining epochs — probe accuracy plateaued in both exp 25 and 26.

## Planned Experiments

| Experiment | Hypothesis | Status |
|-----------|-----------|--------|
| supervised_27_finetune_ae | Fine-tuning exp 25 encoder unlocks non-linear signal that linear probing misses | Ready to run |
| supervised_22_finetune | Original fine-tuning targeting exp 21 checkpoint | Superseded by exp 27 (exp 21 probe was only 58%) |
