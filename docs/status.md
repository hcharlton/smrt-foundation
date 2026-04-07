# Research Status

*Last updated: 2026-04-01*

## Active Objective

**Can self-supervised pretraining on PacBio kinetics produce representations that beat the supervised baseline on CpG methylation classification?**

## Established Results

| Result | Experiment | Date | Notes |
|--------|-----------|------|-------|
| Supervised baseline: ~82% top-1 | exp 20 | ~2026-02 | DirectClassifier from scratch, 20 epochs, ds_limit=0 |
| Contrastive probe: ~58% (declining) | exp 21/23 | ~2026-03 | Full-read data, InfoNCE |
| Autoencoder probe: ~62% (stable) | exp 24 | 2026-03-28 | Full-read data, MSE reconstruction |
| CpG data regime: +4-5pp for both | exp 25/26 | 2026-03-30 | Autoencoder ~66%, contrastive ~63% |
| Fine-tuning exp 25 encoder: 79% | exp 27 | 2026-03-30 | 3pp below supervised baseline |

## Current State

Fine-tuning (exp 27) recovered most of the probe gap (66% -> 79%) but landed 3pp below the supervised baseline (82%). The comparison is confounded by three factors:

1. **Optimizer schedule difference**: Exp 20 uses single-stage AdamW at lr=3e-3 for 20 epochs. Exp 27 uses two-stage (5 frozen + 15 unfrozen at encoder lr=3e-4). The 10x lower encoder LR may undertrain.

2. **No information advantage from pretraining**: Exp 25 pretrains on CpG data with labels removed; exp 27 fine-tunes on the same data with labels. At a 1:1 unlabeled:labeled ratio, pretraining provides no information the supervised model doesn't already have.

3. **Short pretraining**: Exp 25 trained for ~3k steps (12 epochs on 2M samples). Probe plateaued, but more epochs on the same data would just overfit reconstruction.

Two experiments have been designed to disentangle these confounds but are currently deferred (not yet submitted to the cluster).

## Open Questions

1. **Is the 3pp gap from the fine-tuning schedule or from pretraining hurting?** Exp 28 controls for data budget by training exp 20 from scratch with ds_limit=20M (matching exp 27's budget).

2. **Does pretraining on MORE data help?** Exp 29 tests the actual SSL value proposition: 3000-epoch autoencoder on 839K full reads with random cropping (~27M effective windows/epoch, ~1000 GPU hours). This is the first experiment with a genuine data scale advantage.

## Deferred Experiments

| Experiment | Purpose | Status |
|-----------|---------|--------|
| supervised_28_baseline_control | Exp 20 from scratch with ds_limit=20M -- matches exp 27's data budget | Created, deferred |
| ssl_29_large_pretrain | 3000-epoch autoencoder on full SSL data with random cropping (~1000 GPU hours). Tests whether more unlabeled data -> better initialization | Created, deferred |
