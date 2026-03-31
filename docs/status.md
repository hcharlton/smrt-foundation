# Research Status

*Last updated: 2026-03-30*

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

Fine-tuning (exp 27) recovered most of the probe gap (66% → 79%) but landed 3pp below the supervised baseline (82%). The comparison is confounded:

1. **Optimizer schedule difference**: Exp 20 uses single-stage AdamW at lr=3e-3 for 20 epochs. Exp 27 uses two-stage (5 frozen + 15 unfrozen at encoder lr=3e-4). The 10x lower encoder LR may be too conservative.

2. **No information advantage from pretraining**: Exp 25 pretrains on the CpG data with labels removed. Exp 27 fine-tunes on the same data with labels. The unlabeled pretraining data is a subset of the labeled fine-tuning data. SSL is designed to leverage MORE unlabeled data than labeled data (wav2vec 2.0 uses 10-6000x more unlabeled data). At a 1:1 ratio, pretraining provides no information the supervised model doesn't already have.

3. **Short pretraining**: Exp 25 trained for 30 minutes (~3k steps, 12 epochs). Probe plateaued, but more epochs on the same 2M samples would just overfit reconstruction — it wouldn't add new signal.

## Open Questions

1. **Is the 3pp gap from the fine-tuning schedule or from pretraining hurting?** A controlled experiment (exp 20 schedule + exp 27's ds_limit) would disambiguate.

2. **Does pretraining on MORE data (full reads) help fine-tuning?** Exp 24's checkpoint (autoencoder on 839K full reads) could be fine-tuned with the exp 27 schedule. This tests the actual SSL value proposition: more unlabeled data → better initialization.

## Planned Experiments

| Experiment | Purpose | Status |
|-----------|---------|--------|
| supervised_28_baseline_control | Exp 20 from scratch with ds_limit=20M — matches exp 27's data budget | Ready to run |
| ssl_29_large_pretrain | 3000-epoch autoencoder on full SSL data with random cropping (~1000 GPU hours). Tests whether more unlabeled data → better initialization | Ready to run |
