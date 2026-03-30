# Research Status

*Last updated: 2026-03-28*

## Active Objective

**Can self-supervised pretraining on PacBio kinetics produce representations that beat the supervised baseline on CpG methylation classification?**

The supervised baseline (experiment 20) achieves ~80% top-1 accuracy on binary CpG methylation classification. The goal is to pretrain an encoder on unlabeled kinetics data and fine-tune it to surpass this baseline.

## Established Results

| Result | Experiment | Date | Notes |
|--------|-----------|------|-------|
| Supervised baseline: ~80% top-1 | exp 20 (supervised_20_full_v2) | ~2026-02 | Full v2 memmap dataset, DirectClassifier, 20 epochs |
| Legacy baseline reproduced | exp 15 (supervised_15_legacy) | ~2026-02 | Confirmed parquet-based pipeline still works |
| Reverse kinetics bug identified and fixed | exp 17, 18 | ~2026-02 | v1 `zarr_to_methyl_memmap.py` misaligned ri/rp on reverse strand; v2 script fixes this |
| Contrastive SSL converges but doesn't transfer | exp 21 (ssl_21_pretrain) | ~2026-03 | InfoNCE loss decreases; probe accuracy *declines* over epochs (~58%) |
| Short-context SSL doesn't help transfer | exp 23 (ssl_23_shortctx) | ~2026-03 | Reducing context from 4096→128 didn't fix probe degradation |
| Autoencoder marginally better than contrastive | exp 24 (ssl_24_autoencoder) | 2026-03-28 | Probe ~62-63% vs contrastive ~58%. Still far below supervised 80% |

## Open Problems

1. **SSL-to-downstream transfer gap.** Both pretraining objectives (contrastive and reconstruction) produce representations far below the supervised baseline:
   - Contrastive (InfoNCE): ~58% probe accuracy
   - Autoencoder (MSE reconstruction): ~62-63% probe accuracy
   - Supervised baseline: ~82%

   Switching the objective from contrastive to reconstruction yielded only ~4-5pp improvement. The pretraining objective is not the sole bottleneck. Remaining hypotheses:
   - The encoder architecture (128d, 4 layers) may be too small to learn transferable features at the SSL scale
   - Linear probe may be too shallow to exploit pretrained representations — full fine-tuning (exp 22) could close more of the gap
   - The unlabeled SSL data (full reads) may not contain enough methylation-discriminative signal for a frozen encoder to capture

2. **Fine-tuning evaluation (exp 22) — ready to run.** Two-stage fine-tuning (frozen head warmup → differential LR) is the next logical step. Even if the linear probe is weak, full fine-tuning may unlock more of the pretrained signal. Can now use either exp 21 or exp 24 checkpoint.

## Planned Experiments

| Experiment | Hypothesis | Status |
|-----------|-----------|--------|
| supervised_22_finetune | Full fine-tuning with differential LR can close the gap that linear probing cannot | Ready — pretrained checkpoints available from exp 21 and 24 |
