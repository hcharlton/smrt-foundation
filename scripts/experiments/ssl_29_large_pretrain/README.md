# Experiment 29: Large-scale autoencoder pretraining

Tests the actual SSL value proposition: pretrain on ALL available unlabeled data (839K full reads from `ob007_raw.memmap`), then fine-tune on the smaller labeled CpG dataset.

Previous experiments (25, 26) pretrained on the same CpG data used for fine-tuning — a 1:1 unlabeled:labeled ratio that provides no information advantage. This experiment uses the full SSL dataset with random cropping to generate ~27M effective windows per epoch (32x augmentation from 4096→128 random crops).

## Key features

- **Random cropping**: Each epoch, every 4096-position read yields a random 128-base window. Over 500 epochs, the encoder sees diverse views of the same reads.
- **Eval-ready checkpoints**: Every 20 epochs to `checkpoints/epoch_N.pt`. Each checkpoint includes encoder weights AND normalization stats (`norm_means`, `norm_stds`, `norm_log_transform`), so downstream eval can reconstruct the normalizer via `KineticsNorm.load_stats(ckpt)` without access to the original training data.
- **Linear probe evaluation**: Every 100 epochs (5 probes total) to track representation quality during pretraining.
- **Cosine schedule**: 1% warmup (5 epochs), cosine decay over 495 epochs.

## Budget

~100 GPU hours on 8 H100s = ~24 wall hours = ~500 epochs at ~2.5 min/epoch.
