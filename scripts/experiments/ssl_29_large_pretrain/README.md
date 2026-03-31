# Experiment 29: Large-scale autoencoder pretraining

Tests the actual SSL value proposition: pretrain on ALL available unlabeled data (839K full reads from `ob007_raw.memmap`), then fine-tune on the smaller labeled CpG dataset.

Previous experiments (25, 26) pretrained on the same CpG data used for fine-tuning — a 1:1 unlabeled:labeled ratio that provides no information advantage. This experiment uses the full SSL dataset with random cropping to generate ~27M effective windows per epoch (32x augmentation from 4096→128 random crops).

## Key features

- **Random cropping**: Each epoch, every 4096-position read yields a random 128-base window. Over 3000 epochs, the encoder sees diverse views of the same reads.
- **Periodic checkpoints**: Every 100 epochs to `checkpoints/epoch_N.pt`. Allows post-hoc selection of best checkpoint and recovery from timeouts.
- **Reduced probe frequency**: Every 100 epochs (not every epoch) to minimize overhead during the 5-day run.
- **Long cosine schedule**: 1% warmup (30 epochs), cosine decay over 2970 epochs, min LR floor at 5%.

## Budget

~1000 GPU hours on 8 H100s = ~125 wall hours = ~3000 epochs at ~2.5 min/epoch.
