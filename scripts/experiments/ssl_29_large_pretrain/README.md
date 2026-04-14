# Experiment 29: Large-scale autoencoder pretraining

Tests the actual SSL value proposition: pretrain on ALL available unlabeled data (839K full reads from `ob007_raw.memmap`), then fine-tune on the smaller labeled CpG dataset.

Previous experiments (25, 26) pretrained on the same CpG data used for fine-tuning — a 1:1 unlabeled:labeled ratio that provides no information advantage. This experiment uses the full SSL dataset with multi-crop sampling to generate 32× more gradient updates per read compared to single-crop.

## Key features

- **Multi-crop sampling**: `MultiCropNormedDataset` inflates `__len__` to `32 × n_reads`. Each `__getitem__` returns a fresh random 128-base crop. Consecutive indices map to the same underlying read, so warm-cache memmap hits amortise disk I/O across many gradient updates.
- **High worker count**: `num_workers=8` per rank with `persistent_workers=True`. The previous 2-worker setup bottlenecked the tiny model at ~3 it/s (1–2% GPU utilisation); multi-crop + more workers targets 10–20× higher throughput.
- **Eval-ready checkpoints**: Every 20 epochs to `checkpoints/epoch_N.pt`. Each checkpoint includes encoder weights AND normalization stats (`norm_means`, `norm_stds`, `norm_log_transform`), so downstream eval can reconstruct the normalizer via `KineticsNorm.load_stats(ckpt)`.
- **Linear probe evaluation**: Every 20 epochs (5 probes total) to track representation quality during pretraining.
- **Cosine schedule**: 1% warmup, cosine decay over remaining steps. With multi-crop, total_steps = 100 × 206 × 32 ≈ 660K.

## Budget

- Walltime: 24:00:00, 8 H100 GPUs, 64 CPU cores (8 per rank)
- 100 epochs × 32 crops/read × 206 base-steps = ~660K gradient updates
- Equivalent to ~3200 single-crop passes over the 839K-sample dataset
