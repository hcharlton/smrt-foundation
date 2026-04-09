"""Benchmark epoch iteration time for ssl_29_large_pretrain.

Loads the real memmap dataset, times forward+backward passes on 1 GPU,
and extrapolates to multi-GPU DDP training at 3000 epochs.

Usage:
    python -m scripts.benchmark_epoch <memmap_dir> [--batch-size 512] [--context 128] [--steps 30]

Submit on Gefion:
    sbatch --account=cu_0030 --gres=gpu:1 --mem=64gb --time=00:30:00 \
      --wrap="python -m scripts.benchmark_epoch data/01_processed/ob007_raw.memmap"
"""

import argparse
import time
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from smrt_foundation.dataset import ShardedMemmapDataset
from smrt_foundation.normalization import KineticsNorm
from smrt_foundation.model import SmrtAutoencoder
from smrt_foundation.loss import MaskedReconstructionLoss

WARMUP_STEPS = 5


class RandomCropNormedDataset(torch.utils.data.Dataset):
    def __init__(self, inner, norm_fn, context):
        self.inner = inner
        self.norm_fn = norm_fn
        self.context = context

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x = self.inner[idx]
        max_start = x.shape[0] - self.context
        if max_start > 0:
            start = torch.randint(0, max_start, (1,)).item()
            x = x[start:start + self.context]
        else:
            x = x[:self.context]
        return self.norm_fn(x)


def main():
    parser = argparse.ArgumentParser(description="Benchmark epoch time for exp 29")
    parser.add_argument("memmap_dir", help="Path to memmap shard directory")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30, help="Number of timed steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset inventory ---
    ds = ShardedMemmapDataset(args.memmap_dir)
    shard_paths = sorted(glob.glob(os.path.join(os.path.expandvars(args.memmap_dir), "*.npy")))
    first = np.load(shard_paths[0], mmap_mode='r')
    print(f"\n=== Dataset ===")
    print(f"Directory: {args.memmap_dir}")
    print(f"Shards: {len(shard_paths)}")
    print(f"Total samples: {len(ds):,}")
    print(f"Sample shape: {first.shape[1:]}")
    print(f"Shard size: {first.shape[0]:,}")

    steps_1gpu = len(ds) // args.batch_size
    print(f"\nSteps/epoch (1 GPU, bs={args.batch_size}): {steps_1gpu}")
    for n_gpu in [2, 4, 8]:
        steps = len(ds) // n_gpu // args.batch_size
        print(f"Steps/epoch ({n_gpu} GPU DDP, bs={args.batch_size}): {steps}")

    # --- Build pipeline ---
    print(f"\nComputing normalization stats...")
    norm = KineticsNorm(ds, max_samples=16_384)
    print(f"Norm means: {norm.means}, stds: {norm.stds}")

    normed_ds = RandomCropNormedDataset(ds, norm, args.context)
    dl = DataLoader(normed_ds, batch_size=args.batch_size, num_workers=2,
                    pin_memory=True, prefetch_factor=4, shuffle=True)

    # --- Build model ---
    model = SmrtAutoencoder(d_model=128, n_layers=4, n_head=4,
                            max_len=args.context, p_mask=0.15, mask_size=10).to(device)
    criterion = MaskedReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"CNN receptive field: {model.encoder.cnn.r0} bases")

    # --- Phase 1: Data loading throughput ---
    print(f"\n=== Data loading (no GPU) ===")
    it = iter(dl)
    for _ in range(WARMUP_STEPS):
        next(it)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        next(it)
    dl_elapsed = time.perf_counter() - t0
    dl_per_batch = dl_elapsed / args.steps
    print(f"Time/batch: {dl_per_batch:.4f}s ({args.steps} batches)")

    # --- Phase 2: Full training step ---
    print(f"\n=== Training step (forward + backward + optimizer) ===")
    model.train()
    dl2 = DataLoader(normed_ds, batch_size=args.batch_size, num_workers=2,
                     pin_memory=True, prefetch_factor=4, shuffle=True)
    it2 = iter(dl2)

    # Warmup (CUDA caches, JIT)
    for _ in range(WARMUP_STEPS):
        batch = next(it2).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            kin_recon, kin_target, mask = model(batch)
            loss = criterion(kin_recon, kin_target, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Timed steps
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        batch = next(it2).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            kin_recon, kin_target, mask = model(batch)
            loss = criterion(kin_recon, kin_target, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - t0
    sec_per_step = train_elapsed / args.steps
    print(f"Time/step: {sec_per_step:.4f}s ({args.steps} steps)")

    # --- Extrapolation ---
    print(f"\n=== Extrapolation (3000 epochs) ===")
    print(f"{'GPUs':>4}  {'Steps/ep':>10}  {'Epoch time':>12}  {'3000ep wall':>14}  {'GPU-hours':>10}")
    print("-" * 60)
    for n_gpu in [1, 2, 4, 8]:
        steps = len(ds) // n_gpu // args.batch_size
        # DDP overhead: ~15% for gradient all-reduce (skip for 1 GPU)
        overhead = 1.15 if n_gpu > 1 else 1.0
        epoch_sec = steps * sec_per_step * overhead
        total_wall_hr = epoch_sec * 3000 / 3600
        total_gpu_hr = total_wall_hr * n_gpu
        print(f"{n_gpu:>4}  {steps:>10}  {epoch_sec:>10.1f}s  {total_wall_hr:>12.1f}h  {total_gpu_hr:>10.0f}")


if __name__ == "__main__":
    main()
