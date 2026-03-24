"""
IO profiling experiment — no actual training.

Measures DataLoader throughput under different configurations to isolate
whether training is IO-bound or compute-bound.

Usage:
    accelerate launch --num_processes=1 experiments/profile_io/train.py
"""

import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import sys 
module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)
from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EXP_DIR, 'config.yaml')


def profile_dataloader(ds, batch_size, num_workers, num_batches, shuffle, device, model=None):
    """Time a fixed number of batches. Optionally run a model forward pass."""
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # warmup
    warmup = min(5, num_batches)
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        if model is not None:
            with torch.no_grad():
                model(x)
        if i >= warmup - 1:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    batches_done = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        if model is not None:
            with torch.no_grad():
                model(x)
        batches_done += 1
        if batches_done >= num_batches:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    its = batches_done / elapsed
    samples_per_sec = (batches_done * batch_size) / elapsed

    return elapsed, its, samples_per_sec


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device

    clf = config.get('classifier', {})
    prof = config.get('profile', {})
    batch_size = clf.get('batch_size', 2048)
    ds_limit = clf.get('ds_limit', 2_000_000)
    num_batches = prof.get('num_batches', 500)
    workers_sweep = prof.get('num_workers_sweep', [0, 2, 4, 8])

    ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=ds_limit,
    )
    print(f"Dataset size: {len(ds):,} samples")
    print(f"Batch size: {batch_size}")
    print(f"Batches per sweep: {num_batches}")
    print(f"Device: {device}")
    print()

    # --- Phase 1: DataLoader only (pure IO) ---
    print("=" * 70)
    print("PHASE 1: DataLoader only (no model forward pass)")
    print("=" * 70)
    header = f"{'workers':>8} {'shuffle':>8} {'time (s)':>10} {'it/s':>10} {'samples/s':>12}"
    print(header)
    print("-" * len(header))

    for nw in workers_sweep:
        for shuffle in [False, True]:
            elapsed, its, sps = profile_dataloader(
                ds, batch_size, nw, num_batches, shuffle, device, model=None,
            )
            print(f"{nw:>8} {str(shuffle):>8} {elapsed:>10.2f} {its:>10.1f} {sps:>12,.0f}")

    # --- Phase 2: DataLoader + model forward (IO + compute) ---
    print()
    print("=" * 70)
    print("PHASE 2: DataLoader + model forward (no backward)")
    print("=" * 70)

    model = DirectClassifier(
        d_model=clf.get('d_model', 128),
        n_layers=clf.get('n_layers', 4),
        n_head=clf.get('n_head', 4),
        max_len=clf.get('context', 32),
    ).to(device).eval()

    print(header)
    print("-" * len(header))

    for nw in workers_sweep:
        for shuffle in [False, True]:
            elapsed, its, sps = profile_dataloader(
                ds, batch_size, nw, num_batches, shuffle, device, model=model,
            )
            print(f"{nw:>8} {str(shuffle):>8} {elapsed:>10.2f} {its:>10.1f} {sps:>12,.0f}")

    print()
    print("If Phase 1 is slow -> IO bound (fix: smaller shards, more workers, chunked sampler)")
    print("If Phase 1 is fast but Phase 2 is slow -> compute bound (fix: mixed precision, smaller model)")
    print("If shuffle=True is much slower than shuffle=False -> random access bottleneck (fix: chunked sampler)")


if __name__ == "__main__":
    main()
