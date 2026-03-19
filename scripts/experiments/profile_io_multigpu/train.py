"""
Multi-GPU IO profiling — mirrors real training DDP setup.

Measures throughput under the exact same accelerate/DDP conditions as
train_supervised.py to isolate whether multi-GPU introduces an IO bottleneck.

Runs three phases:
  1. DataLoader only (pure IO + data transfer)
  2. DataLoader + model forward (IO + compute, no gradients)
  3. DataLoader + forward + backward (full training step, no optimizer)

Usage:
    accelerate launch --num_processes=8 experiments/profile_io_multigpu/train.py
"""


import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import sys 
module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)
from smrt_foundation.dataset import LabeledMemmapDataset
from smrt_foundation.model import DirectClassifier

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EXP_DIR, 'config.yaml')


def run_phase(accelerator, dataloader, num_batches, model=None, backward=False):
    """Time a fixed number of batches through the prepared dataloader."""
    criterion = nn.BCEWithLogitsLoss() if backward else None

    # warmup
    warmup = min(5, num_batches)
    for i, (x, y) in enumerate(dataloader):
        if model is not None:
            if backward:
                logits = model(x)
                loss = criterion(logits, y.unsqueeze(1).to(torch.float32))
                accelerator.backward(loss)
            else:
                with torch.no_grad():
                    model(x)
        if i >= warmup - 1:
            break

    accelerator.wait_for_everyone()

    t0 = time.perf_counter()
    batches_done = 0
    for i, (x, y) in enumerate(dataloader):
        if model is not None:
            if backward:
                logits = model(x)
                loss = criterion(logits, y.unsqueeze(1).to(torch.float32))
                accelerator.backward(loss)
                model.zero_grad()
            else:
                with torch.no_grad():
                    model(x)
        batches_done += 1
        if batches_done >= num_batches:
            break

    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - t0

    return elapsed, batches_done


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    clf = config.get('classifier', {})
    prof = config.get('profile', {})
    batch_size = clf.get('batch_size', 2048)
    ds_limit = clf.get('ds_limit', 8_000_000)
    num_batches = prof.get('num_batches', 500)
    num_workers = prof.get('num_workers', 4)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='no',
        kwargs_handlers=[ddp_kwargs],
    )

    ds = LabeledMemmapDataset(
        config['pos_data_train'], config['neg_data_train'],
        limit=ds_limit,
    )

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model = DirectClassifier(
        d_model=clf.get('d_model', 128),
        n_layers=clf.get('n_layers', 4),
        n_head=clf.get('n_head', 4),
        max_len=clf.get('context', 32),
    )

    model, dl = accelerator.prepare(model, dl)

    if accelerator.is_main_process:
        print(f"Dataset size: {len(ds):,} samples")
        print(f"Batch size (per GPU): {batch_size}")
        print(f"Num GPUs: {accelerator.num_processes}")
        print(f"Num workers (per GPU): {num_workers}")
        print(f"Total workers: {num_workers * accelerator.num_processes}")
        print(f"Batches per phase: {num_batches}")
        print()

    # --- Phase 1: DataLoader only ---
    model.eval()
    elapsed, done = run_phase(accelerator, dl, num_batches, model=None)
    if accelerator.is_main_process:
        its = done / elapsed
        sps = (done * batch_size) / elapsed
        print(f"Phase 1 — DataLoader only:           {elapsed:7.2f}s  {its:7.1f} it/s  {sps:>10,.0f} samples/s")

    # --- Phase 2: DataLoader + forward ---
    model.eval()
    elapsed, done = run_phase(accelerator, dl, num_batches, model=model, backward=False)
    if accelerator.is_main_process:
        its = done / elapsed
        sps = (done * batch_size) / elapsed
        print(f"Phase 2 — DataLoader + forward:       {elapsed:7.2f}s  {its:7.1f} it/s  {sps:>10,.0f} samples/s")

    # --- Phase 3: DataLoader + forward + backward ---
    model.train()
    elapsed, done = run_phase(accelerator, dl, num_batches, model=model, backward=True)
    if accelerator.is_main_process:
        its = done / elapsed
        sps = (done * batch_size) / elapsed
        print(f"Phase 3 — DataLoader + fwd + bwd:     {elapsed:7.2f}s  {its:7.1f} it/s  {sps:>10,.0f} samples/s")

    if accelerator.is_main_process:
        print()
        print("Compare Phase 1 here vs profile_io (single GPU) to see DDP/DistributedSampler overhead.")
        print("Compare Phase 1 vs 2 vs 3 to see IO vs compute vs gradient breakdown.")


if __name__ == "__main__":
    main()
