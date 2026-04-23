"""Grid-sweep IO interventions for SSL-51 SimCLR training throughput (d128 only).

Follow-up to the prior DL-config grid. After that sweep shipped into
_shared_train.py (num_workers 8->4, window-averaged stdout it/s), real
d128 training honestly measured ~1.6 it/s sustained. Two sharpening
observations from that run:

  - Node memory sits at ~4% of 448 GB. Nothing is thrashing page cache;
    disk IO bandwidth is almost certainly not the bottleneck.
  - Every __getitem__ reads 36 KB per sample (4096 positions * 9 features
    uint8) but AugmentationPolicy.random_subcrop immediately throws
    99.2% of that away. ~18 MB per bs=512 batch crosses the Python /
    numpy / IPC pipeline when only ~144 KB is actually needed — a 128x
    over-read that scales numpy copy, float32 promotion, and worker-to-
    main tensor serialization all linearly.

This sweep tests the hypothesis that cutting bytes-per-sample at
__getitem__ time (via a SubcropShardedMemmapDataset sibling) gives a
near-proportional throughput recovery. A ChunkedRandomSampler sanity
cell is included as a falsification check — if sampler alone helps,
the primary hypothesis is wrong and the bottleneck is random disk
seeks after all.

Submission (via run.sh):
    bash run.sh scripts/experiments/profile_ssl51_dataloader_grid
"""
import gc
import glob
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from smrt_foundation.dataset import (
    ShardedMemmapDataset, PairedViewDataset, ChunkedRandomSampler,
)
from smrt_foundation.model import SimCLRSmrt
from smrt_foundation.loss import NTXent
from smrt_foundation.normalization import KineticsNorm

from scripts.profile_ssl51_compute_bound import (
    make_policy, CONTEXT, MEMMAP_PATH,
)


# d128 only — IO cost is per-sample and model-independent. d128 is the
# most sensitive: smallest compute (17.7 ms), biggest gap vs dataloader.
SIZE = dict(name='d128_L4', d_model=128, n_layers=4, n_head=2, batch_size=512)

FULL_READ_LEN = 4096    # native stored context in ob007_raw memmap
CHUNK_SIZE = 2048       # for ChunkedRandomSampler cells
NUM_WORKERS = 4         # rank 1 from the prior CPU-scheduling grid
PREFETCH_FACTOR = 4
N_WARMUP_STEPS = 30     # fixed step count (never wall-clock — see timed_sustained)
MEASURE_SECS = 60
N_SUB_WINDOWS = 4       # 15-sec sub-windows for drift detection

# (label, sampler_kind, read_len)
GRID = [
    ("baseline (shuffle, read=4096)",     "shuffle", 4096),
    ("shuffle, read=256",                 "shuffle", 256),
    ("shuffle, read=64",                  "shuffle", 64),
    ("shuffle, read=32 (no aug room)",    "shuffle", 32),
    ("chunked, read=4096",                "chunked", 4096),
    ("chunked, read=64",                  "chunked", 64),
]


class SubcropShardedMemmapDataset(Dataset):
    """Sibling of ShardedMemmapDataset that materialises only a random
    `read_len`-position window per __getitem__.

    Shard-loading and LRU-caching mirror `ShardedMemmapDataset`
    (smrt_foundation/dataset.py:33-60); the difference is that we slice
    the memmap row *before* `np.array()`, so the copy is `read_len * 9`
    bytes instead of the full `4096 * 9`.

    `read_len == FULL_READ_LEN` returns the full row (behaviourally
    identical to `ShardedMemmapDataset`; not used here since the
    baseline cell uses the real class). `read_len < FULL_READ_LEN`
    picks a uniform random start in `[0, FULL_READ_LEN - read_len]`
    per call.

    The downstream `AugmentationPolicy.random_subcrop` still runs; at
    `read_len=64` each view has `64 - 32 + 1 = 33` valid start
    positions (vs 4065 from the full row), which is the augmentation-
    diversity trade-off to watch if we adopt this for real training.
    """

    def __init__(self, data_dir, read_len, cache_size=100, limit=0):
        expanded_dir = os.path.expandvars(data_dir)
        self.shard_paths = sorted(glob.glob(os.path.join(expanded_dir, "*.npy")))
        first_shard = np.load(self.shard_paths[0], mmap_mode='r')
        self.shard_size = first_shard.shape[0]
        self.full_read_len = first_shard.shape[1]
        last_shard = np.load(self.shard_paths[-1], mmap_mode='r')
        full_len = ((len(self.shard_paths) - 1) * self.shard_size) + last_shard.shape[0]
        self.len = full_len if limit == 0 else min(full_len, limit)
        self.cache_size = cache_size
        self.memmaps = OrderedDict()
        self.read_len = int(read_len)
        if self.read_len > self.full_read_len:
            raise ValueError(
                f"read_len={self.read_len} > shard full length {self.full_read_len}"
            )
        self._max_start = self.full_read_len - self.read_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.len:
            raise IndexError(idx)
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        if shard_idx not in self.memmaps:
            if len(self.memmaps) >= self.cache_size:
                self.memmaps.popitem(last=False)
            self.memmaps[shard_idx] = np.load(self.shard_paths[shard_idx], mmap_mode='r')
        else:
            self.memmaps.move_to_end(shard_idx)
        start = random.randint(0, self._max_start) if self._max_start > 0 else 0
        window = self.memmaps[shard_idx][local_idx, start:start + self.read_len]
        return torch.from_numpy(np.array(window)).float()


def build_dataset(read_len):
    # Baseline cell uses the real production class so cell 1 is a
    # true apples-to-apples reproduction of real training's code path.
    if read_len == FULL_READ_LEN:
        return ShardedMemmapDataset(MEMMAP_PATH, limit=0)
    return SubcropShardedMemmapDataset(MEMMAP_PATH, read_len=read_len, limit=0)


def build_dl(paired_ds, batch_size, sampler_kind):
    if sampler_kind == "chunked":
        sampler = ChunkedRandomSampler(paired_ds, CHUNK_SIZE, shuffle_within=True)
        shuffle = False
    elif sampler_kind == "shuffle":
        sampler = None
        shuffle = True
    else:
        raise ValueError(sampler_kind)
    return DataLoader(
        paired_ds, batch_size=batch_size, num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
        shuffle=shuffle, sampler=sampler,
        persistent_workers=True,
    )


PRE_TIME_STEPS = 20


def timed_sustained(step_fn, measure_secs, n_sub_windows, accelerator):
    """Sustained-throughput measurement that is safe under DDP.

    `step_fn` may contain DDP collectives (NTXent's dist_nn.all_gather
    in the loss, DDP's gradient all_reduce in backward) and Accelerate's
    DataLoaderShard may fire a broadcast at batch boundaries in some
    paths. *Any* wall-clock-bounded loop would let different ranks run
    different step counts, which makes rank N post collective K while
    rank M posts a different collective K — NCCL hangs 10 minutes then
    kills the job.

    Every phase uses a fixed step count identical across ranks:
      1. Warmup: N_WARMUP_STEPS — absorbs worker spawn and cache warm.
      2. Pre-time: PRE_TIME_STEPS — estimate step_time.
      3. Broadcast rank 0's `target_steps = measure_secs / step_time`
         so all ranks agree.
      4. Measurement: exactly `target_steps`, timed end-to-end, with
         equal-sized sub-windows for drift detection.
    """
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Fixed-count warmup
    for _ in range(N_WARMUP_STEPS):
        step_fn()
    _sync()

    # Fixed-count pre-timing
    pre_t0 = time.perf_counter()
    for _ in range(PRE_TIME_STEPS):
        step_fn()
    _sync()
    pre_elapsed = time.perf_counter() - pre_t0
    est_step = pre_elapsed / PRE_TIME_STEPS if pre_elapsed > 0 else 0.1

    # Decide target on rank 0, broadcast so everyone agrees despite
    # per-rank roundoff in est_step.
    target_local = max(
        PRE_TIME_STEPS * 2,
        int(measure_secs / est_step) if est_step > 0 else PRE_TIME_STEPS * 2,
    )
    target_tensor = torch.tensor([int(target_local)], device=accelerator.device, dtype=torch.long)
    if dist.is_initialized():
        dist.broadcast(target_tensor, src=0)
    target_steps = int(target_tensor[0].item())

    if accelerator.is_main_process:
        print(
            f"    pre-time: est {est_step*1000:.1f} ms/step, target {target_steps} steps "
            f"(expected ~{target_steps * est_step:.0f}s)",
            flush=True,
        )

    # Fixed-count measurement with sub-window drift tracking
    window_size = max(1, target_steps // n_sub_windows)
    sub_its = []
    _sync()
    t0 = time.perf_counter()
    cur_window_start = t0
    steps_in_window = 0
    for _ in range(target_steps):
        step_fn()
        steps_in_window += 1
        if steps_in_window >= window_size and len(sub_its) < n_sub_windows - 1:
            now = time.perf_counter()
            sub_its.append(
                steps_in_window / (now - cur_window_start)
                if now > cur_window_start else 0.0
            )
            cur_window_start = now
            steps_in_window = 0
    _sync()
    total_end = time.perf_counter()
    if steps_in_window > 0:
        sub_its.append(
            steps_in_window / (total_end - cur_window_start)
            if total_end > cur_window_start else 0.0
        )
    while len(sub_its) < n_sub_windows:
        sub_its.append(0.0)
    sub_its = sub_its[:n_sub_windows]

    total_elapsed = total_end - t0
    total_its = target_steps / total_elapsed if total_elapsed > 0 else 0.0
    return total_its, sub_its


def reduce_scalar(accelerator, x):
    t = torch.tensor(float(x), device=accelerator.device, dtype=torch.float32)
    return accelerator.reduce(t, reduction='mean').item()


def reduce_list(accelerator, xs):
    t = torch.tensor([float(v) for v in xs], device=accelerator.device, dtype=torch.float32)
    return accelerator.reduce(t, reduction='mean').tolist()


def run_cell(label, sampler_kind, read_len, norm, accelerator):
    """Run phases A (pure next(it)) and B5 (real training step) for one
    cell. Everything is built fresh and torn down cleanly so the next
    cell starts with no inherited state.
    """
    ds = build_dataset(read_len)
    policy = make_policy()
    paired_ds = PairedViewDataset(ds, policy=policy, norm_fn=norm)
    dl = build_dl(paired_ds, SIZE['batch_size'], sampler_kind)

    model = SimCLRSmrt(
        d_model=SIZE['d_model'], n_layers=SIZE['n_layers'],
        n_head=SIZE['n_head'], max_len=CONTEXT,
        projection_dim=128, projection_layers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = NTXent(temperature=0.1)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    # --- Phase A: pure next(it) through the prepared DL, no GPU work ---
    it = iter(dl)

    def _a():
        next(it)

    if accelerator.is_main_process:
        print("  phase A (pure next(it))", flush=True)
    A_its_raw, A_sub_raw = timed_sustained(_a, MEASURE_SECS, N_SUB_WINDOWS, accelerator)

    # Drop the phase-A iterator, rebuild a fresh DL for phase B5 so its
    # measurement doesn't start with a phase-A-drained prefetch queue.
    del it
    gc.collect()
    dl2 = build_dl(paired_ds, SIZE['batch_size'], sampler_kind)
    dl2 = accelerator.prepare(dl2)
    it2 = iter(dl2)

    def _b5():
        u1, u2 = next(it2)
        z1, z2 = model(u1, u2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    if accelerator.is_main_process:
        print("  phase B5 (real training step)", flush=True)
    B5_its_raw, B5_sub_raw = timed_sustained(_b5, MEASURE_SECS, N_SUB_WINDOWS, accelerator)

    # Cross-rank means
    A_its = reduce_scalar(accelerator, A_its_raw)
    B5_its = reduce_scalar(accelerator, B5_its_raw)
    A_sub = reduce_list(accelerator, A_sub_raw)
    B5_sub = reduce_list(accelerator, B5_sub_raw)

    bytes_per_batch = read_len * 9 * SIZE['batch_size']

    del model, optimizer, dl, dl2, it2, criterion, paired_ds, ds
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        label=label, sampler=sampler_kind, read_len=read_len,
        A_its=A_its, B5_its=B5_its,
        A_sub=A_sub, B5_sub=B5_sub,
        bytes_per_batch=bytes_per_batch,
    )


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='no', kwargs_handlers=[ddp_kwargs])
    set_seed(42)
    is_main = accelerator.is_main_process

    if is_main:
        print("=== SSL-51 SimCLR IO intervention sweep (d128_L4 only) ===", flush=True)
        print(
            f"World size: {accelerator.num_processes}  "
            f"device: {accelerator.device}  precision: fp32",
            flush=True,
        )
        print(
            f"Per cell: {N_WARMUP_STEPS} warmup steps + {PRE_TIME_STEPS} pre-time steps "
            f"+ {MEASURE_SECS}s-targeted measurement "
            f"(split into {N_SUB_WINDOWS} sub-windows for drift). "
            f"All phases use fixed step counts identical across ranks to keep "
            f"DDP collective ordering in lockstep.",
            flush=True,
        )
        print(
            f"DL constants: num_workers={NUM_WORKERS}, "
            f"prefetch_factor={PREFETCH_FACTOR}, persistent_workers=True "
            f"(the prior grid's rank-1 CPU-scheduling config)",
            flush=True,
        )

    t0 = time.perf_counter()
    norm_ds = ShardedMemmapDataset(MEMMAP_PATH, limit=0)
    if is_main:
        print(f"\nSSL dataset: {len(norm_ds):,} reads from {MEMMAP_PATH}", flush=True)
    norm = KineticsNorm(norm_ds, max_samples=16_384)
    del norm_ds
    gc.collect()
    if is_main:
        print(
            f"Norm stats (means={norm.means}, stds={norm.stds}) "
            f"computed in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    rows = []
    for label, sampler_kind, read_len in GRID:
        if is_main:
            print(f"\n--- cell: {label} ---", flush=True)
        try:
            r = run_cell(label, sampler_kind, read_len, norm, accelerator)
            rows.append(r)
            if is_main:
                print(
                    f"  A  it/s: {r['A_its']:6.2f}   "
                    f"sub-windows: {', '.join(f'{v:5.1f}' for v in r['A_sub'])}",
                    flush=True,
                )
                print(
                    f"  B5 it/s: {r['B5_its']:6.2f}   "
                    f"sub-windows: {', '.join(f'{v:5.1f}' for v in r['B5_sub'])}",
                    flush=True,
                )
                print(
                    f"  bytes/batch: {r['bytes_per_batch']/1e6:.2f} MB",
                    flush=True,
                )
        except Exception as e:
            if is_main:
                print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)
            accelerator.free_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if is_main and rows:
        print("\n\n=== Per-cell sustained throughput ===", flush=True)
        hdr = (
            f"{'cell':<36}  "
            f"{'A':>6} {'B5':>6}  "
            f"{'A drift (1st->4th)':>22}  "
            f"{'B5 drift (1st->4th)':>22}  "
            f"{'bytes/batch':>12}"
        )
        print(hdr, flush=True)
        print('-' * len(hdr), flush=True)
        for r in rows:
            A_drift = f"{r['A_sub'][0]:5.1f} -> {r['A_sub'][-1]:5.1f}"
            B5_drift = f"{r['B5_sub'][0]:5.1f} -> {r['B5_sub'][-1]:5.1f}"
            print(
                f"{r['label']:<36}  "
                f"{r['A_its']:>6.2f} {r['B5_its']:>6.2f}  "
                f"{A_drift:>22}  {B5_drift:>22}  "
                f"{r['bytes_per_batch']/1e6:>9.2f} MB",
                flush=True,
            )

        baseline_b5 = next(
            (r['B5_its'] for r in rows
             if r['read_len'] == FULL_READ_LEN and r['sampler'] == 'shuffle'),
            None,
        )
        print("\n=== Ranked by B5 sustained it/s (best first) ===", flush=True)
        print(
            f"{'rank':>4}  {'cell':<36}  {'B5 it/s':>8}  "
            f"{'vs cell 1':>11}",
            flush=True,
        )
        ranked = sorted(rows, key=lambda r: -r['B5_its'])
        for i, r in enumerate(ranked):
            speedup = (
                r['B5_its'] / baseline_b5
                if baseline_b5 and baseline_b5 > 0
                else float('nan')
            )
            print(
                f"{i+1:>4}  {r['label']:<36}  {r['B5_its']:>8.2f}  "
                f"{speedup:>10.2f}x",
                flush=True,
            )


if __name__ == '__main__':
    main()
