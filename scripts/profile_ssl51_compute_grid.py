"""Compute-side optimisation sweep for SSL SimCLR on H100.

Fixes IO at the established winner from profile_ssl51_dataloader_grid
(chunked sampler, read_len=64, ~8.5 it/s in fp32) and progressively
layers compute optimisations to find the throughput ceiling for d128_L4
on 8xH100 SXM:

  Cell  Change vs previous         Hypothesis
  -----------------------------------------------------------
  1     fp32 baseline              Reproduce prior profiler's best IO cell
  2     + bf16 autocast            ~2x from H100 bf16 tensor cores
  3     + torch.compile            Kernel fusion, reduced launch overhead
  4     + compile(reduce-overhead) CUDA-graph fragments via dynamo
  5     + fused AdamW              Fuse param update into single kernel
  6     + workers=8                More prefetch parallelism
  7     + bs=1024                  Amortise fixed per-step overhead
  8     + bs=2048                  Further amortisation (OOM risk)

Each cell is additive: cell N includes all optimisations from cell N-1.
Phase A (pure dataloader) is measured for all cells so IO/compute
interactions surface; Phase B5 is the real training step.

Submission (8xH100, ~45 min):
    sbatch --account=cu_0030 --cpus-per-task=64 --mem=448gb \\
           --time=01:00:00 --gres=gpu:8 --job-name=profile_compute \\
           --output=scripts/profile_ssl51_compute_grid.%j.out \\
           --wrap="source .venv/bin/activate && cd $(pwd) && \\
                   accelerate launch --num_processes=8 --mixed_precision=no \\
                     scripts/profile_ssl51_compute_grid.py"
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
from smrt_foundation.augment import AugmentationPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT = 32
MEMMAP_PATH = 'data/01_processed/ssl_sets/ob007_raw.memmap'
FULL_READ_LEN = 4096

READ_LEN = 64          # IO winner from prior profiler
CHUNK_SIZE = 2048       # ChunkedRandomSampler chunk size
PREFETCH_FACTOR = 4

N_WARMUP = 30           # baseline warmup (non-compile phases)
N_COMPILE_EXTRA = 10    # extra warmup for compiled models
MEASURE_SECS = 60       # target wall time per phase
N_SUB_WINDOWS = 4       # drift detection sub-windows
PRE_TIME_STEPS = 20     # steps used to estimate step time

# Model (d128_L4 — most IO-sensitive, smallest compute gap)
D_MODEL = 128
N_LAYERS = 4
N_HEAD = 2

# (label, config_dict)
CELLS = [
    ("fp32 baseline",
     dict(amp=False, compile_mode=None,              fused=False, workers=4,  bs=512)),
    ("+ bf16",
     dict(amp=True,  compile_mode=None,              fused=False, workers=4,  bs=512)),
    ("+ compile",
     dict(amp=True,  compile_mode='default',         fused=False, workers=4,  bs=512)),
    ("+ reduce-overhead",
     dict(amp=True,  compile_mode='reduce-overhead', fused=False, workers=4,  bs=512)),
    ("+ fused AdamW",
     dict(amp=True,  compile_mode='reduce-overhead', fused=True,  workers=4,  bs=512)),
    ("+ workers=8",
     dict(amp=True,  compile_mode='reduce-overhead', fused=True,  workers=8,  bs=512)),
    ("+ bs=1024",
     dict(amp=True,  compile_mode='reduce-overhead', fused=True,  workers=8,  bs=1024)),
    ("+ bs=2048",
     dict(amp=True,  compile_mode='reduce-overhead', fused=True,  workers=8,  bs=2048)),
]


# ---------------------------------------------------------------------------
# SubcropShardedMemmapDataset (copied from profile_ssl51_dataloader_grid
# to avoid fragile cross-profiler imports)
# ---------------------------------------------------------------------------

class SubcropShardedMemmapDataset(Dataset):
    """ShardedMemmapDataset sibling that materialises only a random
    `read_len`-position window per __getitem__, cutting bytes-per-sample
    from 4096*9 to read_len*9."""

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
            self.memmaps[shard_idx] = np.load(
                self.shard_paths[shard_idx], mmap_mode='r',
            )
        else:
            self.memmaps.move_to_end(shard_idx)
        start = random.randint(0, self._max_start) if self._max_start > 0 else 0
        window = self.memmaps[shard_idx][local_idx, start:start + self.read_len]
        return torch.from_numpy(np.array(window)).float()


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def make_policy():
    """Augmentation policy matching ssl_51 configs."""
    return AugmentationPolicy(
        CONTEXT,
        rc_lookup=None,
        revcomp_p=0.0,
        channel_dropout_p=0.2,
        gaussian_noise_p=0.8, gaussian_noise_sigma=0.1,
        blur_p=0.5, blur_sigma_range=(0.2, 2.0),
    )


def build_dl(paired_ds, batch_size, num_workers):
    sampler = ChunkedRandomSampler(paired_ds, CHUNK_SIZE, shuffle_within=True)
    return DataLoader(
        paired_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, prefetch_factor=PREFETCH_FACTOR,
        shuffle=False, sampler=sampler,
        persistent_workers=True,
    )


def timed_sustained(step_fn, measure_secs, n_sub_windows, accelerator,
                    n_warmup=N_WARMUP):
    """Sustained-throughput measurement safe under DDP.

    Every phase uses a fixed step count identical across ranks to keep
    DDP collective ordering in lockstep. See
    profile_ssl51_dataloader_grid.timed_sustained for the full rationale.
    """
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Fixed-count warmup
    for _ in range(n_warmup):
        step_fn()
    _sync()

    # Fixed-count pre-timing
    pre_t0 = time.perf_counter()
    for _ in range(PRE_TIME_STEPS):
        step_fn()
    _sync()
    pre_elapsed = time.perf_counter() - pre_t0
    est_step = pre_elapsed / PRE_TIME_STEPS if pre_elapsed > 0 else 0.1

    # Broadcast target from rank 0 so all ranks agree
    target_local = max(
        PRE_TIME_STEPS * 2,
        int(measure_secs / est_step) if est_step > 0 else PRE_TIME_STEPS * 2,
    )
    target_tensor = torch.tensor(
        [int(target_local)], device=accelerator.device, dtype=torch.long,
    )
    if dist.is_initialized():
        dist.broadcast(target_tensor, src=0)
    target_steps = int(target_tensor[0].item())

    if accelerator.is_main_process:
        print(
            f"    pre-time: est {est_step*1000:.1f} ms/step, "
            f"target {target_steps} steps "
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
    t = torch.tensor(
        [float(v) for v in xs], device=accelerator.device, dtype=torch.float32,
    )
    return accelerator.reduce(t, reduction='mean').tolist()


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_cell(label, cfg, paired_ds, accelerator):
    """Build model/optimizer/dataloader for this cell's config, measure
    Phase A (pure dataloader) and Phase B5 (real training step), tear
    everything down cleanly."""

    is_main = accelerator.is_main_process
    use_amp = cfg['amp']
    compile_mode = cfg['compile_mode']
    fused = cfg['fused']
    workers = cfg['workers']
    bs = cfg['bs']

    # --- Model ---
    model = SimCLRSmrt(
        d_model=D_MODEL, n_layers=N_LAYERS, n_head=N_HEAD,
        max_len=CONTEXT, projection_dim=128, projection_layers=2,
    )
    if compile_mode is not None:
        model = torch.compile(model, mode=compile_mode)

    # Move to GPU before optimizer so fused=True can see CUDA params
    model = model.to(accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4, fused=fused,
    )
    criterion = NTXent(temperature=0.1)

    dl = build_dl(paired_ds, bs, workers)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    # Reset peak memory tracker
    torch.cuda.reset_peak_memory_stats(accelerator.device)

    # Warmup for compile cells: first call triggers compilation
    warmup = N_WARMUP + (N_COMPILE_EXTRA if compile_mode is not None else 0)

    # --- Phase A: pure next(it), no GPU compute ---
    it_a = iter(dl)

    def _a():
        next(it_a)

    if is_main:
        print("  phase A (pure next(it))", flush=True)
    A_its_raw, A_sub_raw = timed_sustained(
        _a, MEASURE_SECS, N_SUB_WINDOWS, accelerator, n_warmup=N_WARMUP,
    )

    # Drop phase-A iterator, rebuild for B5
    del it_a
    gc.collect()
    dl_b5 = build_dl(paired_ds, bs, workers)
    dl_b5 = accelerator.prepare(dl_b5)
    it_b5 = iter(dl_b5)

    # --- Phase B5: real training step ---
    if use_amp:
        def _b5():
            u1, u2 = next(it_b5)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                z1, z2 = model(u1, u2)
                loss = criterion(z1, z2)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
    else:
        def _b5():
            u1, u2 = next(it_b5)
            z1, z2 = model(u1, u2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

    if is_main:
        print("  phase B5 (real training step)", flush=True)
    B5_its_raw, B5_sub_raw = timed_sustained(
        _b5, MEASURE_SECS, N_SUB_WINDOWS, accelerator, n_warmup=warmup,
    )

    peak_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 2)

    # Cross-rank means
    A_its = reduce_scalar(accelerator, A_its_raw)
    B5_its = reduce_scalar(accelerator, B5_its_raw)
    A_sub = reduce_list(accelerator, A_sub_raw)
    B5_sub = reduce_list(accelerator, B5_sub_raw)
    peak_mem = reduce_scalar(accelerator, peak_mem_mb)

    # Cleanup
    del model, optimizer, dl, dl_b5, it_b5, criterion
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        label=label, bs=bs, workers=workers,
        amp=use_amp, compile_mode=compile_mode, fused=fused,
        A_its=A_its, B5_its=B5_its,
        A_sub=A_sub, B5_sub=B5_sub,
        peak_mem_mb=peak_mem,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='no', kwargs_handlers=[ddp_kwargs],
    )
    set_seed(42)
    is_main = accelerator.is_main_process

    if is_main:
        print("=== SSL-51 SimCLR compute optimisation sweep "
              "(d128_L4, 8xH100) ===", flush=True)
        print(
            f"World size: {accelerator.num_processes}  "
            f"device: {accelerator.device}  "
            f"IO fixed at: chunked sampler, read_len={READ_LEN}, "
            f"chunk_size={CHUNK_SIZE}",
            flush=True,
        )
        print(
            f"Per cell: {N_WARMUP}(+{N_COMPILE_EXTRA} compile) warmup + "
            f"{PRE_TIME_STEPS} pre-time + {MEASURE_SECS}s measurement "
            f"({N_SUB_WINDOWS} sub-windows). "
            f"All step counts fixed across ranks for DDP safety.",
            flush=True,
        )
        print(f"{len(CELLS)} cells to run.\n", flush=True)

    # --- Dataset (shared across cells — IO config is fixed) ---
    t0 = time.perf_counter()
    ds = SubcropShardedMemmapDataset(MEMMAP_PATH, read_len=READ_LEN, limit=0)
    if is_main:
        print(f"SSL dataset: {len(ds):,} reads from {MEMMAP_PATH}", flush=True)
    norm = KineticsNorm(ShardedMemmapDataset(MEMMAP_PATH, limit=0),
                        max_samples=16_384)
    if is_main:
        print(
            f"Norm stats (means={norm.means}, stds={norm.stds}) "
            f"computed in {time.perf_counter()-t0:.1f}s\n",
            flush=True,
        )
    policy = make_policy()
    paired_ds = PairedViewDataset(ds, policy=policy, norm_fn=norm)

    # --- Run cells ---
    rows = []
    for label, cfg in CELLS:
        if is_main:
            parts = []
            parts.append(f"bf16" if cfg['amp'] else "fp32")
            if cfg['compile_mode']:
                parts.append(f"compile({cfg['compile_mode']})")
            if cfg['fused']:
                parts.append("fused-AdamW")
            parts.append(f"w={cfg['workers']}")
            parts.append(f"bs={cfg['bs']}")
            print(
                f"--- cell: {label} [{', '.join(parts)}] ---",
                flush=True,
            )
        try:
            r = run_cell(label, cfg, paired_ds, accelerator)
            rows.append(r)
            if is_main:
                sps = r['B5_its'] * r['bs']
                print(
                    f"  A  it/s: {r['A_its']:6.2f}   "
                    f"sub-windows: "
                    f"{', '.join(f'{v:5.1f}' for v in r['A_sub'])}",
                    flush=True,
                )
                print(
                    f"  B5 it/s: {r['B5_its']:6.2f}   "
                    f"sub-windows: "
                    f"{', '.join(f'{v:5.1f}' for v in r['B5_sub'])}",
                    flush=True,
                )
                print(
                    f"  B5 samples/s: {sps:,.0f}   "
                    f"peak mem: {r['peak_mem_mb']:,.0f} MB\n",
                    flush=True,
                )
        except Exception as e:
            if is_main:
                print(f"  [ERROR] {type(e).__name__}: {e}\n", flush=True)
            accelerator.free_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Summary tables ---
    if not is_main or not rows:
        return

    baseline_b5 = rows[0]['B5_its'] if rows else 1.0

    print("\n=== Per-cell sustained throughput ===", flush=True)
    hdr = (
        f"{'cell':<25}  "
        f"{'A':>6} {'B5':>6} {'samp/s':>8}  "
        f"{'B5 drift (1st->4th)':>22}  "
        f"{'mem MB':>7}  "
        f"{'vs base':>7}"
    )
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)
    for r in rows:
        B5_drift = f"{r['B5_sub'][0]:5.1f} -> {r['B5_sub'][-1]:5.1f}"
        sps = r['B5_its'] * r['bs']
        speedup = r['B5_its'] / baseline_b5 if baseline_b5 > 0 else 0
        print(
            f"{r['label']:<25}  "
            f"{r['A_its']:>6.2f} {r['B5_its']:>6.2f} {sps:>8,.0f}  "
            f"{B5_drift:>22}  "
            f"{r['peak_mem_mb']:>7,.0f}  "
            f"{speedup:>6.2f}x",
            flush=True,
        )

    print("\n=== Ranked by B5 samples/s (best first) ===", flush=True)
    ranked = sorted(rows, key=lambda r: -(r['B5_its'] * r['bs']))
    baseline_sps = baseline_b5 * rows[0]['bs']
    print(
        f"{'rank':>4}  {'cell':<25}  "
        f"{'B5 it/s':>8}  {'samp/s':>8}  {'vs base':>8}",
        flush=True,
    )
    for i, r in enumerate(ranked):
        sps = r['B5_its'] * r['bs']
        speedup = sps / baseline_sps if baseline_sps > 0 else 0
        print(
            f"{i+1:>4}  {r['label']:<25}  "
            f"{r['B5_its']:>8.2f}  {sps:>8,.0f}  "
            f"{speedup:>7.2f}x",
            flush=True,
        )

    # Estimate wall time for 100 epochs at best config
    best = ranked[0]
    best_sps = best['B5_its'] * best['bs']
    n_samples = len(paired_ds)
    epoch_secs = n_samples / (best_sps * accelerator.num_processes)
    print(
        f"\nBest config: {best['label']}\n"
        f"  {best_sps:,.0f} samples/s/rank x {accelerator.num_processes} ranks "
        f"= {best_sps * accelerator.num_processes:,.0f} global samples/s\n"
        f"  {n_samples:,} samples/epoch -> "
        f"{epoch_secs/60:.1f} min/epoch, "
        f"{100*epoch_secs/3600:.1f} h for 100 epochs",
        flush=True,
    )


if __name__ == '__main__':
    main()
