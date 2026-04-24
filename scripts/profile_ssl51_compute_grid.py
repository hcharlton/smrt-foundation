"""GPU-saturation profile for the ssl_53 SimCLR size grid on 8×H100.

Produces a per-size configuration recommendation and a throughput attribution
so future ssl_53-class experiments don't burn cluster time on an
unsaturated pipeline. One SLURM job, four interleaved sweeps.

Sweeps:
  A (all 4 sizes, 3 cells each):
      C0       = ssl_53 production today (full-read, norm-pre, w=4/pf=4,
                 bf16 autocast, no compile, fused=False, bs=production).
      C_ALL    = all fixes (SubcropShardedMemmapDataset r=64, norm-post,
                 w=8/pf=16, compile(reduce-overhead), fused AdamW,
                 set_to_none=True, bs=production).
      C_CEIL   = GPU-cached ceiling (C_ALL compute, one batch replayed
                 from GPU memory — no DL, no worker procs).

  B (d128_L4 only — single-knob attribution ladder from C0 to C_ALL, plus
      diagnostic bs cells):
      C1 +subcrop  → C2 +norm-post  → C3 +w=8/pf=16  → C4 +compile
        → C5 +fused AdamW (== C_ALL at d128)
        → C6 +bs=1024  → C7 +bs=2048  (diagnostic only, see output risks)

  C (d768_L8 only — same ladder as B minus bs variation; bs>256 OOM-risk).

  D (d256_L8, d512_L8 — compile regression guard):
      C_ALL_NOCOMPILE = C_ALL with compile_mode=None. If C_ALL <
      C_ALL_NOCOMPILE at either size, compile is regressing at that
      param count and reduce-overhead mode should not ship for it.

Output (rank 0):
  1. Per-size summary: today it/s | post-fix it/s | ceiling it/s | gain |
     GPU-busy% (post-fix, NVML-sampled on rank 0).
  2. Attribution ladders at d128 and d768.
  3. Compile-regression verdict at d256 and d512.
  4. Copy-paste config snippet per size for _shared_train.py + config.yaml.

Adoption caveats printed at the end of main():
  - Post-subcrop norm requires re-fit KineticsNorm stats in production.
  - bs>512 changes SimCLR loss semantics (Chen 2020 §B.5); diagnostic only.
  - compile(reduce-overhead) + Accelerate resume may diverge first steps.
  - SubcropShardedMemmapDataset read_len=64 assumes context ≤ 64.

Submission (8×H100, ~50 min expected; ask for 1h):
    sbatch --account=cu_0030 --cpus-per-task=64 --mem=448gb \\
           --time=01:00:00 --gres=gpu:8 --job-name=profile_saturation \\
           --output=scripts/profile_ssl51_compute_grid.%j.out \\
           --wrap="source .venv/bin/activate && cd $(pwd) && \\
                   TORCHDYNAMO_NUMWORKERS=1 accelerate launch \\
                     --num_processes=8 --mixed_precision=no \\
                     scripts/profile_ssl51_compute_grid.py"
"""
import gc
import glob
import os
import random
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

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

# NVML is optional — GPU-busy% samples degrade to NaN if unavailable.
try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT = 32
MEMMAP_PATH = 'data/01_processed/ssl_sets/ob007_raw.memmap'
FULL_READ_LEN = 4096

READ_LEN = 64           # IO winner from profile_ssl51_io_interventions
CHUNK_SIZE = 2048       # ChunkedRandomSampler chunk size

# Measurement window tuned for stable p95/p99 at ~5 it/s baseline:
# 60s × ~5 it/s = ~300 steps → ~3 stall events at 1% rate, sufficient
# for percentile stability.
MEASURE_SECS = 60
N_WARMUP = 20
N_COMPILE_EXTRA = 5     # dynamo cache warms after first compiled call
N_SUB_WINDOWS = 4
PRE_TIME_STEPS = 20

# Model grid — matches ssl_53_simclr_grid_r2_step/size_*/config.yaml.
# (label, d_model, n_layers, n_head, prod_bs)
SIZES = [
    ('d128_L4', 128, 4, 2,  512),
    ('d256_L8', 256, 8, 4,  512),
    ('d512_L8', 512, 8, 8,  512),
    ('d768_L8', 768, 8, 12, 256),
]
SIZE_BY_LABEL = {s[0]: s for s in SIZES}


# ---------------------------------------------------------------------------
# SubcropShardedMemmapDataset — materialises only a random read_len window
# per __getitem__ instead of the full 4096×9 read. Cuts bytes-per-sample
# ~64× at read_len=64 and keeps norm work proportional. Still inline here
# (not yet promoted into smrt_foundation.dataset) so the profile remains
# self-contained.
# ---------------------------------------------------------------------------

class SubcropShardedMemmapDataset(Dataset):
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
        if self.read_len < CONTEXT:
            raise ValueError(
                f"read_len={self.read_len} < CONTEXT={CONTEXT}; policy subcrop "
                f"would overrun. Raise read_len or lower CONTEXT."
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
# PostNormPairedView — PairedViewDataset variant applying norm_fn AFTER the
# augmentation policy's subcrop, not before. Diagnostic only: KineticsNorm
# is fit on the full-read distribution, so applying it to 32-position crops
# yields correct stats only in expectation. Throughput is stats-invariant,
# so this is safe for the profile; production adoption would require
# re-fitting norm on subcrops.
# ---------------------------------------------------------------------------

class PostNormPairedView(Dataset):
    def __init__(self, inner, policy, norm_fn):
        self.inner = inner
        self.policy = policy
        self.norm_fn = norm_fn

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x = self.inner[idx]
        if isinstance(x, (tuple, list)):
            x = x[0]
        # random_subcrop returns a .clone()d tensor, so in-place norm_fn is safe.
        v1 = self.norm_fn(self.policy._one_view(x))
        v2 = self.norm_fn(self.policy._one_view(x))
        return v1, v2


# ---------------------------------------------------------------------------
# NvmlSampler — rank-0 only. Spawns a daemon thread sampling GPU util at
# 20 Hz. Call start() before the measurement window, stop_and_mean() after.
# Returns NaN if pynvml is unavailable.
# ---------------------------------------------------------------------------

class NvmlSampler:
    def __init__(self, device_idx=0, interval_s=0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        self.handle = None
        self.available = False
        if _NVML_AVAILABLE and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                self.available = True
            except Exception:
                self.available = False

    def start(self):
        if not self.available:
            return
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        if pynvml is None:
            return
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.samples.append(float(u.gpu))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def stop_and_mean(self):
        if not self.available or self._thread is None:
            return float('nan')
        self._stop.set()
        self._thread.join(timeout=1.0)
        if not self.samples:
            return float('nan')
        return sum(self.samples) / len(self.samples)


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

def cell(label, *, subcrop, norm_location, workers, prefetch,
         amp=True, compile_mode=None, fused=False,
         bs_override=None, gpu_cached=False):
    """Build a cell config dict. amp defaults to True because ssl_53 runs
    bf16; set False only to measure the fp32 floor.
    """
    return dict(
        label=label, subcrop=subcrop, norm_location=norm_location,
        workers=workers, prefetch=prefetch, amp=amp,
        compile_mode=compile_mode, fused=fused,
        bs_override=bs_override, gpu_cached=gpu_cached,
    )


def build_sweeps():
    """Return list of (sweep_name, sizes_list, cells_list) tuples."""

    c0_baseline = cell(
        'C0 baseline (ssl_53 today)',
        subcrop=False, norm_location='pre',
        workers=4, prefetch=4,
        compile_mode=None, fused=False,
    )

    c_all = cell(
        'C_ALL all fixes',
        subcrop=True, norm_location='post',
        workers=8, prefetch=16,
        compile_mode='reduce-overhead', fused=True,
    )

    c_ceil = cell(
        'C_CEIL GPU-cached ceiling',
        subcrop=True, norm_location='post',
        workers=0, prefetch=2,  # unused for gpu_cached
        compile_mode='reduce-overhead', fused=True,
        gpu_cached=True,
    )

    sweep_A = ('A', [s[0] for s in SIZES], [c0_baseline, c_all, c_ceil])

    # Sweep B: d128 attribution — single-knob deltas from C0 to C_ALL + bs
    b_cells = [
        cell('C1 + subcrop',
             subcrop=True, norm_location='pre',
             workers=4, prefetch=4, compile_mode=None, fused=False),
        cell('C2 + norm-post',
             subcrop=True, norm_location='post',
             workers=4, prefetch=4, compile_mode=None, fused=False),
        cell('C3 + w=8 pf=16',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16, compile_mode=None, fused=False),
        cell('C4 + compile(reduce-overhead)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode='reduce-overhead', fused=False),
        cell('C5 + fused AdamW (== C_ALL@d128)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode='reduce-overhead', fused=True),
        cell('C6 + bs=1024 (diagnostic)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode='reduce-overhead', fused=True,
             bs_override=1024),
        cell('C7 + bs=2048 (diagnostic)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode='reduce-overhead', fused=True,
             bs_override=2048),
    ]
    sweep_B = ('B', ['d128_L4'], b_cells)

    # Sweep C: d768 attribution — same ladder, no bs variation
    c_cells = [
        cell('C1 + subcrop',
             subcrop=True, norm_location='pre',
             workers=4, prefetch=4, compile_mode=None, fused=False),
        cell('C2 + norm-post',
             subcrop=True, norm_location='post',
             workers=4, prefetch=4, compile_mode=None, fused=False),
        cell('C3 + w=8 pf=16',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16, compile_mode=None, fused=False),
        cell('C4 + compile(reduce-overhead)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode='reduce-overhead', fused=False),
        cell('C5 + fused AdamW (== C_ALL@d768)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode='reduce-overhead', fused=True),
    ]
    sweep_C = ('C', ['d768_L8'], c_cells)

    # Sweep D: d256/d512 compile regression guard
    d_cells = [
        cell('C_ALL_NOCOMPILE (all fixes − compile)',
             subcrop=True, norm_location='post',
             workers=8, prefetch=16,
             compile_mode=None, fused=True),
    ]
    sweep_D = ('D', ['d256_L8', 'd512_L8'], d_cells)

    return [sweep_A, sweep_B, sweep_C, sweep_D]


# ---------------------------------------------------------------------------
# Dataset + policy builders — the cell-to-dataset decision lives here
# ---------------------------------------------------------------------------

def make_policy():
    """Augmentation policy matching ssl_53 configs (revcomp off)."""
    return AugmentationPolicy(
        CONTEXT,
        rc_lookup=None,
        revcomp_p=0.0,
        channel_dropout_p=0.2,
        gaussian_noise_p=0.8, gaussian_noise_sigma=0.1,
        blur_p=0.5, blur_sigma_range=(0.2, 2.0),
    )


# Cache datasets across cells — memmaps are lazy so re-using is cheap.
_DATASET_CACHE = {}

def get_inner_dataset(subcrop):
    """Return a cached inner dataset of the requested kind."""
    key = ('subcrop', READ_LEN) if subcrop else ('full',)
    if key not in _DATASET_CACHE:
        if subcrop:
            _DATASET_CACHE[key] = SubcropShardedMemmapDataset(
                MEMMAP_PATH, read_len=READ_LEN, limit=0,
            )
        else:
            _DATASET_CACHE[key] = ShardedMemmapDataset(MEMMAP_PATH, limit=0)
    return _DATASET_CACHE[key]


def build_paired_dataset(cell_cfg, norm):
    inner = get_inner_dataset(cell_cfg['subcrop'])
    policy = make_policy()
    if cell_cfg['norm_location'] == 'post':
        return PostNormPairedView(inner, policy, norm)
    else:
        return PairedViewDataset(inner, policy=policy, norm_fn=norm)


def build_dl(paired_ds, batch_size, num_workers, prefetch_factor):
    sampler = ChunkedRandomSampler(paired_ds, CHUNK_SIZE, shuffle_within=True)
    # pin_memory_device='cuda' enables the non-blocking H2D path with
    # less CPU overhead than the default pin worker.
    return DataLoader(
        paired_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, pin_memory_device='cuda',
        prefetch_factor=prefetch_factor,
        shuffle=False, sampler=sampler,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def timed_sustained(step_fn, measure_secs, n_sub_windows, accelerator,
                    n_warmup, nvml_sampler=None):
    """Sustained-throughput measurement safe under DDP.

    Every phase uses a fixed step count identical across ranks to keep
    DDP collective ordering in lockstep. Optionally takes an NvmlSampler
    to record GPU-busy% over the measurement window.

    Returns (total_its, sub_its, step_times_ms, gpu_busy_pct).
    """
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(n_warmup):
        step_fn()
    _sync()

    pre_t0 = time.perf_counter()
    for _ in range(PRE_TIME_STEPS):
        step_fn()
    _sync()
    pre_elapsed = time.perf_counter() - pre_t0
    est_step = pre_elapsed / PRE_TIME_STEPS if pre_elapsed > 0 else 0.1

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

    if nvml_sampler is not None:
        nvml_sampler.start()

    window_size = max(1, target_steps // n_sub_windows)
    sub_its = []
    step_times = []
    _sync()
    t0 = time.perf_counter()
    cur_window_start = t0
    steps_in_window = 0
    for _ in range(target_steps):
        _sync()
        step_t0 = time.perf_counter()
        step_fn()
        _sync()
        step_times.append((time.perf_counter() - step_t0) * 1000.0)

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

    gpu_busy_pct = nvml_sampler.stop_and_mean() if nvml_sampler is not None else float('nan')

    total_elapsed = total_end - t0
    total_its = target_steps / total_elapsed if total_elapsed > 0 else 0.0
    return total_its, sub_its, step_times, gpu_busy_pct


def percentiles(times_ms):
    """p50/p95/p99/max/stdev/stall_pct from per-step times in ms.

    stall_pct = fraction of steps taking > 2× p50. A steady pipeline has
    stall_pct ~ 0; a bursty one has stall_pct >> 0.
    """
    if not times_ms:
        return dict(p50=0, p95=0, p99=0, max=0, stdev=0, stall_pct=0)
    s = sorted(times_ms)
    n = len(s)
    mean = sum(s) / n
    stdev = (sum((x - mean) ** 2 for x in s) / n) ** 0.5
    p50 = s[n // 2]
    stall_threshold = 2.0 * p50
    stall_count = sum(1 for t in s if t > stall_threshold)
    return dict(
        p50=p50,
        p95=s[min(int(0.95 * n), n - 1)],
        p99=s[min(int(0.99 * n), n - 1)],
        max=s[-1],
        stdev=stdev,
        stall_pct=100.0 * stall_count / n,
    )


def reduce_scalar(accelerator, x):
    t = torch.tensor(float(x), device=accelerator.device, dtype=torch.float32)
    return accelerator.reduce(t, reduction='mean').item()


def reduce_list(accelerator, xs):
    t = torch.tensor(
        [float(v) for v in xs], device=accelerator.device, dtype=torch.float32,
    )
    return accelerator.reduce(t, reduction='mean').tolist()


def reduce_percentiles(accelerator, pct):
    keys = ['p50', 'p95', 'p99', 'max', 'stdev', 'stall_pct']
    t = torch.tensor(
        [pct[k] for k in keys], device=accelerator.device, dtype=torch.float32,
    )
    t = accelerator.reduce(t, reduction='mean')
    return {k: t[i].item() for i, k in enumerate(keys)}


def dynamo_graph_count():
    """Return the current dynamo-compiled graph count, 0 if torch._dynamo
    isn't loaded. Used to detect unexpected recompilation inside a cell.
    """
    try:
        import torch._dynamo as dynamo  # noqa: F401
        counters = dynamo.utils.counters
        return int(counters.get('stats', {}).get('unique_graphs', 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_cell(size_cfg, cell_cfg, norm, accelerator) -> dict[str, Any]:
    """Measure one (size, cell) combination. Returns a result dict.

    If cell_cfg['gpu_cached']=True the dataloader is replaced by replaying
    one cached batch from GPU memory — that cell's B5 throughput is the
    pure compute ceiling under the cell's compute settings.
    """
    is_main = accelerator.is_main_process
    size_label, d_model, n_layers, n_head, prod_bs = size_cfg

    use_amp = cell_cfg['amp']
    compile_mode = cell_cfg['compile_mode']
    fused = cell_cfg['fused']
    workers = cell_cfg['workers']
    prefetch = cell_cfg['prefetch']
    bs = cell_cfg['bs_override'] if cell_cfg['bs_override'] is not None else prod_bs
    gpu_cached = cell_cfg['gpu_cached']

    # --- Model ---
    model = SimCLRSmrt(
        d_model=d_model, n_layers=n_layers, n_head=n_head,
        max_len=CONTEXT, projection_dim=128, projection_layers=2,
    )
    if compile_mode is not None:
        model = torch.compile(model, mode=compile_mode)

    model = model.to(accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4, fused=fused,
    )
    criterion = NTXent(temperature=0.1)

    paired_ds = build_paired_dataset(cell_cfg, norm)

    # Pre-bind so Pyright sees these names in both branches; only populated
    # on the relevant path, only read by the matching closure.
    cached_v1 = None
    cached_v2 = None
    it_b5 = None
    dl = None
    A_its: float = 0.0
    A_sub = [0.0] * N_SUB_WINDOWS
    A_pct = percentiles([])

    if gpu_cached:
        # One-off DL just to grab a real batch; drop it immediately.
        tmp_dl = build_dl(paired_ds, bs, num_workers=4, prefetch_factor=4)
        model, optimizer, tmp_dl = accelerator.prepare(model, optimizer, tmp_dl)
        model.train()
        # Pull a few batches so we don't cache the cold-start one.
        tmp_it = iter(tmp_dl)
        for _ in range(4):
            cached_v1, cached_v2 = next(tmp_it)
        del tmp_it, tmp_dl
        gc.collect()
    else:
        dl = build_dl(paired_ds, bs, workers, prefetch)
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        model.train()

    torch.cuda.reset_peak_memory_stats(accelerator.device)

    warmup = N_WARMUP + (N_COMPILE_EXTRA if compile_mode is not None else 0)
    graphs_before = dynamo_graph_count() if compile_mode is not None else 0

    # --- Phase A: pure next(it) — skipped for GPU-cached cells ---
    if not gpu_cached:
        assert dl is not None
        it_a = iter(dl)

        def _a():
            next(it_a)

        if is_main:
            print("  phase A (pure next(it))", flush=True)
        A_its_raw, A_sub_raw, A_steps_raw, _ = timed_sustained(
            _a, MEASURE_SECS, N_SUB_WINDOWS, accelerator,
            n_warmup=N_WARMUP, nvml_sampler=None,
        )
        del it_a
        gc.collect()

        # Rebuild DL for B5 so Phase A's iterator doesn't pollute state.
        dl_b5 = build_dl(paired_ds, bs, workers, prefetch)
        dl_b5 = accelerator.prepare(dl_b5)
        it_b5 = iter(dl_b5)

        A_pct_raw = percentiles(A_steps_raw)
        A_its = reduce_scalar(accelerator, A_its_raw)
        A_sub = reduce_list(accelerator, A_sub_raw)
        A_pct = reduce_percentiles(accelerator, A_pct_raw)

    # --- Phase B5: real training step (or cached replay) ---
    if gpu_cached:
        if use_amp:
            def _b5():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    z1, z2 = model(cached_v1, cached_v2)
                    loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()
        else:
            def _b5():
                z1, z2 = model(cached_v1, cached_v2)
                loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()
    else:
        assert it_b5 is not None
        real_it = it_b5
        if use_amp:
            def _b5():
                u1, u2 = next(real_it)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    z1, z2 = model(u1, u2)
                    loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()
        else:
            def _b5():
                u1, u2 = next(real_it)
                z1, z2 = model(u1, u2)
                loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()

    if is_main:
        phase_label = "phase B5 (GPU-cached replay)" if gpu_cached else "phase B5 (real training step)"
        print(f"  {phase_label}", flush=True)

    nvml = NvmlSampler(device_idx=0) if is_main else None
    B5_its_raw, B5_sub_raw, B5_steps_raw, gpu_busy_pct = timed_sustained(
        _b5, MEASURE_SECS, N_SUB_WINDOWS, accelerator,
        n_warmup=warmup, nvml_sampler=nvml,
    )

    peak_mem_mb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 2)

    B5_pct_raw = percentiles(B5_steps_raw)
    B5_its = reduce_scalar(accelerator, B5_its_raw)
    B5_sub = reduce_list(accelerator, B5_sub_raw)
    B5_pct = reduce_percentiles(accelerator, B5_pct_raw)
    peak_mem = reduce_scalar(accelerator, peak_mem_mb)

    graphs_after = dynamo_graph_count() if compile_mode is not None else 0
    graphs_delta = graphs_after - graphs_before

    del model, optimizer, criterion
    if dl is not None:
        del dl
    if it_b5 is not None:
        del it_b5
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        size=size_label, label=cell_cfg['label'],
        bs=bs, workers=workers, prefetch=prefetch,
        amp=use_amp, compile_mode=compile_mode, fused=fused,
        subcrop=cell_cfg['subcrop'], norm_location=cell_cfg['norm_location'],
        gpu_cached=gpu_cached,
        A_its=A_its, B5_its=B5_its,
        A_sub=A_sub, B5_sub=B5_sub,
        A_pct=A_pct, B5_pct=B5_pct,
        peak_mem_mb=peak_mem,
        gpu_busy_pct=gpu_busy_pct,
        dynamo_graphs_delta=graphs_delta,
    )


# ---------------------------------------------------------------------------
# Summary & config-snippet printing (rank 0 only)
# ---------------------------------------------------------------------------

def print_summary_A(rows):
    """Per-size: today / post-fix / ceiling."""
    by_size = {}
    for r in rows:
        if r['label'].startswith(('C0 ', 'C_ALL ', 'C_CEIL ')):
            by_size.setdefault(r['size'], {})[r['label'].split()[0]] = r

    print("\n=== Per-size summary (Sweep A) ===", flush=True)
    hdr = (
        f"{'Size':<10}  "
        f"{'Today it/s':>11}  {'Post-fix it/s':>14}  {'Ceiling it/s':>13}  "
        f"{'Gain':>7}  {'GPU busy%':>10}  {'stall%':>7}"
    )
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)
    for size_label, _, _, _, _ in SIZES:
        d = by_size.get(size_label, {})
        c0 = d.get('C0')
        ca = d.get('C_ALL')
        cc = d.get('C_CEIL')
        if not c0 or not ca or not cc:
            continue
        gain = ca['B5_its'] / c0['B5_its'] if c0['B5_its'] > 0 else 0
        print(
            f"{size_label:<10}  "
            f"{c0['B5_its']:>11.2f}  {ca['B5_its']:>14.2f}  {cc['B5_its']:>13.2f}  "
            f"{gain:>6.1f}x  {ca['gpu_busy_pct']:>9.1f}%  "
            f"{ca['B5_pct']['stall_pct']:>6.1f}%",
            flush=True,
        )


def print_attribution_ladder(rows, size_label, sweep_name):
    """d128/d768 single-knob attribution from the C0 baseline."""
    # Pull C0 baseline from Sweep A and the cells from this sweep.
    c0 = None
    for r in rows:
        if r['size'] == size_label and r['label'].startswith('C0 '):
            c0 = r
            break

    ladder = []
    if c0 is not None:
        ladder.append(c0)
    for r in rows:
        if r['size'] != size_label:
            continue
        if not r['label'].startswith(('C1 ', 'C2 ', 'C3 ', 'C4 ', 'C5 ', 'C6 ', 'C7 ')):
            continue
        ladder.append(r)
    # Append C_CEIL last for reference
    for r in rows:
        if r['size'] == size_label and r['label'].startswith('C_CEIL'):
            ladder.append(r)
            break

    if len(ladder) < 2:
        return

    print(f"\n=== Attribution ladder @ {size_label} (Sweep {sweep_name}) ===", flush=True)
    hdr = (
        f"{'Cell':<38}  {'bs':>5}  {'B5 it/s':>8}  {'Cum gain':>9}  "
        f"{'Δ prev':>7}  {'stall%':>7}  {'GPU busy%':>10}"
    )
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)
    base_its = ladder[0]['B5_its']
    prev_its = base_its
    for r in ladder:
        cum = r['B5_its'] / base_its if base_its > 0 else 0
        delta = r['B5_its'] / prev_its if prev_its > 0 else 0
        print(
            f"{r['label']:<38}  {r['bs']:>5}  {r['B5_its']:>8.2f}  "
            f"{cum:>8.1f}x  {delta:>6.2f}x  "
            f"{r['B5_pct']['stall_pct']:>6.1f}%  "
            f"{r['gpu_busy_pct']:>9.1f}%",
            flush=True,
        )
        prev_its = r['B5_its']


def print_compile_guard(rows):
    """Sweep D output: does compile help or regress at d256/d512?"""
    print("\n=== Compile regression guard (Sweep D) ===", flush=True)
    hdr = (
        f"{'Size':<10}  {'C_ALL it/s':>11}  {'No-compile it/s':>16}  "
        f"{'Verdict':<30}  {'Graph recompiles':>17}"
    )
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)
    for size_label in ('d256_L8', 'd512_L8'):
        c_all = None
        c_nc = None
        for r in rows:
            if r['size'] != size_label:
                continue
            if r['label'].startswith('C_ALL ') and not r['gpu_cached']:
                c_all = r
            if 'NOCOMPILE' in r['label']:
                c_nc = r
        if c_all is None or c_nc is None:
            print(f"{size_label:<10}  (missing cells)", flush=True)
            continue
        ratio = c_all['B5_its'] / c_nc['B5_its'] if c_nc['B5_its'] > 0 else 0
        if ratio >= 1.05:
            verdict = f"HELPS +{ratio:.2f}x"
        elif ratio >= 0.95:
            verdict = f"NEUTRAL ({ratio:.2f}x)"
        else:
            verdict = f"REGRESSES ({ratio:.2f}x)"
        print(
            f"{size_label:<10}  {c_all['B5_its']:>11.2f}  {c_nc['B5_its']:>16.2f}  "
            f"{verdict:<30}  {c_all['dynamo_graphs_delta']:>17d}",
            flush=True,
        )


def print_config_snippets(rows, n_processes):
    """Per-size: copy-paste snippet for the best shippable config.

    Shippable = C_ALL at production bs, NOT bs-override cells (those
    require temperature retuning).
    """
    print("\n=== Recommended production config (per size) ===", flush=True)
    print("The snippets below are the C_ALL config at each size's production bs.", flush=True)
    print("Adoption requires the follow-up changes listed under 'Caveats' below.\n", flush=True)

    for size_label, _d_model, _n_layers, _n_head, prod_bs in SIZES:
        c0 = None
        c_all = None
        for r in rows:
            if r['size'] != size_label:
                continue
            if r['label'].startswith('C0 '):
                c0 = r
            if r['label'].startswith('C_ALL ') and not r['gpu_cached']:
                c_all = r
        if c0 is None or c_all is None:
            continue
        n_samples = 0
        try:
            n_samples = len(get_inner_dataset(subcrop=False))
        except Exception:
            pass
        today_epoch_h = (n_samples / (c0['B5_its'] * c0['bs'] * n_processes) / 3600.0) if c0['B5_its'] > 0 and n_samples else 0
        ideal_epoch_h = (n_samples / (c_all['B5_its'] * c_all['bs'] * n_processes) / 3600.0) if c_all['B5_its'] > 0 and n_samples else 0
        print(f"# --- {size_label} -------------------------------------------------", flush=True)
        print(f"# Today:    {c0['B5_its']:>6.2f} it/s  → ~{today_epoch_h:>5.2f} h/epoch", flush=True)
        print(f"# Post-fix: {c_all['B5_its']:>6.2f} it/s  → ~{ideal_epoch_h:>5.2f} h/epoch  "
              f"({c_all['B5_its']/max(c0['B5_its'],1e-9):.1f}× over today)", flush=True)
        print(f"# In config.yaml/simclr:", flush=True)
        print(f"#   batch_size: {prod_bs}", flush=True)
        print(f"# In _shared_train.py (or derived file):", flush=True)
        print(f"#   DataLoader: num_workers=8, prefetch_factor=16, pin_memory_device='cuda'", flush=True)
        print(f"#   Dataset:    SubcropShardedMemmapDataset(read_len={READ_LEN})   "
              f"# promote into smrt_foundation.dataset first", flush=True)
        print(f"#   Norm:       applied post-subcrop (see PostNormPairedView pattern); "
              f"re-fit KineticsNorm on subcrops", flush=True)
        print(f"#   model:      torch.compile(model, mode='reduce-overhead')", flush=True)
        print(f"#   optimizer:  torch.optim.AdamW(..., fused=True)", flush=True)
        print(f"#   zero_grad:  optimizer.zero_grad(set_to_none=True)", flush=True)
        print("", flush=True)


def print_caveats():
    print("=== Caveats for production adoption ===", flush=True)
    print("1. Post-subcrop norm requires re-fitting KineticsNorm on SubcropShardedMemmapDataset.", flush=True)
    print("   Rebuild norm_stats.pt on the new dataset before resume-compatible training.", flush=True)
    print("2. bs>512 changes SimCLR NTXent semantics (Chen 2020 §B.5). Retune temperature.", flush=True)
    print("3. compile(reduce-overhead) may diverge first post-resume steps vs a clean start.", flush=True)
    print("   Add compile_mode to run_metadata.yaml sidecar or accept first-epoch drift.", flush=True)
    print("4. SubcropShardedMemmapDataset with read_len=64 assumes CONTEXT ≤ 64.", flush=True)
    print("   Raise read_len if any future config increases ssl context.", flush=True)
    print("5. TORCHDYNAMO_NUMWORKERS=1 in the job env to avoid dynamo/DL worker CPU contention.", flush=True)


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
        print("=== ssl_53 GPU-saturation profile (4 sizes, 8×H100) ===", flush=True)
        print(
            f"World size: {accelerator.num_processes}  "
            f"device: {accelerator.device}  "
            f"CONTEXT={CONTEXT}  "
            f"IO subcrop read_len={READ_LEN}  chunk_size={CHUNK_SIZE}",
            flush=True,
        )
        print(
            f"Per cell: {N_WARMUP}(+{N_COMPILE_EXTRA} compile) warmup + "
            f"{PRE_TIME_STEPS} pre-time + {MEASURE_SECS}s measurement "
            f"({N_SUB_WINDOWS} sub-windows). DDP step counts fixed across ranks.",
            flush=True,
        )
        print(f"NVML sampling: {'on' if _NVML_AVAILABLE else 'OFF (pynvml unavailable)'}\n", flush=True)

    # Norm stats — computed once from the full-read dataset. Cells that
    # apply norm post-subcrop reuse these stats (diagnostic; see caveats).
    t0 = time.perf_counter()
    norm_ds = get_inner_dataset(subcrop=False)
    if is_main:
        print(f"SSL dataset: {len(norm_ds):,} reads from {MEMMAP_PATH}", flush=True)
    norm = KineticsNorm(norm_ds, max_samples=16_384)
    if is_main:
        print(
            f"Norm stats (means={norm.means}, stds={norm.stds}) "
            f"computed in {time.perf_counter()-t0:.1f}s\n",
            flush=True,
        )

    # Warm the subcrop dataset cache so Sweep A's first subcrop cell
    # doesn't eat the init cost inside a measurement.
    _ = get_inner_dataset(subcrop=True)

    sweeps = build_sweeps()
    rows = []

    for sweep_name, sizes_list, cells_list in sweeps:
        if is_main:
            print(f"\n########## Sweep {sweep_name}  "
                  f"({len(sizes_list)} sizes × {len(cells_list)} cells) ##########",
                  flush=True)
        for size_label in sizes_list:
            if size_label not in SIZE_BY_LABEL:
                if is_main:
                    print(f"  [skip] unknown size {size_label}", flush=True)
                continue
            size_cfg = SIZE_BY_LABEL[size_label]
            for cell_cfg in cells_list:
                if is_main:
                    parts = []
                    parts.append("bf16" if cell_cfg['amp'] else "fp32")
                    if cell_cfg['compile_mode']:
                        parts.append(f"compile({cell_cfg['compile_mode']})")
                    if cell_cfg['fused']:
                        parts.append("fused-AdamW")
                    parts.append(f"w={cell_cfg['workers']}")
                    parts.append(f"pf={cell_cfg['prefetch']}")
                    parts.append(f"subcrop={cell_cfg['subcrop']}")
                    parts.append(f"norm={cell_cfg['norm_location']}")
                    if cell_cfg['bs_override'] is not None:
                        parts.append(f"bs={cell_cfg['bs_override']}")
                    if cell_cfg['gpu_cached']:
                        parts.append("gpu-cached")
                    print(f"--- {size_label} | {cell_cfg['label']} "
                          f"[{', '.join(parts)}] ---", flush=True)
                try:
                    r = run_cell(size_cfg, cell_cfg, norm, accelerator)
                    rows.append(r)
                    if is_main:
                        sps = r['B5_its'] * r['bs'] * accelerator.num_processes
                        ap = r['A_pct']
                        bp = r['B5_pct']
                        if not r['gpu_cached']:
                            print(
                                f"  A  it/s: {r['A_its']:6.2f}   "
                                f"sub: {', '.join(f'{v:5.1f}' for v in r['A_sub'])}   "
                                f"p50={ap['p50']:.1f}ms p99={ap['p99']:.1f}ms "
                                f"stall={ap['stall_pct']:.1f}%",
                                flush=True,
                            )
                        print(
                            f"  B5 it/s: {r['B5_its']:6.2f}   "
                            f"sub: {', '.join(f'{v:5.1f}' for v in r['B5_sub'])}   "
                            f"p50={bp['p50']:.1f}ms p99={bp['p99']:.1f}ms "
                            f"stall={bp['stall_pct']:.1f}%",
                            flush=True,
                        )
                        print(
                            f"  global samples/s: {sps:,.0f}   "
                            f"GPU busy% (rank0): {r['gpu_busy_pct']:.1f}   "
                            f"peak mem: {r['peak_mem_mb']:,.0f} MB   "
                            f"dynamo recompiles: {r['dynamo_graphs_delta']}\n",
                            flush=True,
                        )
                except Exception as e:
                    if is_main:
                        print(f"  [ERROR] {type(e).__name__}: {e}\n", flush=True)
                    accelerator.free_memory()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if not is_main or not rows:
        return

    print_summary_A(rows)
    print_attribution_ladder(rows, 'd128_L4', 'B')
    print_attribution_ladder(rows, 'd768_L8', 'C')
    print_compile_guard(rows)
    print_config_snippets(rows, accelerator.num_processes)
    print_caveats()


if __name__ == '__main__':
    main()
