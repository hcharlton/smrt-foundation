"""Profile SSL-51 SimCLR step time on an 8-GPU pod.

For each of the four ssl_51_simclr_grid_r1 sizes, progressively adds one
training stage at a time and times it, so the deltas attribute wall time
to individual sources:

    A   I/O only                                   (dataloader + H2D)
    B1  forward (cached batch)                     (encoder + projection, no comms)
    B2  + NTXent loss                              (adds differentiable all_gather)
    B3  + backward                                 (adds DDP grad all_reduce + z_all reduce-scatter)
    B4  + optimizer                         = compute-only step
    B5  full step with real dataloader      = real training step

Deltas: B2-B1 = NTXent comms, B3-B2 = backward (compute + grad comms),
B4-B3 = optimizer, B5-B4 = dataloader overhead in the real loop.
Diagnosis: A/B4 < 0.7 → compute-bound; > 1.3 → dataloader-bound.

Submit on Gefion (8 GPUs, ~12 min expected, 30 min budget):

    sbatch --account=cu_0030 --cpus-per-task=64 --mem=448gb \\
           --time=00:30:00 --gres=gpu:8 --job-name=profile_ssl51 \\
           --output=scripts/profile_ssl51_compute_bound.%j.out \\
           --wrap="source .venv/bin/activate && cd $(pwd) && \\
                   accelerate launch --num_processes=8 --mixed_precision=no \\
                     scripts/profile_ssl51_compute_bound.py"

Local CPU dry-run (API-shape check; timings are meaningless):

    accelerate launch --num_processes=1 --cpu scripts/profile_ssl51_compute_bound.py
"""
import gc
import os
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from smrt_foundation.dataset import ShardedMemmapDataset, PairedViewDataset
from smrt_foundation.model import SimCLRSmrt
from smrt_foundation.loss import NTXent
from smrt_foundation.normalization import KineticsNorm
from smrt_foundation.augment import AugmentationPolicy


# Matches the four size_*/config.yaml tuples. Largest first so allocator
# fragmentation from a smaller size can't crowd out d768's allocation.
SIZES = [
    dict(name='d768_L8', d_model=768, n_layers=8, n_head=12, batch_size=256),
    dict(name='d512_L8', d_model=512, n_layers=8, n_head=8,  batch_size=512),
    dict(name='d256_L8', d_model=256, n_layers=8, n_head=4,  batch_size=512),
    dict(name='d128_L4', d_model=128, n_layers=4, n_head=2,  batch_size=512),
]

WARMUP = 5
N_TIMED = 20
CONTEXT = 32
MEMMAP_PATH = 'data/01_processed/ssl_sets/ob007_raw.memmap'


def make_policy():
    # Matches the `augment:` section of every size_*/config.yaml.
    return AugmentationPolicy(
        CONTEXT,
        rc_lookup=None,
        revcomp_p=0.0,
        channel_dropout_p=0.2,
        gaussian_noise_p=0.8, gaussian_noise_sigma=0.1,
        blur_p=0.5, blur_sigma_range=(0.2, 2.0),
    )


def build_dataloader(paired_ds, batch_size):
    # persistent_workers=False so workers shut down on `del it; gc.collect()`,
    # freeing CPU for the cached-batch GPU-only phases (B1-B4). In real
    # training persistent_workers=True avoids per-epoch restart cost; within
    # a single epoch the per-batch throughput is identical.
    return DataLoader(
        paired_ds, batch_size=batch_size, num_workers=8,
        pin_memory=True, prefetch_factor=4, shuffle=True,
        persistent_workers=False,
    )


def timed(fn, n):
    """Wall time of fn() over n iterations, CUDA-synced each side.

    Returns (p50_ms, p95_ms) over this rank's own observations.
    """
    cuda = torch.cuda.is_available()
    times = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(int(0.95 * len(times)), len(times) - 1)]
    return p50, p95


def reduce_pair(accelerator, pair):
    """Cross-rank mean of (p50, p95)."""
    t = torch.tensor(list(pair), device=accelerator.device, dtype=torch.float32)
    t = accelerator.reduce(t, reduction='mean')
    return t[0].item(), t[1].item()


def profile_size(cfg, ds, norm, policy, accelerator):
    is_main = accelerator.is_main_process
    paired_ds = PairedViewDataset(ds, policy=policy, norm_fn=norm)

    dl = build_dataloader(paired_ds, cfg['batch_size'])
    model = SimCLRSmrt(
        d_model=cfg['d_model'], n_layers=cfg['n_layers'], n_head=cfg['n_head'],
        max_len=CONTEXT, projection_dim=128, projection_layers=2,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4,
    )
    criterion = NTXent(temperature=0.1)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    rf = accelerator.unwrap_model(model).encoder.cnn.r0

    if is_main:
        print(
            f"\n--- size {cfg['name']} "
            f"(d={cfg['d_model']}, L={cfg['n_layers']}, h={cfg['n_head']}, "
            f"bs_rank={cfg['batch_size']}, "
            f"bs_global={cfg['batch_size'] * accelerator.num_processes}, "
            f"params={n_params/1e6:.1f}M, RF={rf}) ---",
            flush=True,
        )

    # --- Phase A: I/O only, through the prepared DL (matches real H2D path) ---
    it = iter(dl)
    for _ in range(WARMUP):
        next(it)
    A = timed(lambda: next(it), N_TIMED)

    # Cache one batch, then drop the iterator so workers shut down before
    # the cached-batch phases. gc.collect() forces _MultiProcessingDataLoaderIter
    # to run its __del__, which calls _shutdown_workers().
    v1, v2 = next(it)
    del it
    gc.collect()

    # --- B1: forward only, cached batch ---
    def _b1():
        out = model(v1, v2)
        del out
    for _ in range(WARMUP):
        _b1()
    B1 = timed(_b1, N_TIMED)

    # --- B2: forward + NTXent (differentiable all_gather on z1, z2) ---
    def _b2():
        z1, z2 = model(v1, v2)
        loss = criterion(z1, z2)
        del z1, z2, loss
    for _ in range(WARMUP):
        _b2()
    B2 = timed(_b2, N_TIMED)

    # --- B3: forward + loss + backward (DDP grad all_reduce) ---
    def _b3():
        z1, z2 = model(v1, v2)
        loss = criterion(z1, z2)
        accelerator.backward(loss)
        for p in model.parameters():
            p.grad = None
    for _ in range(WARMUP):
        _b3()
    B3 = timed(_b3, N_TIMED)

    # --- B4: full step on cached batch = compute-only training step ---
    def _b4():
        z1, z2 = model(v1, v2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
    # Extra warmup before B4: first optimizer.step() allocates AdamW state
    # (moment buffers) — don't let that land inside the timed window.
    for _ in range(WARMUP):
        _b4()
    B4 = timed(_b4, N_TIMED)

    # --- B5: full step with a fresh real dataloader ---
    dl2 = build_dataloader(paired_ds, cfg['batch_size'])
    dl2 = accelerator.prepare(dl2)
    it2 = iter(dl2)

    def _b5():
        u1, u2 = next(it2)
        z1, z2 = model(u1, u2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    for _ in range(WARMUP):
        _b5()
    B5 = timed(_b5, N_TIMED)

    A_m  = reduce_pair(accelerator, A)
    B1_m = reduce_pair(accelerator, B1)
    B2_m = reduce_pair(accelerator, B2)
    B3_m = reduce_pair(accelerator, B3)
    B4_m = reduce_pair(accelerator, B4)
    B5_m = reduce_pair(accelerator, B5)

    if is_main:
        step_c = B4_m[0]
        rows = [
            ('A  I/O only',                A_m,  None),
            ('B1 forward (cached)',        B1_m, B1_m[0]),
            ('B2 +NTXent (all_gather)',    B2_m, B2_m[0] - B1_m[0]),
            ('B3 +backward (all_reduce)',  B3_m, B3_m[0] - B2_m[0]),
            ('B4 +optimizer (step_c)',     B4_m, B4_m[0] - B3_m[0]),
            ('B5 step w/ real dl',         B5_m, B5_m[0] - B4_m[0]),
        ]
        print(
            f"{'phase':<28} {'p50 ms':>8} {'p95 ms':>8} "
            f"{'delta':>8} {'% of B4':>8}",
            flush=True,
        )
        for label, (p50, p95), delta in rows:
            delta_s = f"{delta:+.2f}" if delta is not None else "—"
            pct = f"{100*p50/step_c:>3.0f}%" if step_c > 0 else "—"
            print(
                f"{label:<28} {p50:>8.2f} {p95:>8.2f} "
                f"{delta_s:>8} {pct:>8}",
                flush=True,
            )

        ratio = A_m[0] / step_c if step_c > 0 else float('inf')
        if ratio < 0.7:
            bound = 'compute-bound'
        elif ratio > 1.3:
            bound = 'dataloader-bound'
        else:
            bound = 'overlap'
        dl_pct  = 100 * (B5_m[0] - B4_m[0]) / step_c
        ag_pct  = 100 * (B2_m[0] - B1_m[0]) / step_c
        bwd_pct = 100 * (B3_m[0] - B2_m[0]) / step_c
        fwd_pct = 100 *  B1_m[0]            / step_c
        print("\n  diagnosis:", flush=True)
        print(f"    A / B4                    = {ratio:.2f}  -> {bound}", flush=True)
        print(f"    (B5 - B4) / B4            = {dl_pct:+.1f}%  (real-loop dataloader penalty)", flush=True)
        print(f"    all_gather fraction       = {ag_pct:.1f}%  (NTXent loss comms)", flush=True)
        print(f"    bwd + grad-comm fraction  = {bwd_pct:.1f}%  (backward + grad all_reduce + z_all reduce-scatter)", flush=True)
        print(f"    forward-compute fraction  = {fwd_pct:.1f}%", flush=True)

    result = dict(
        name=cfg['name'], bs_r=cfg['batch_size'],
        bs_g=cfg['batch_size'] * accelerator.num_processes,
        A=A_m[0], B1=B1_m[0], B2=B2_m[0], B3=B3_m[0], B4=B4_m[0], B5=B5_m[0],
    )

    # Full cleanup before next size: free_memory() clears accelerator's
    # internal _models/_optimizers/_dataloaders lists that `del` does not.
    del model, optimizer, dl, dl2, it2, v1, v2, paired_ds, criterion
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='no',
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(42)
    is_main = accelerator.is_main_process

    if is_main:
        print("=== SSL-51 SimCLR step-time decomposition ===", flush=True)
        print(
            f"World size: {accelerator.num_processes}  "
            f"device: {accelerator.device}  "
            f"precision: fp32 (matches --mixed_precision=no in run.sh)",
            flush=True,
        )
        print(
            f"Per phase: {WARMUP} warmup + {N_TIMED} timed steps "
            f"(p50/p95 per rank; reported as cross-rank mean)",
            flush=True,
        )

    t0 = time.perf_counter()
    ds = ShardedMemmapDataset(MEMMAP_PATH, limit=0)
    if is_main:
        print(f"\nSSL dataset: {len(ds):,} reads from {MEMMAP_PATH}", flush=True)
    norm = KineticsNorm(ds, max_samples=16_384)
    if is_main:
        print(
            f"Norm stats (means={norm.means}, stds={norm.stds}) "
            f"computed in {time.perf_counter()-t0:.1f}s",
            flush=True,
        )
    policy = make_policy()

    results = []
    for cfg in SIZES:
        try:
            results.append(profile_size(cfg, ds, norm, policy, accelerator))
        except Exception as e:
            if is_main:
                print(
                    f"\n[ERROR] size {cfg['name']} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            accelerator.free_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if is_main and results:
        print("\n\n=== Summary (all times p50 ms, cross-rank mean) ===", flush=True)
        hdr = (
            f"{'size':<10} {'bs_r':>4} {'bs_g':>5} "
            f"{'IO':>7} {'fwd':>7} {'AG':>6} {'bwd':>6} {'opt':>6} "
            f"{'step_c':>7} {'step_r':>7} {'AG%':>5} {'bwd%':>5} {'bound':>18}"
        )
        print(hdr, flush=True)
        print('-' * len(hdr), flush=True)
        for r in results:
            ag  = r['B2'] - r['B1']
            bwd = r['B3'] - r['B2']
            opt = r['B4'] - r['B3']
            ag_pct  = 100 * ag  / r['B4'] if r['B4'] > 0 else 0.0
            bwd_pct = 100 * bwd / r['B4'] if r['B4'] > 0 else 0.0
            ratio = r['A'] / r['B4'] if r['B4'] > 0 else float('inf')
            bound = ('compute-bound' if ratio < 0.7
                     else 'dataloader-bound' if ratio > 1.3
                     else 'overlap')
            print(
                f"{r['name']:<10} {r['bs_r']:>4} {r['bs_g']:>5} "
                f"{r['A']:>7.1f} {r['B1']:>7.1f} {ag:>6.1f} {bwd:>6.1f} {opt:>6.1f} "
                f"{r['B4']:>7.1f} {r['B5']:>7.1f} "
                f"{ag_pct:>4.0f}% {bwd_pct:>4.0f}% {bound:>18}",
                flush=True,
            )


if __name__ == '__main__':
    main()
