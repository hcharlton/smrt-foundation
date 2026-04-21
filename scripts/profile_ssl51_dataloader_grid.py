"""Grid-sweep DataLoader config for SSL-51 SimCLR to close the real-loop penalty.

Follow-up to scripts/profile_ssl51_compute_bound.py, which found:
    - compute is fine (A/B4 = 0.03-0.24, all 4 sizes "compute-bound" by A/B4)
    - NTXent all_gather is cheap (0.5-0.7 ms, flat across sizes)
    - DDP grad all_reduce is not saturating (bwd% flat at 67-72% across 48x params)
    - BUT real-loop penalty (B5-B4)/B4 is +31% (d768) to +311% (d128), with
      p95 multi-second tails on phase A — intermittent worker stalls drain the
      prefetch queue and leave the GPU idle.

This script grids DL config and measures phase A (pure IO) + phase B5 (real
training step) on two representative sizes:
    d128_L4 — most dataloader-sensitive (smallest compute, biggest penalty)
    d768_L8 — most compute-bound (largest compute, smallest penalty headroom)

Grid: (num_workers, prefetch_factor, persistent_workers). ~11 configs x 2 sizes.

Submission (via run.sh):
    bash run.sh scripts/experiments/profile_ssl51_dataloader_grid

Or direct sbatch (falls back to the raw form used by profile_ssl51_compute_bound).
"""
import gc
import os
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

from scripts.profile_ssl51_compute_bound import (
    timed, reduce_pair, make_policy,
    CONTEXT, MEMMAP_PATH, WARMUP, N_TIMED,
)


# Two representative sizes. Using the bracketed endpoints; middle sizes
# scale predictably given the first-run data.
SIZES = [
    dict(name='d128_L4', d_model=128, n_layers=4, n_head=2,  batch_size=512),
    dict(name='d768_L8', d_model=768, n_layers=8, n_head=12, batch_size=256),
]

# DataLoader grid. Each tuple is (num_workers, prefetch_factor, persistent_workers).
# Total in-flight batches per rank = num_workers * prefetch_factor. On a 64-core
# node with 8 ranks: num_workers=16 hits 128 total processes (2x oversub).
GRID = [
    # (nw, pf, persistent)
    ( 2, 4, True),   # under-provisioned baseline
    ( 4, 2, True),
    ( 4, 4, True),
    ( 4, 8, True),
    ( 8, 2, True),
    ( 8, 4, True),   # current config (real training default)
    ( 8, 8, True),
    (16, 2, True),
    (16, 4, True),
    (16, 8, True),   # over-provisioned ceiling
    ( 8, 4, False),  # is persistent_workers=True actually helping?
]


def profile_config(size_cfg, dl_cfg, ds, norm, policy, accelerator):
    """Build everything fresh for one (size, dl) cell and return (A, B5) timings.

    Model is rebuilt per cell so each measurement is independent (no leaked
    persistent_workers from a prior cell, no DDP state cross-contamination).
    """
    nw, pf, pw = dl_cfg
    paired_ds = PairedViewDataset(ds, policy=policy, norm_fn=norm)
    dl = DataLoader(
        paired_ds, batch_size=size_cfg['batch_size'], num_workers=nw,
        pin_memory=True, prefetch_factor=pf, shuffle=True,
        persistent_workers=pw,
    )
    model = SimCLRSmrt(
        d_model=size_cfg['d_model'], n_layers=size_cfg['n_layers'],
        n_head=size_cfg['n_head'], max_len=CONTEXT,
        projection_dim=128, projection_layers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = NTXent(temperature=0.1)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model.train()

    # Phase A: prepared DL only
    it = iter(dl)
    for _ in range(WARMUP):
        next(it)
    A = timed(lambda: next(it), N_TIMED)

    # Phase B5: real training step from the same DL
    def _b5():
        u1, u2 = next(it)
        z1, z2 = model(u1, u2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
    for _ in range(WARMUP):
        _b5()
    B5 = timed(_b5, N_TIMED)

    A_m  = reduce_pair(accelerator, A)
    B5_m = reduce_pair(accelerator, B5)

    # Full teardown — free_memory clears accelerator's internal lists which
    # otherwise keep persistent_workers alive across cells.
    del model, optimizer, dl, it, criterion, paired_ds
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return A_m, B5_m


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='no', kwargs_handlers=[ddp_kwargs])
    set_seed(42)
    is_main = accelerator.is_main_process

    if is_main:
        print("=== SSL-51 SimCLR DataLoader grid sweep ===", flush=True)
        print(
            f"World size: {accelerator.num_processes}  "
            f"device: {accelerator.device}  precision: fp32",
            flush=True,
        )
        print(
            f"Grid: {len(GRID)} DL configs x {len(SIZES)} sizes = "
            f"{len(GRID)*len(SIZES)} cells; {WARMUP} warmup + {N_TIMED} timed "
            f"per phase (A, B5)",
            flush=True,
        )

    # Build dataset + norm ONCE (shared across all cells).
    t0 = time.perf_counter()
    ds = ShardedMemmapDataset(MEMMAP_PATH, limit=0)
    if is_main:
        print(f"\nSSL dataset: {len(ds):,} reads from {MEMMAP_PATH}", flush=True)
    norm = KineticsNorm(ds, max_samples=16_384)
    policy = make_policy()
    if is_main:
        print(f"Norm stats ready in {time.perf_counter()-t0:.1f}s\n", flush=True)

    all_rows = []
    for size_cfg in SIZES:
        if is_main:
            print(
                f"\n--- size {size_cfg['name']} "
                f"(d={size_cfg['d_model']}, L={size_cfg['n_layers']}, "
                f"h={size_cfg['n_head']}, bs_rank={size_cfg['batch_size']}, "
                f"bs_global={size_cfg['batch_size']*accelerator.num_processes}) ---",
                flush=True,
            )
            print(
                f"{'nw':>3} {'pf':>3} {'persist':>8}  "
                f"{'A_p50':>8} {'A_p95':>10}  "
                f"{'B5_p50':>8} {'B5_p95':>10}  "
                f"{'penalty':>8}",
                flush=True,
            )

        # step_c from the compute-bound profile (reference line; compute does
        # not change with DL config, so penalty = B5_p50 - step_c).
        step_c_ref = 17.7 if size_cfg['name'] == 'd128_L4' else 80.2

        for dl_cfg in GRID:
            nw, pf, pw = dl_cfg
            try:
                A, B5 = profile_config(
                    size_cfg, dl_cfg, ds, norm, policy, accelerator,
                )
                penalty = B5[0] - step_c_ref
                if is_main:
                    print(
                        f"{nw:>3} {pf:>3} {str(pw):>8}  "
                        f"{A[0]:>8.2f} {A[1]:>10.2f}  "
                        f"{B5[0]:>8.2f} {B5[1]:>10.2f}  "
                        f"{penalty:>+7.1f}",
                        flush=True,
                    )
                all_rows.append(dict(
                    size=size_cfg['name'], nw=nw, pf=pf, pw=pw,
                    A_p50=A[0], A_p95=A[1],
                    B5_p50=B5[0], B5_p95=B5[1],
                    penalty=penalty,
                ))
            except Exception as e:
                if is_main:
                    print(
                        f"{nw:>3} {pf:>3} {str(pw):>8}  "
                        f"[ERROR] {type(e).__name__}: {e}",
                        flush=True,
                    )

    if is_main and all_rows:
        print("\n\n=== Best configs per size (sorted by B5_p50) ===", flush=True)
        for size_name in [s['name'] for s in SIZES]:
            rows = sorted(
                [r for r in all_rows if r['size'] == size_name],
                key=lambda r: r['B5_p50'],
            )
            print(f"\n{size_name}:", flush=True)
            print(
                f"{'rank':>4}  {'nw':>3} {'pf':>3} {'persist':>8}  "
                f"{'B5_p50':>8} {'B5_p95':>10}  {'A_p50':>8} {'A_p95':>10}  "
                f"{'penalty':>8}",
                flush=True,
            )
            for i, r in enumerate(rows):
                print(
                    f"{i+1:>4}  {r['nw']:>3} {r['pf']:>3} {str(r['pw']):>8}  "
                    f"{r['B5_p50']:>8.2f} {r['B5_p95']:>10.2f}  "
                    f"{r['A_p50']:>8.2f} {r['A_p95']:>10.2f}  "
                    f"{r['penalty']:>+7.1f}",
                    flush=True,
                )


if __name__ == '__main__':
    main()
