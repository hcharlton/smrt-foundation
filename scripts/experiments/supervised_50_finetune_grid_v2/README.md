# supervised_50_finetune_grid_v2

Matrixed fine-tune evaluation: **4 architectures x 3 encoder inits x 3 train_sizes = 33 runs**, all driven from one config via `scripts/ds_grid_v2.py` (the v2 successor to `ds_grid.py`).

## What this experiment answers

Three questions on one schedule, with architecture-matched baselines so the comparisons are clean:

1. **Does masked-prediction pretraining (ssl_57) beat supervised-from-scratch?** Compare `ssl_57_300k` to `random` at each architecture.
2. **Does masked-prediction beat contrastive (ssl_55)?** Compare `ssl_57_300k` to `ssl_55_300k` at d=128, d=256, d=512.
3. **How does the SSL-vs-baseline gap scale with model size and labeled data?** Three SimCLR-canonical train_sizes (low/mid/full = 10k / 100k / 8M) cover the data axis; four architectures (d=128 to d=768) cover the capacity axis.

## The matrix

|              | random | ssl_57_300k | ssl_55_300k |
|--------------|:------:|:-----------:|:-----------:|
| d128_L4      |  yes   |     yes     |     yes     |
| d256_L8      |  yes   |     yes     |     yes     |
| d512_L8      |  yes   |     yes     |     yes     |
| d768_L8      |  yes   |     yes     | **skipped** |

ssl_55 d=768 is skipped because the pre-LN-head SimCLR projection had a magnitude-runaway failure at that capacity; no usable checkpoint exists. Full post-mortem at `docs/negative_results.md` 2026-04-27.

Each (arch, init) combo runs at three train_sizes: `[10000, 100000, 8000000]`. Total 11 combos x 3 sizes = 33 runs.

## Schedule (single-stage, mirrors supervised_46/47/48)

- `AdamW` `max_lr=3e-3`, `weight_decay=0.02`, `pct_start=0.1`
- `batch_size=4096` cap, no per-size scaling
- `max_epochs=200`, `min_steps=100`, `max_steps=400000`
- 40 log-spaced evaluations per run, first at step 100
- Validation set capped at 1M samples (balanced)

`min_steps: 100` is the deliberate departure from `supervised_47`'s `min_steps: 10000`. At n=10k, bs=4096, max_epochs=200 gives a natural budget of ~600 steps; `min_steps: 10000` would force ~3300 epochs of training over the same 10k samples, which is the silent-overtraining failure mode `supervised_49` traced. Letting `max_epochs` govern at small n is correct here.

## Pretrained encoder sources

- `ssl_57_300k`: 300k-step input-masked-prediction encoders from `scripts/experiments/ssl_57_inputmask_grid_lnhead/size_{arch}/checkpoints/final_model.pt`. Probe accuracies at 300k: d=128 0.578, d=256 0.631, d=512 0.636, d=768 0.644.
- `ssl_55_300k`: 300k-step SimCLR encoders with LayerNorm projection head from `scripts/experiments/ssl_55_simclr_grid_lnhead/size_{arch}/checkpoints/final_model.pt`. Probe accuracies oscillate; reading at the latest checkpoint: d=128 ~0.616, d=256 [pending if available], d=512 ~0.661.
- The `ds_grid_v2` checkpoint loader (inherited from v1's `load_pretrained_encoder`) filters `pe.pe` shape mismatches automatically, which is load-bearing for the SSL-ctx=4096 -> classifier-ctx=32 transfer.

## Output structure

Nested by init first so debugging tends to inspect a single init's runs together:

```
supervised_50_finetune_grid_v2/
  config.yaml
  train.py
  README.md
  results.csv                              <-- merged top-level CSV
  training_logs/                           <-- TB root
    random/d128_L4/n10000/...
    ssl_57_300k/d256_L8/n100000/...
    ssl_55_300k/d512_L8/n8000000/...
  random/
    d128_L4/n10000/results.csv  step{N}.pt ...
    d256_L8/n100000/...
    ...
  ssl_57_300k/
    ...
  ssl_55_300k/
    ...
```

The merged `results.csv` carries `arch_name` and `init_name` as the leftmost columns, so the three-way comparison plots directly:

```python
import pandas as pd
df = pd.read_csv('results.csv')
final = df.groupby(['arch_name', 'init_name', 'train_size']).last()
# Color by init_name, facet by arch_name, x=train_size, y=val_accuracy
```

## Reading the curves

Three things to look for once the run lands:

1. **Random-init scaling vs init-supplied scaling at fixed arch.** The gap between `random` and `ssl_57_300k` (or `ssl_55_300k`) at the same `(arch, train_size)` is the SSL value. The gap should be largest at small n and shrink to ~0 at full data (saturation).
2. **ssl_57 vs ssl_55 head-to-head at d=128, d=256, d=512.** With matched schedules and matched random-init baselines, this disambiguates "does the SSL objective matter for transfer" vs "does the encoder capacity matter."
3. **Cross-arch slope.** Within a fixed init, larger models should help more on harder tasks (smaller n). Look for the d=128 -> d=768 slope to widen at low n if the SSL representation is genuinely supplying useful prior structure.

## How to submit

```bash
bash run.sh scripts/experiments/supervised_50_finetune_grid_v2
```

Resources are written for one node (8 GPUs); `mp.spawn` distributes the 33 work items via FLOPS-weighted load balancing in `assign_combos`. With the user's 16-GPU pod budget, two such submissions in parallel halve walltime.

## Verification before submit

`scripts/ds_grid_v2.py` honors `DS_GRID_DRY_RUN=1`, which prints the expanded combo list, the per-GPU assignment preview, and the load-balance spread, then exits before any GPU dispatch. Run that locally to confirm the matrix is what you expect:

```bash
DS_GRID_DRY_RUN=1 .venv/bin/python -m scripts.ds_grid_v2 \
    scripts/experiments/supervised_50_finetune_grid_v2/config.yaml
```

`DS_GRID_FORCE_WORLD_SIZE=8` in combination simulates an 8-GPU assignment from a no-GPU machine, useful for inspecting the load-balance spread before submission.

## Note on load balance

The matrix has a structural imbalance: the d=768 combos at n=8M dominate per-GPU work (the FLOPS proxy puts each at ~72x the weight of d=128/n=8M), so on 8 GPUs the greedy assignment will leave the heaviest GPUs at ~7x the work of the lightest. Wallclock = max(per-GPU times); the lighter GPUs sit idle once their assigned combos finish. This is acceptable for a one-shot research experiment but represents wasted GPU time on pods.

Two ways to mitigate, in order of effort:

1. **Submit twice on 2 pods.** Run two configs that partition the matrix; e.g. one config with only `d128_L4 + d256_L8` (lighter, more combos) and another with only `d512_L8 + d768_L8` (heavier, fewer combos). Each pod handles its half. This is a config-only change (drop the unwanted architectures from each `architectures` block).
2. **Tune the FLOPS proxy.** `_combo_weight` in `scripts/ds_grid_v2.py` uses `d_model^2 * n_layers`. At ctx=32 most compute is fixed-cost CNN frontend, not transformer attention/FFN, so `d^2` likely overweights large archs vs reality. After a calibration run, adjust to `d^1.5` or use a hand-tuned dict.

## Out of scope

- ssl_56 fine-tuning (probe declined during pretraining; features are worse than random init for this downstream task).
- Two-stage gradual-unfreeze schedules. v2 is single-stage only; for two-stage use v1 (`scripts/ds_grid.py`).
- The 16-point train_sizes grid from `supervised_47`. Three points are sufficient for the SimCLR-style "low / mid / full" reading and avoid memorization at small n with d=512+.
