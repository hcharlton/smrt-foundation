# supervised_51_baseline_size_grid

Sweeps `DirectClassifier` parameter count under the supervised_40_baseline_v2
recipe, on a 24h walltime budget per size, full v2 dataset (`ds_limit=0`).

## Why

`supervised_40_baseline_v2` finished at eval_top1 = 0.8179 with the curve
still climbing through epoch 20 (0.808 -> 0.818 over the run, no plateau
visible in TensorBoard at run end). Two leverage axes:

1. **Capacity**: v40 is the smallest configuration in the project's
   model-size grid (~2M params). The supervised side has never been
   swept at the upper end on the v2 dataset with the LR-scheduler fix
   in place. If the asymptote is capacity-bound, larger models should
   beat 0.82 without any other change.
2. **Schedule horizon**: 20 epochs (~600k steps) at d=128 didn't
   plateau. 24h walltime gives ~50 epochs at d=128 and progressively
   fewer at larger sizes — both expansions in one run.

## The grid

Mirrors ssl_55/57's axes (`head_dim=64`):

| Size      | d_model | n_layers | n_head | params (approx) |
|-----------|---------|----------|--------|-----------------|
| d128_L4   | 128     | 4        | 2      | 2.0 M           |
| d256_L8   | 256     | 8        | 4      | 11.4 M          |
| d512_L8   | 512     | 8        | 8      | 45.5 M          |
| d768_L8   | 768     | 8        | 12     | 102.2 M         |

Param counts validated against ssl_55's measured numbers — same encoder
arch, same downsample geometry, the classifier head adds <0.3M which
doesn't shift the bucket.

### What's preserved from v40

- DirectClassifier at ctx=32, AdamW lr=3e-3 wd=0.02, cosine schedule
  with pct_start=0.1, KineticsNorm log_transform=True, balanced
  cpg_pos/neg_v2 train data, seed=42.
- LR-scheduler fix: created on the prepared optimizer and stepped
  manually, **not** wrapped by `accelerator.prepare(scheduler)`. The
  AcceleratedScheduler wrap advances the wrapped LambdaLR by
  `num_processes` per call, which compresses the cosine horizon by 8x
  on an 8-GPU run. v40 was the supervised fix; ssl_57's v3 patched the
  same bug for SSL.
- bs=512 per GPU x 8 GPUs = 4096 effective. ctx=32 means even d=768 L=8
  fits at bs=512 on H100 — no per-size batch-size scaling needed.
- Per-epoch checkpoint: `epoch_NN.pt` containing `model_state_dict`,
  `encoder_state_dict`, `config`, `epoch`, `metrics`, and norm stats.
  Loadable via the v40 recipe; the `model_state_dict` + `config` + norm
  stats are the inference artifact.
- Per-step train_loss / learning_rate / epoch logged to TensorBoard at
  every step; per-epoch eval_top1 / eval_f1 / eval_auroc / eval_auprc
  logged at the epoch boundary.

### What's new vs v40

- **Per-epoch CSV** at `<size_dir>/results.csv`. One row per completed
  epoch, written by main process. Schema:
  `epoch, global_step, walltime_s, train_loss_avg, lr_at_epoch_end, eval_top1, eval_f1, eval_auroc, eval_auprc, d_model, n_layers, n_head, batch_size_per_gpu, effective_bs, params_count`
- **`architecture/param_count`** logged to TensorBoard at step 0
  alongside the existing `cnn_receptive_field` scalar.
- **n_head=2 at d=128** instead of v40's n_head=4, to keep head_dim=64
  constant across the grid. The d=128 point in this experiment is
  therefore *not* a strict v40 reproduction — it is the smallest size
  in a head_dim-matched scaling grid.

## Schedule and walltime budget

`epochs=60` is the cosine-horizon target across all sizes. Walltime is
24h per size, which means cosine bottoms cleanly only at the smallest
size. v40's measured throughput at d=128 ctx=32 bs=512 on 8 GPUs was
~18.6 it/s, which projects to:

| Size      | proj. it/s | 24h steps | ~epochs |
|-----------|-----------:|----------:|--------:|
| d128_L4   | 18         | 1.6 M     | ~53     |
| d256_L8   | 12-15      | 1.0-1.3 M | ~30     |
| d512_L8   | 6-10       | 500-800 k | ~17     |
| d768_L8   | 4-6        | 350-500 k | ~12     |

Cosine LR doesn't reach the floor for d=512/d=768 — that is intentional
per the user direction "as many epochs as fits." This means the larger
sizes should be read against d=128 at a *matched epoch budget* (e.g.,
compare all four at epoch ~12), not at end-of-walltime. Plot eval_top1
vs epoch for clean apples-to-apples; eval_top1 vs walltime for total
training cost; eval_top1 vs params at fixed epochs for the scaling
slope.

## Resources (per-size)

```yaml
resources:
  cores: 16
  memory: 256gb
  walltime: "24:00:00"
  gres: gpu:8
  num_processes: 8
```

8 GPUs x 4 sizes = 32 GPUs (the available budget). Same DDP shape
across the grid -> fair effective-batch read across sizes.

## Submission

```bash
bash run.sh scripts/experiments/supervised_51_baseline_size_grid/size_d128_L4
bash run.sh scripts/experiments/supervised_51_baseline_size_grid/size_d256_L8
bash run.sh scripts/experiments/supervised_51_baseline_size_grid/size_d512_L8
bash run.sh scripts/experiments/supervised_51_baseline_size_grid/size_d768_L8
```

All four run concurrently on the user's 32-GPU budget.

## Layout

- `_shared_train.py` — single training loop shared by all four sizes.
- `size_<d>_L<L>/config.yaml` — per-size knobs (`d_model`, `n_layers`,
  `n_head`); everything else (resources, data paths, ctx, batch size,
  optimizer, schedule, epochs) identical across the four.
- `size_<d>_L<L>/train.py` — 12-line thin wrapper (verbatim
  ssl_55/56/57 pattern).

## Pass criterion

At least one size beats v40's eval_top1 = 0.8179 by >= 0.5pp at any
checkpoint. Failure to do so falsifies the capacity hypothesis at this
recipe, in which case the next move is data-side (more diverse train
data, augmentation) or schedule-side (lower LR for larger models, more
warmup), not capacity-side.

## Inference artifact

Each `size_*/checkpoints/epoch_NN.pt` is a self-contained inference
artifact. Load recipe:

```python
import torch
from smrt_foundation.model import DirectClassifier
from smrt_foundation.normalization import KineticsNorm

ckpt = torch.load('size_d256_L8/checkpoints/epoch_15.pt', map_location='cpu')
cfg = ckpt['config']['classifier']
model = DirectClassifier(d_model=cfg['d_model'], n_layers=cfg['n_layers'],
                          n_head=cfg['n_head'], max_len=cfg['context'])
model.load_state_dict(ckpt['model_state_dict'])
norm = KineticsNorm.load_stats({'means': ckpt['means'], 'stds': ckpt['stds']})
# model + norm are now ready for inference at any batch size.
```
