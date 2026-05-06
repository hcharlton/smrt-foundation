# supervised_52_tissue

First supervised run on the new tissue-classification task: 8-way prediction of tissue identity from PacBio HiFi kinetics on the yoran individual.

## Why

A new task in this project. Goals for this initial pass:

1. Validate the new architecture end to end (`TissueClassifier`, `SmrtEncoderTissue`, modified `KineticsNorm`, `TissueMemmapDataset`). Surface any bugs at scale before committing GPU-time to a large run.
2. Demonstrate that the model can fit a small training subset to near-zero loss while validation loss develops a clean U-shape. If we cannot overfit, the architecture or pipeline is broken.
3. Establish a within-cell vs across-cell generalization gap by running two validation sets in parallel: `val_s1` (same cell as train) and `val_s2` (a cell the model never saw during training). The `val_s1` vs `val_s2` gap tells us how much of the model's accuracy is real tissue signal vs cell-batch signal.

This is a feasibility / sanity run, not a final model. Larger sweeps, augmentation, SSL-pretrained init, and scaled training subsets come after.

## Dataset

`data/01_processed/tissue_sets/yoran_ctx4096`. Built by `scripts/bam_to_labeled_memmap.py`, single individual (yoran), 8 solid tissues (`colon, kidney, liver, lung, muscle, skin, spleen, testis`; blood excluded as a single-cell confound). Built at `context=4096`, `max_reads_per_tissue=200000`, length filter drops reads < context. Padding is identically zero across the dataset by construction (verified by EDA).

EDA findings (`report/eda/tissue_dataset_overview/overview_4096.svg`) that shape the design here:

- Manifest: 1,555,968 windows. Per-tissue counts are within 0.5% of each other (~194k each). The dataset is balanced by sampling, not by raw availability; we do not need to balance further at the dataloader.
- Two cells: `m84108_251007_115244_s1` (753k rows) and `m84108_250930_153107_s2` (803k rows). Both contain all 8 tissues, so held-out-cell validation is feasible for every tissue.
- Read-length distributions are similar across tissues (medians 14.0 - 15.5 kb). Length-filter retention is uniform 96.7 - 97.7% across every (tissue, cell) bucket: no length-driven selection bias.

## Architecture

### New code paths added by this experiment

- `TissueClassifier(d_model, n_layers, n_head, max_len, n_classes=8, n_continuous=4)` (in `smrt_foundation/model.py`). Encoder + center-latent pool + `Linear(d_model, d_model//2) -> GELU -> Linear(d_model//2, n_classes)` head. Returns raw logits; loss is `nn.CrossEntropyLoss()` with long targets.
- `SmrtEncoderTissue(SmrtEncoder)` subclass. Replaces the parent's `SmrtEmbedding(n_continuous=2)` with `n_continuous=4` and overrides `get_latents` to slice channels as `seq=x[..., 0]`, `kin=x[..., 1:5]`, `pad=x[..., -1]`. The default `SmrtEncoder` and every model that uses it (`Smrt2Vec`, `DirectClassifier`, `SmrtAutoencoder*`) is unchanged.
- `KineticsNorm` is modified in place to accept `n_continuous=2|4` (default 2 preserves CpG/SSL behavior). When `n_continuous=4`, `log1p` and z-score apply to channels `[1, 2, 3, 4]` instead of `[1, 2]`. `save_stats` records `norm_n_continuous` so checkpoints round-trip correctly. `load_stats` defaults the field to 2 for backward compat with existing CpG/SSL artifacts.

### Architecture grid

Mirrors ssl_55/57/58 and supervised_51 with `head_dim = 64` constant:

| Size         | d_model | n_layers | n_head | head_dim | bs/gpu (start) | global bs |
|--------------|--------:|---------:|-------:|---------:|---------------:|----------:|
| size_d128_L4 |     128 |        4 |      2 |       64 |            128 |      1024 |
| size_d256_L8 |     256 |        8 |      4 |       64 |            128 |      1024 |
| size_d512_L8 |     512 |        8 |      8 |       64 |             64 |       512 |
| size_d768_L8 |     768 |        8 |     12 |       64 |             32 |       256 |

Per-GPU batch size scales for activation memory at `ctx=2048`. Effective batch size is **not** constant across the grid: this is an architecture-eval / debug run, not a scaling study, so we accept the cross-size comparison being approximate. After the first run reads memory headroom from `nvidia-smi`, batch sizes can be revised upward (especially for d=128 and d=256). Large sizes (d=512, d=768) may need further reductions if OOM appears.

## Data pipeline

- `TissueMemmapDataset(data_dir, filter_expr, norm_fn)` from `smrt_foundation/dataset.py`. Manifest-driven, polars `filter_expr` for splitting.
- Online normalization via `KineticsNorm(log_transform=True, n_continuous=4)`. Stats are computed once on a 16k-row sample of the training split at startup. The dataset stores raw uint8 and the normalizer is the dataloader's job. This matches the project convention.
- Center-crop from 4096 down to 2048 at load time. Deterministic: every row maps to the same 2048-bp window every epoch. **No** random-crop augmentation. This is intentional - random cropping fights the overfit demonstration.
- `ChunkedRandomSampler(chunk_size=2048)` for sequential-within-chunk shard reads. Avoids LRU thrash in `TissueMemmapDataset` when shuffling.

### Splits

Subset sized for the overfit demo: small enough that the smaller models can memorize within walltime. Three named splits, where the val split names encode the source cell so the partition is self-documenting:

- `train`: 50,000 rows from cell `m84108_251007_115244_s1` (suffix `s1`), stratified at 6,250 per tissue. **Only cell s1.**
- `val_s1`: 10,000 rows from cell `s1`, disjoint from `train`, stratified at 1,250 per tissue. Same cell as train, different reads. Tests within-distribution generalization at the read level.
- `val_s2`: 10,000 rows from cell `m84108_250930_153107_s2` (suffix `s2`), stratified at 1,250 per tissue. **A cell never seen in train.** Tests across-cell generalization. The `val_s1` vs `val_s2` gap quantifies the cell-batch effect.

The val split names (`val_s1`, `val_s2`) come from the trailing token of each cell ID (`cell_str.split('_')[-1]`). Reading the partition.csv tells you immediately which cell each row came from.

#### How the splits are realized

Splits are **register-based**, not on-disk partitions. The same `shard_NNNNN.npy` files on disk serve every split: no duplication, no second build, no rebuild required to change a split definition.

`TissueMemmapDataset(data_dir, filter_expr, norm_fn)` reads `manifest.parquet` once at construction time, applies the polars `filter_expr`, and keeps the surviving `(shard_idx, row_idx, tissue_id)` tuples in memory. At `__getitem__`, it memmaps the shard and reads the corresponding row directly. A "split" is the polars predicate plus the persisted partition file described below.

#### Verifiable determinism: persisted partition

The partition is keyed by `read_name` (the unique PacBio identifier `<cell>/<zmw>/ccs` that already lives in the manifest) and saved to disk as `<data_dir>/partition.csv`, alongside `manifest.parquet` and the shard files. The partition is a structural property of the dataset (which reads belong to which split), so it lives with the dataset rather than with any one experiment. Experiments that want a different partition either pass an alternate `--partition_path` or regenerate `partition.csv` with new arguments.

Schema:

| Column      | Type | Description                                                |
|-------------|------|------------------------------------------------------------|
| `read_name` | str  | Unique PacBio read identifier (`<cell>/<zmw>/ccs`)         |
| `split`     | str  | `train` (cell s1) / `val_s1` (cell s1) / `val_s2` (cell s2) |

Workflow:

The partition is produced by a standalone utility, **not** by the training loop. This separates split definition from training, makes the partition step explicit and independently testable, and avoids race conditions when 4 grid jobs start concurrently.

1. **Build the partition (once, before any training submission)**:
   ```bash
   python -m scripts.make_tissue_partition \
       --data_dir data/01_processed/tissue_sets/yoran_ctx4096 \
       --train_cell m84108_251007_115244_s1 \
       --heldout_cell m84108_250930_153107_s2 \
       --train_per_tissue 6250 \
       --val_train_cell_per_tissue 1250 \
       --val_heldout_cell_per_tissue 1250 \
       --seed 42
   ```
   By default the script writes to `<data_dir>/partition.csv` (override with `--output_path`). It reads `manifest.parquet`, sorts by `read_name` for canonical ordering, seeds `np.random.default_rng(42)`, and draws stratified samples per tissue. If `partition.csv` already exists the script refuses to overwrite (use `--overwrite` to replace).

2. **Training submission**:
   - Each size's `train.py` reads `partition_path` from its config (defaults to `<data_dir>/partition.csv`). It asserts the file exists on startup; if missing, it errors out with a pointer to the partition script. The training loop never recomputes or modifies the partition.
   - Resumes load the same `partition.csv`. Version drift in polars / numpy cannot silently shift the partition mid-experiment.

3. **Construction of each `TissueMemmapDataset`** in `_shared_train.py`:
   ```python
   partition = pl.read_csv(partition_path)
   train_names = partition.filter(pl.col('split') == 'train')['read_name']
   train = TissueMemmapDataset(
       data_dir,
       filter_expr=pl.col('read_name').is_in(train_names),
       norm_fn=norm_fn,
   )
   ```

CSV chosen over parquet for this small (~70k row) audit artifact: the partition is read once at startup, never on a hot path, and the dominant requirement is human-readable verifiability. `cat`, `grep`, `diff`, and `git` all work directly on CSV; parquet would require polars or pyarrow to inspect. Project convention already uses CSV for small audit logs (`results.csv`, `eval_history.csv`) and parquet for bulk typed datasets (`manifest.parquet`, ~1.5M rows). The partition fits the CSV side of that split.

This gives absolute verifiable determinism: anyone can read `<data_dir>/partition.csv` and see exactly which reads were in train vs each val set. The algorithm and seed are auditable in `scripts/make_tissue_partition.py`; the resulting split is auditable on disk and travels with the dataset.

IO-wise: the per-split rows are scattered across the same set of shard files (the build script wrote in BAM iteration order, not by cell or split). With `ChunkedRandomSampler(chunk_size=2048)`, reads tend to land in the same shard for each chunk, so the LRU shard cache (size 8 in `TissueMemmapDataset`) is effective despite the scatter.

## Training

- Optimizer: `AdamW`, `lr=3e-3`, `weight_decay=0.02` (matches supervised_40/51). No SSL-pretrained init in v1; from random init.
- Schedule: cosine with `pct_start=0.1`, **not** wrapped by `accelerator.prepare(scheduler)`. The supervised_40/v51 fix is preserved (`AcceleratedScheduler` would compress the cosine horizon by `num_processes`).
- `bf16` mixed precision, `find_unused_parameters=True`.
- `grad_clip=5.0` with non-finite-grad skip + `nonfinite_skip_count` TB scalar (defensive; pattern from ssl_57/58).
- Loss: `nn.CrossEntropyLoss()`. Targets are `long`.
- Walltime: 4h per size.

## Logging

Per-step TensorBoard (every step):

- `train_loss` (cross-entropy, reduced across ranks)
- `learning_rate`
- `grad_norm`
- `embed_z_std`, `embed_z_norm` - logged on the encoder transformer output `c` via a forward hook. Lifted from ssl_58. Not collapse-pathological in a supervised setting but cheap and useful for spotting representation degeneracy.
- `step_time_ms`, `iters_per_sec`, `nonfinite_skip_count`

Eval cadence: every 5,000 global steps. At each eval tick:

- Run validation on both `val_s1` and `val_s2`.
- Log to TB under each val split's namespace: `val_s1/loss, val_s1/top1, val_s1/top3, val_s1/macro_f1` and the same with `val_s2/` prefix. Optional confusion matrix as a TB image. (If `partition.csv` is regenerated for a different dataset, the namespaces follow whatever the partition's val split names are.)
- Append a row to `<size_dir>/eval_history.csv` with all the above plus `walltime_s`.
- Save `<size_dir>/checkpoints/step_<N>.pt` containing full model state, encoder-only state, config, step, metrics, and `KineticsNorm.save_stats()`. Loadable for offline inference at any step.

5k cadence balances overfit-curve resolution against eval cost: at the smallest model that's ~17 - 40 evals in 4h; at the largest ~4 - 8.

## Layout

```
data/01_processed/tissue_sets/yoran_ctx4096/
    schema.json
    manifest.parquet
    partition.csv                              # produced by scripts/make_tissue_partition.py
    shard_*.npy

scripts/experiments/supervised_52_tissue/
    README.md
    _shared_train.py
    size_d128_L4/{config.yaml, train.py, checkpoints/, eval_history.csv}
    size_d256_L8/{config.yaml, train.py, checkpoints/, eval_history.csv}
    size_d512_L8/{config.yaml, train.py, checkpoints/, eval_history.csv}
    size_d768_L8/{config.yaml, train.py, checkpoints/, eval_history.csv}
```

`_shared_train.py` is a single training loop used by all four sizes. Per-size knobs (`d_model`, `n_layers`, `n_head`, `batch_size`) live in `size_*/config.yaml`; everything else is identical across the grid.

`partition.csv` lives in the data directory alongside the manifest and shards. It is produced by the standalone `scripts/make_tissue_partition.py` once, then read (never written) by all four sizes' training scripts. One partition is shared across the architecture grid since each size evaluates the same data.

## Submission

1. **Build the partition (run once before any training submission):**
   ```bash
   python -m scripts.make_tissue_partition \
       --data_dir data/01_processed/tissue_sets/yoran_ctx4096 \
       --train_cell m84108_251007_115244_s1 \
       --heldout_cell m84108_250930_153107_s2 \
       --train_per_tissue 6250 \
       --val_train_cell_per_tissue 1250 \
       --val_heldout_cell_per_tissue 1250 \
       --seed 42
   ```
   Writes `data/01_processed/tissue_sets/yoran_ctx4096/partition.csv` by default.

2. **Submit each size as an independent sbatch job:**
   ```bash
   bash run.sh scripts/experiments/supervised_52_tissue/size_d128_L4
   bash run.sh scripts/experiments/supervised_52_tissue/size_d256_L8
   bash run.sh scripts/experiments/supervised_52_tissue/size_d512_L8
   bash run.sh scripts/experiments/supervised_52_tissue/size_d768_L8
   ```

   Each `train.py` resolves `partition_path` from its config (default `<data_dir>/partition.csv`) and asserts the file exists; if missing, it errors out immediately. All four sizes can run concurrently on the 32-GPU budget (4 nodes x 8 H100s).

## Inference artifact

Each `step_<N>.pt` is a self-contained inference artifact:

```python
import torch
from smrt_foundation.model import TissueClassifier
from smrt_foundation.normalization import KineticsNorm

ckpt = torch.load('size_d128_L4/checkpoints/step_50000.pt', map_location='cpu')
cfg = ckpt['config']['classifier']
model = TissueClassifier(
    d_model=cfg['d_model'], n_layers=cfg['n_layers'],
    n_head=cfg['n_head'], max_len=cfg['context'],
    n_classes=cfg['n_classes'], n_continuous=cfg['n_continuous'],
)
model.load_state_dict(ckpt['model_state_dict'])
norm = KineticsNorm.load_stats(ckpt)
```

## Caveats and things to watch

- The 50k train subset is small enough that the smaller models should hit near-zero training loss within walltime. If training loss stalls high, the architecture or normalization has a bug; debug before scaling subset size.
- Effective batch size is not constant across the grid. Cross-size accuracy comparisons here are approximate, not apples-to-apples.
- Initial per-GPU batch sizes are conservative starting points; revise upward after the first run reads memory headroom.
- The held-out cell `s2` is from a single sequencing run, so the across-cell gap reflects a within-individual cell-batch effect. Across-individual generalization is a separate experiment that requires a multi-individual labeled dataset.
- No data augmentation in v1. The deterministic center-crop and absence of any kinetics jitter favor the overfit demo. Adding augmentation comes after we have confirmed the architecture trains cleanly.
