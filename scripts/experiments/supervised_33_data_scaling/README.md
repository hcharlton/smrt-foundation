# Experiment 33: Data Scaling

## Objective

Measure how CpG methylation classification performance scales with training data volume. Same model architecture and hyperparameters as exp 31 (DirectClassifier, d_model=128, n_layers=4, n_head=4, ctx=32), with only training set size varying.

## Design

- **Training sizes**: 100, 500, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k
- **Steps per size**: 200k (fixed budget — smaller datasets see more epochs)
- **Evals per size**: 10 (every 20k steps)
- **Validation**: First 1M samples (deterministic, no shuffle)
- **GPUs**: 8 (parallelised via `torch.multiprocessing.spawn`, 1 GPU per training size, no DDP)
- **Controlled variables**: Seed (42), optimizer (AdamW lr=3e-3 wd=0.02), cosine schedule (pct_start=0.1), batch size (512)

Each training size gets the same number of optimizer steps. For 100 samples this means 200k epochs; for 128k samples this means ~800 epochs. This intentionally lets small datasets overfit, revealing the full data scaling curve shape.

All 10 sizes run in parallel across 8 GPUs (GPUs 0-1 handle 2 sizes sequentially, GPUs 2-7 handle 1 each). Each worker writes its own per-size CSV, which are merged into `results.csv` at the end. For each training size, the model is reinitialised from the same seed, KineticsNorm is recomputed from that training subset, and a fresh optimizer/schedule is created.

## Outputs

### CSV (`results.csv`)

100 rows (10 sizes x 10 eval points):

| Column | Description |
|--------|-------------|
| train_size | Number of training samples |
| eval_point | Evaluation checkpoint (1-10) |
| step | Optimizer step at this eval |
| train_loss | Average training loss since last eval |
| val_loss | Average validation loss |
| val_f1 | Binary F1 on validation set |
| val_auroc | Area under ROC curve |
| val_auprc | Area under precision-recall curve |
| val_accuracy | Binary accuracy |
| epochs_completed | Number of full passes through training data |

### TensorBoard (`training_logs/`)

One run directory per training size (`n100/`, `n500/`, ..., `n128000/`). All runs appear together in TensorBoard for visual comparison.

### Checkpoints (`checkpoints/`)

One checkpoint per training size after the final step: `n{size}_final.pt`. Contains model weights, encoder weights, config, normalization stats, and final metrics.

Loading a checkpoint:
```python
ckpt = torch.load('checkpoints/n128000_final.pt', map_location='cpu')
c = ckpt['config']['classifier']
model = DirectClassifier(d_model=c['d_model'], n_layers=c['n_layers'],
                         n_head=c['n_head'], max_len=c['context'])
model.load_state_dict(ckpt['model_state_dict'])
norm_fn = KineticsNorm.load_stats(ckpt)
```

## Submission

```bash
bash run.sh scripts/experiments/supervised_33_data_scaling
```
