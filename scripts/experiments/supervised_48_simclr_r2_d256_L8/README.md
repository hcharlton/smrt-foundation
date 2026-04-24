# Experiment 48: ssl_53 SimCLR d256_L8 → supervised fine-tune

Fine-tunes the Round-2 SimCLR encoder (`ssl_53_simclr_grid_r2_step/size_d256_L8`) on the CpG methylation classification task via the unified `ds_grid` trainer, matching exp 47's baseline schedule so the only meaningful differences from the random-init baseline are (a) encoder init from a pretrained checkpoint and (b) the larger d256 / L8 architecture the SimCLR checkpoint requires.

## Why

Exps 42–46 all fine-tuned a small d128 / L4 encoder (ssl_29 autoencoder, ~2M params). ssl_53 is the first SimCLR encoder in the d256 / L8 regime (~11M params) trained to full-dataset epochs with step-based probing — the grid's best prospect for encoder quality so far. No downstream comparison has been run for it yet. This experiment establishes whether the SimCLR checkpoint transfers to classification at a level competitive with, or exceeding, the d128 autoencoder lineage.

Using `ds_grid.py` rather than a bespoke train script keeps the step budget, eval schedule, and logging format byte-identical to exps 47 / 46, so scaling curves can be overlaid directly.

## Configuration

- **Encoder**: `scripts/experiments/ssl_53_simclr_grid_r2_step/size_d256_L8/checkpoints/final_model.pt` (edit to `step_<N>.pt` if `final_model.pt` isn't yet present on the cluster)
- **Classifier arch**: d=256, L=8, h=4, ctx=32 — matches the SimCLR encoder dimensions (no PE reinit needed since ctx is unchanged)
- **Schedule**: single-stage AdamW at `max_lr=3e-3` for all params, cosine with `pct_start=0.1`. No frozen-head phase — `ds_grid.py` enters the single-stage branch because `finetune:` is absent from the config.
- **Batch size**: `min(4096, max(64, n_train // 8))` per size. Same formula as exp 46.
- **Step budget**: `min(max(200*steps_per_epoch, 10k), 400k)` — identical to exps 41/46/47.
- **Train sizes**: 16 log-spaced points, 100 → 8M. Val limit 1M.

## Hypothesis

If the ssl_53 encoder has learned useful kinetics representations, exp 48 should beat exp 47's scaling curve across most train sizes, and the gap should be largest at small n (where a strong initialisation matters most). If exp 48 matches or lags exp 47 at large n, the SimCLR pretraining provides no downstream lift once enough supervised data is available — and the experiment becomes a cost-benefit question about the small-n regime.

## Submission

```bash
bash run.sh scripts/experiments/supervised_48_simclr_r2_d256_L8
```
