# Research Status

*Last updated: 2026-05-08*

## Active objectives

Two parallel threads are running:

1. **Tissue provenance**: paused after the sklearn probe sweep falsified the premise on 50k labelled reads.
2. **CpG methylation foundation model**: scaling autoencoder pretraining and revamping fine-tuning to close the LP→FT gap.

The Gefion grant expires in ~144 hours (2026-05-14). 7,500 H100-hours remaining; this is the last weekend window of unconstrained compute. After expiry, capacity drops sharply.

## Established CpG results

| Result | Experiment | Notes |
|--------|-----------|-------|
| Supervised baseline ~82% top-1 | supervised_20 | DirectClassifier d=128/L=4, 20ep, full ds |
| Supervised v40 still climbing at ep20: 0.808 → 0.818 | supervised_40_baseline_v2 | Scheduler-fix preserved (no AcceleratedScheduler wrap) |
| Autoencoder probe 66% top-1 | ssl_25 | First SSL variant with positive transfer |
| Fine-tune ssl_25 encoder → 79% (+13pp over LP) | supervised_27 | 3pp below supervised, schedule-confounded |
| Every contrastive variant ssl_50–57 fails to beat random init or collapses | — | See `docs/negative_results.md` |

## Current state of CpG work

### Why the new wave of experiments

`supervised_51_finetune_ssl58_grid` showed a much smaller LP→FT lift on the ssl_58 checkpoints than ssl_25 saw: at d=512/d=768, fine-tuning barely beats the linear probe, where ssl_25 had +13pp. This is structurally suspicious. Three candidate explanations the new experiments target:

1. **Top-layer reconstruction-saturation**: at scale, the autoencoder loss may pressure the top transformer layer to be reconstruction-specialised, displacing classification-relevant features into middle layers. Fine-tune always reads the top layer, so the read-out is wrong, not the representation.
2. **Recipe mismatch**: uniform-LR unfreeze shocks the encoder; layer-wise LR decay and an LP-FT warmup are standard MAE/BERT fine-tune practice and ssl_58 fine-tunes use neither.
3. **Cosine never finished at scale**: d=768's 1M-step schedule needs ~320h of d=768 throughput; the 96h base/long variants never reach min-lr. So the d=768 plateau may be a schedule artefact, not a capacity limit.

### What's queued for the weekend

**Pretraining (3 new SSL jobs, each 1 node × 8 H100, 120h walltime)**

- `ssl_58_autoencoder_grid/size_d1024_L8` (A1) — one tier wider than the existing grid. bs=32/rank to preserve activation memory at d=1024.
- `ssl_58_autoencoder_grid/size_d768_L8_finished_cosine` (A3) — total_steps=350k so the cosine actually bottoms inside walltime at d=768 throughput.
- `ssl_59_mae/size_d512_L8` (B1) — true MAE sparse encoder (`SmrtAutoencoderMAE`): encoder transformer sees only the kept ~25% post-CNN latents, decoder fills in [mask] tokens and reconstructs kinetics. mask_ratio=0.75, decoder_n_layers=2, bs=192/rank.

**Fine-tune revamp (4 new supervised jobs, 24 combos each, 1 node × 8 H100, 24h walltime)**

All under `supervised_53_finetune_revamp/`, driven by new `scripts/ds_grid_v3.py` (treatment-aware extension of v2). All four read ssl_58 checkpoints via `scripts/utils/select_best_ssl_checkpoint.py` — picks the best-probe step per size, not `final_model.pt`, since smaller ssl_58 sizes peak before the end of training.

| Treatment | What it tests | New class |
|---|---|---|
| `midlayer/` (F1) | Top-layer-saturation hypothesis; reads from configurable `layer_idx` | `DirectClassifierMidLayer` + `SmrtEncoder.forward_to_layer` |
| `lpft_lldr/` (F2) | Recipe-mismatch hypothesis; LP-FT + layer-wise LR decay (decay=0.7) | (uses `DirectClassifier(SmallRF)` + new optimizer wiring) |
| `decoder_init/` (F3) | "Don't throw the projection head away"; reuses SSL `SmrtDecoder.upsample` | `DirectClassifierWithDecoder` |
| `recipe_match/` (F4) | Head / recipe control for F1; supervised_20 recipe + 3-layer MLP head | `DirectClassifierBigHead` |

### Free harness improvements (apply to all new runs)

- `train_history.csv` (per-LOG_EVERY): step, train_loss, lr, grad_norm, embed_z_std/norm, step_time_ms, iters_per_sec, nonfinite_skip_count.
- `pair_val_history.csv` (long format): one row per (step, split, gap) for clean post-hoc plotting.
- ssl_58 milestones now bundle `decoder_state_dict` (`SmrtDecoder.upsample` + Linear(d, 2)) and `mask_config` so F3 can warm-start the supervised classifier from them. ssl_59 milestones bundle the full non-encoder portion of the model under `decoder_state_dict` (decoder_blocks/decoder_pe/decoder_upsample/mask_token) for symmetric reusability.

## Tissue provenance — paused

Sklearn probe sweep (2026-05-07) confirmed: at 50k labelled yoran reads, kinetics features at every granularity (per-window summary, per-bin summary, per-position) carry no extractable tissue signal that generalises across reads or cells. Per-tissue 1-vs-rest AUROCs lie in [0.50, 0.57]. Cell s1-vs-s2 LogReg val 0.55 — small but real batch effect. The deep model trains to memorisation; an sklearn probe with the same features does the same. No further iteration on supervised_52 at this dataset size.

## Open questions

1. **MAE vs BERT-style autoencoder**: does the asymmetric encoder/decoder produce a cleaner LP→FT lift than ssl_58's in-place input masking?
2. **Top-layer saturation**: does F1 (middle-layer read-out) close the LP→FT gap?
3. **Recipe vs representation**: does F2 (LP-FT + LLDR) close the gap on its own? If so, F1's middle-layer story is weakened.
4. **Decoder reuse**: does F3 beat the random-init upsample baseline?
5. **Capacity ceiling**: does d=1024 (A1) keep paying past d=768?
6. **Cosine artefact**: does A3 (d=768 with finished cosine) beat the existing d=768 plateau?

## Pending experiments

| Experiment | Status |
|-----------|--------|
| ssl_58_autoencoder_grid/{d128_L4, d256_L8, d512_L8, d768_L8, *_long, *_ctx4096} | In flight on Gefion |
| ssl_58_autoencoder_grid/size_d1024_L8 (A1) | Built, ready to submit |
| ssl_58_autoencoder_grid/size_d768_L8_finished_cosine (A3) | Built, ready to submit |
| ssl_59_mae/size_d512_L8 (B1) | Built, tests pass, ready to submit |
| supervised_53_finetune_revamp/{midlayer, lpft_lldr, decoder_init, recipe_match} (F1-F4) | Built, tests pass, midlayer probe sweep pending before submit |
| supervised_52_tissue (4 sizes) | Paused — sklearn probe sweep falsified premise |
