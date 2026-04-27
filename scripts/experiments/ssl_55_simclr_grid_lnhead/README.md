# ssl_55_simclr_grid_lnhead

Same architecture, data, and step budget as ssl_54, with two coordinated
fixes targeting the projection-head magnitude-runaway failure that
collapsed ssl_54's d=768 run.

## What changed vs ssl_54

1. **`SimCLRSmrtLN`** in place of `SimCLRSmrt` — uses `MLPProjectionHeadLN`,
   which adds `LayerNorm` between `Linear` and `GELU` on hidden layers and
   a final `LayerNorm` after the last `Linear` (immediately before
   `F.normalize` inside `NTXent`). LN placement mirrors SimCLR v1 §4.2's
   BN placement; LN rather than BN to avoid DDP sync semantics. Bounds
   the per-channel scale of the projection output and prevents the slow
   magnitude drift that turned into a catastrophic explosion at d=768.
2. **Non-finite-grad skip** in `_shared_train.py`: when
   `accelerator.clip_grad_norm_` returns a non-finite value, skip both
   `optimizer.step()` and `scheduler.step()` for that batch and increment
   a `nonfinite_skip_count` TB scalar. Defensive — converts a single
   outlier batch from "permanently corrupts the run" into "one lost
   step." With the LN head fix this counter should stay at 0; if it ever
   fires, the LN didn't fully prevent the runaway and the diagnosis
   needs revisiting.

Everything else is identical to ssl_54: yoran dataset, ds_limit=0,
epochs=50, batch sizes (512/512/512/256), AdamW max_lr=3e-4, wd=1e-4,
pct_start=0.05, grad_clip=5.0, augmentations, ChunkedRandomSampler,
step-based probe + checkpoint cadences (10k).

## Predicted outcome

- d=768: `embed_z_std` stays bounded (predicted in the 1–5 range, vs
  ssl_54's ~3.6M plateau). `probe_top1` at minimum holds the ssl_54
  epoch-1 peak of 0.65 instead of collapsing to 0.50.
- d=512: bump-and-recover at ~step 100k disappears; probe trajectory
  becomes monotone instead of partially recovering from in-bump damage.
  May exceed ssl_54's ~0.65 plateau.
- Size scaling becomes monotone d=128 → d=768 (the question the grid was
  originally designed to answer).
