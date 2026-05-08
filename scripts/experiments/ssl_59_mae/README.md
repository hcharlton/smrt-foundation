# ssl_59_mae

True MAE pretraining (He et al. 2022 asymmetric encoder/decoder) lifted
onto the CNN-transformer hybrid encoder.

## Why

ssl_58 established that masked-kinetics reconstruction beats every
contrastive variant we tried, but the BERT-style in-place input masking
ssl_58 uses is not the modern MAE structure: the ssl_58 encoder still
sees all positions, just with kinetics zeroed at masked spans. He et al.
2022 (and most subsequent MAE work) drops the masked tokens from the
encoder entirely and lets a small decoder transformer fill them in
afterward. The asymmetry is supposed to push the encoder representation
harder by forcing it to be useful for cross-position reconstruction, not
just local denoising.

We also have a more specific motivation: ssl_58's LP→FT gap looks
suspiciously small at scale (d=512 / d=768) compared to the 13pp lift
ssl_25 saw. One hypothesis is that the autoencoder loss pressures the
top transformer layer to be reconstruction-specialised, displacing
classification-relevant features into middle layers. MAE's asymmetric
decoder absorbs more of that "reconstruction-specific" load, which may
leave the encoder top layer more useful for downstream tasks. The
fine-tune revamp (supervised_53) also tests middle-layer features
directly and provides a clean parallel comparison.

## Architecture

`SmrtAutoencoderMAE` (`smrt_foundation/model.py`):

- **Encoder**: shared `SmrtEncoderSmallRF` (CNN RF=27, 4x downsample,
  same as ssl_58's grid).
- **Mask point**: AFTER the CNN, on the T/4 latents. Stride-2 convs
  assume contiguous inputs so dropping pre-CNN tokens would break
  locality. Post-CNN, the latents are already at T/4 resolution and
  dropping is safe.
- **Encoder transformer**: processes only the kept (~25%) latents.
  ~4x fewer transformer ops vs ssl_58 at the same architecture.
- **Decoder**: a small transformer (2 layers by default), full T/4
  sequence with a learnable `[mask]` token at dropped positions, plus
  the existing `SmrtDecoder` ConvTranspose1d upsample T/4 → T and
  Linear(d, 2) head. Same loss target as ssl_58: kinetics MSE at masked
  positions.

## What changes vs ssl_58's harness

Three places in the harness:

1. **Model**: `SmrtAutoencoderMAE` instead of `SmrtAutoencoderSmallRF`.
2. **Mask config**: `mask_ratio` (default 0.75) and `decoder_n_layers`
   (default 2) replace `p_mask` and `mask_size` in DEFAULT and in the
   resume-compatibility arch_keys check.
3. **Decoder save format**: the decoder is now split across
   `decoder_blocks`, `decoder_pe`, `decoder_upsample`, and `mask_token`,
   so milestone saves bundle the full non-encoder portion of the model
   into a single `decoder_state_dict` dict (keys preserve their original
   prefixes so a downstream loader can pick out the upsample stack).

## Pass criterion

Same as ssl_58 d=128: `probe_top1 ≥ 0.67` and non-decreasing over the
last 3 evals. At d=512 the more interesting target is the LP→FT gap on
supervised_53 fine-tunes — does MAE pretraining produce a representation
where fine-tuning lifts probe accuracy more than ssl_58 d=512 did?

## Layout

- `_shared_train.py` — branched from ssl_58's; swaps in
  `SmrtAutoencoderMAE` and adjusts the save format.
- `size_d512_L8/{config.yaml, train.py}` — the only size in the lineage
  for now. Per-rank batch=192 vs ssl_58 d=512's bs=128 because the
  encoder transformer only sees 25% of latents.

Submit: `bash run.sh scripts/experiments/ssl_59_mae/size_d512_L8`
