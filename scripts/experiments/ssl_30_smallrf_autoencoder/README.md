# Experiment 30: Small-RF autoencoder (A1 variant)

Trains `SmrtAutoencoderSmallRF` on CpG-centered windows (labels discarded). Identical to experiment 25 except the encoder uses `CNNSmallRF` — a 4-block CNN (k=3, k=3, k=3 stride=2, k=3 stride=2) with receptive field 27 instead of the default 11-block CNN with receptive field 107. The 4x downsampling ratio is preserved, so latent count and positional geometry match exp 25 at ctx=32.

Hypothesis: at ctx=32, the default CNN's RF=107 causes every latent to be a function of the entire input, erasing per-latent locality. Masked-prediction SSL objectives implicitly depend on this locality ("predict the missing position given the surrounding context"). Restoring RF < context should restore locality and improve probe/fine-tune accuracy. If this closes the 3pp gap between exp 27 fine-tune (79%) and exp 20 supervised baseline (82%), the RF mismatch was the dominant remaining bottleneck.

Every other knob (data, normalization, optimizer, loss, probe schedule) is identical to exp 25 so the probe delta is attributable to the CNN architecture alone.
