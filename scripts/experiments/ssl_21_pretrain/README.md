# Experiment 21: SSL pretraining with input masking

SSL pretraining using `Smrt2VecInputMask` (wav2vec 2.0 style input masking) on full-length PacBio reads. Trains a contrastive encoder to reconstruct masked kinetics from surrounding context.

## Result

The InfoNCE loss converges, but the linear probe accuracy on CpG methylation classification **declines** over training epochs. Three issues were identified.

### Issue 1: Normalization mismatch (fixed)

The probe was computing its own `KineticsNorm` from the CpG dataset, while the encoder was trained on SSL data normalized with different statistics. The encoder saw out-of-distribution inputs during probing. Fixed by passing the SSL normalization through to the probe.

### Issue 2: Sequence length mismatch

The encoder trains on 4096-position sequences (→ 1024 positions after CNN downsampling). The probe evaluates on 32-position CpG windows (→ 8 positions). The transformer learns attention patterns for 1024-position sequences that don't transfer to 8 positions — softmax attention becomes nearly uniform. The CNN's ~107-base receptive field also operates in a fundamentally different regime: for 32-length input every position sees the entire input, whereas for 4096-length input each position sees a local window.

This mismatch amplifies over training as the transformer specializes for long-sequence patterns.

### Issue 3: Task alignment question

The contrastive loss teaches reconstruction of local kinetics from context: "given surrounding kinetics, predict what should be at position X." For methylation classification, the encoder needs to detect subtle kinetics shifts at CpG sites relative to what's expected for that sequence context. The SSL task treats all positions equally — CpG sites have no special status — and as the encoder specializes in reconstruction, it may discard the methylation signal as noise.

The declining probe accuracy suggests the encoder's randomly initialized features had incidental correlation with methylation that gets destroyed as SSL training progresses.

## Follow-up experiments

- **Experiment 23** (`ssl_23_shortctx`): Tests issue 2 by pretraining with context=128 to close the length gap
- If both issues 1 and 2 don't help, issue 3 (task alignment) needs a different pretraining approach
