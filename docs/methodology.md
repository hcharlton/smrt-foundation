# Methodology Log

## 2026-02: Hybrid Embedding (Split Nucleotide + Kinetics)

**Motivation:** PacBio data has two fundamentally different input types: categorical nucleotide tokens (A/C/G/T/N) and continuous kinetics values (IPD, pulse width).

**Method:** `SmrtEmbedding` concatenates a learned `Embedding(5, d/2)` for nucleotides with a `Linear(2, d/2)` projection for kinetics, followed by LayerNorm. Both branches are scaled by `sqrt(d_model)`.

**Justification:** Linear projection preserves numeric relationships between kinetics values (a 2× difference in IPD maps to a 2× difference in embedding space). A shared embedding would force continuous values through a discrete lookup, destroying this structure.

## 2026-02: CNN with 4x Downsampling Before Transformer

**Motivation:** Raw PacBio reads are 4000-20000 bases. Self-attention is O(T^2), making full-length sequences computationally prohibitive.

**Method:** 11 ResBlocks with two stride-2 layers produce 4× downsampling. GroupNorm (not BatchNorm) for per-sample stability. Receptive field ≈ 107 bases.

**Justification:** 4× reduction makes T=4096 sequences tractable (1024 positions for attention). The 107-base receptive field captures multi-base kinetics context. GroupNorm avoids batch-level statistics that could collapse per-read variation.

## 2026-02: Input-Level Masking (Not Latent-Level)

**Motivation:** Original `Smrt2Vec` masked at the latent level (after CNN). The CNN's 107-base receptive field meant adjacent unmasked latents shared 96% of input bases with masked positions, making reconstruction trivially easy.

**Method:** `Smrt2VecInputMask` zeros kinetics channels (IPD, PW) at the input level before the CNN. Sequence tokens are preserved. A conservative downsampling rule maps input masks to latent masks (any overlapping masked input → latent is masked).

**Justification:** Information genuinely disappears at the input. The CNN cannot fill in missing kinetics from overlapping receptive fields. Forces the encoder to learn from broader context rather than local interpolation.

## 2026-02: Per-Read MAD Normalization for SSL Memmaps

**Motivation:** Raw kinetics values vary across reads due to sequencing conditions (polymerase speed, loading concentration). Global z-score is sensitive to outliers.

**Method:** `normalize_read_mad` applies per-read: log1p → subtract median → divide by MAD × 1.4826 (consistency constant for Gaussian equivalence).

**Justification:** MAD is robust to outliers (a single extreme IPD value doesn't distort the entire read's normalization). Per-read normalization removes inter-read variation while preserving intra-read kinetics patterns.

## 2026-02: Global KineticsNorm for Online Normalization

**Motivation:** Per-read MAD normalization requires the full read at inference time (not available for short CpG windows). Need a normalization that works on fixed-length windows.

**Method:** `KineticsNorm` computes global mean/std from a sample of the training data (excluding padding positions), then applies log1p + z-score normalization to each window independently. Same instance shared between pretraining and probe evaluation.

**Justification:** Global statistics are stable and reproducible. Sharing the normalization instance between SSL and downstream prevents distribution shift at transfer time (a lesson from experiment 21 where separate normalization caused mismatch).

## 2026-02: Separate Positive/Negative Shard Directories for Labeled Data

**Motivation:** CpG methylation is a binary classification task. Need balanced training data from methylated and unmethylated reads.

**Method:** `LabeledMemmapDataset` loads from two directories (pos/neg). Labels are assigned based on directory membership (pos=1.0, neg=0.0). Optional balancing truncates to `min(len(pos), len(neg))` shards.

**Justification:** Keeps data generation simple (one script per condition). Directory-based labeling avoids encoding labels in the tensor format. Balance option prevents class imbalance from dominating BCE loss.

## 2026-02: Center-Position Classification Head

**Motivation:** CpG methylation is a site-specific prediction. A fixed-length context window is centered on the CpG site.

**Method:** `DirectClassifier` extracts the center position from the encoder output: `c[:, T//2, :]` → MLP → logit. No pooling or attention aggregation.

**Justification:** The CpG site is always at the center by construction. Using only the center position avoids diluting the signal with flanking context that may not be informative. Simpler than attention pooling with fewer parameters to overfit.

## 2026-03: AgInfoNCE with max_negatives Subsampling

**Motivation:** Input-level masking with p_mask=0.15 produces many masked positions. With 8 GPUs and all_gather, the similarity matrix can exceed GPU memory.

**Method:** `AgInfoNCE` optionally subsamples masked positions to `max_negatives` before computing the similarity matrix. Targets are gathered across all ranks for more negatives.

**Justification:** Subsampling keeps memory tractable without fundamentally changing the contrastive objective. More negatives (via all_gather) make the task harder and the representations more discriminative.

## 2026-03: Shallow Decoder for Masked Autoencoder

**Motivation:** A deep decoder can reconstruct inputs from minimal encoder features, allowing the encoder to be lazy.

**Method:** `SmrtDecoder` has only two ConvTranspose1d layers (4× upsampling to reverse CNN) and a single linear layer (d→2 kinetics channels).

**Justification:** With minimal reconstruction capacity in the decoder, the encoder must learn rich, informative representations to support accurate kinetics reconstruction. This is the same design principle as MAE (He et al., 2022): asymmetric encoder-decoder where the encoder does the heavy lifting.

## 2026-03: Pretraining Data Scale Must Exceed Labeled Data

**Motivation:** Exp 27 fine-tuned the exp 25 autoencoder encoder and reached 79% — 3pp below the supervised baseline trained from scratch (82%). The pretraining data (CpG memmap with labels removed) is a subset of the fine-tuning data (same CpG memmap with labels). At a 1:1 unlabeled:labeled ratio, pretraining provides no information advantage.

**Observation:** wav2vec 2.0 uses 10-6000x more unlabeled data than labeled. The SSL value proposition is leveraging abundant unlabeled data that can't be labeled. When pretraining and fine-tuning use the same data, pretraining adds only a different loss function's inductive bias, not additional information.

**Implication:** Future pretraining experiments should use a larger unlabeled corpus (e.g., the full ob007_raw.memmap at ~839K reads) and fine-tune on a smaller labeled subset. The data regime mismatch (full reads vs CpG windows) remains a challenge but fine-tuning may bridge it where linear probing couldn't.

## 2026-03: SSL on Downstream Data Distribution (CpG Windows)

**Motivation:** Experiments 21-24 trained SSL on full-read segments (ob007_raw.memmap, context=4096 or 128) but evaluated on 32-base CpG windows. Three mismatches: data distribution (genome-wide vs CpG-biased), context length (128→32), and normalization statistics. All three compound to prevent transfer.

**Method:** Experiments 25/26 train SSL directly on the CpG memmap data (pos + neg combined, labels discarded). Context=32 matches the probe exactly. KineticsNorm computed from CpG data. Both autoencoder and contrastive architectures tested head-to-head on the same data.

**Justification:** This is a diagnostic experiment. If probe accuracy improves significantly, the bottleneck was data regime, not the pretraining objective. If it doesn't, the reconstruction/contrastive loss itself is insufficient for learning discriminative methylation features regardless of data. Reduced mask_size from 10 to 5 (proportional to shorter context: 5/32 ≈ 10/64).

## 2026-03: Cosine Schedule with 25% Warmup

**Motivation:** Transformer training is sensitive to learning rate scheduling. Too-fast warmup destabilizes attention weights; no warmup causes divergence.

**Method:** Linear warmup over 25% of total steps, then cosine decay to 5% of peak LR.

**Justification:** 25% warmup is conservative but stable for the relatively small model (128d, 4 layers). Cosine decay is smooth and avoids the sharp transitions of step schedules. The 5% minimum LR floor prevents complete stagnation in late training.

## 2026-03: Data-Budget Controlled Comparison (exp 28)

**Motivation:** Exp 27 (fine-tuned autoencoder encoder) reached 79% vs exp 20's 82%, but the comparison is confounded: exp 20 trains on the full dataset while exp 27 uses ds_limit=20M. The 3pp deficit might be explained by data budget alone rather than pretraining being harmful.

**Method:** Exp 28 reruns exp 20's DirectClassifier from scratch with ds_limit=20M, matching exp 27's data budget exactly. All other hyperparameters (lr=3e-3, 20 epochs, single-stage optimizer) remain identical to exp 20.

**Justification:** This is the minimal control needed to interpret exp 27. If exp 28 also lands at ~79%, the deficit is from the smaller data budget, not from pretraining. If exp 28 still reaches ~82%, pretraining is actively hurting by 3pp under these conditions.

## 2026-03: Random Cropping for Large-Scale Pretraining (exp 29)

**Motivation:** Previous SSL experiments pre-segment reads into fixed windows, creating a fixed-size dataset. With 839K reads and context=128, this yields ~27M windows. Training for 3000 epochs on a fixed dataset would simply overfit reconstruction.

**Method:** Instead of pre-segmenting, exp 29 randomly crops a 128-base window from each 4096-base read at each epoch. The crop position is sampled uniformly at runtime, so the model sees different subsequences every epoch. This creates a virtually infinite dataset from the same reads.

**Justification:** Random cropping makes long training schedules meaningful: 3000 epochs x 839K reads = 2.5B training windows, each unique. This is the first experiment with a genuine data scale advantage over the labeled CpG data (~40M windows). The 128-base crop length matches exp 24's context while being large enough for the 107-base CNN receptive field.

## 2026-03: Receptive Field Logging

**Motivation:** The CNN's receptive field determines how much local context the encoder uses at each position. When comparing experiments with different kernel configurations or context lengths, the effective receptive field relative to context is an important architectural parameter.

**Method:** `model.encoder.cnn.r0` is computed from the CNN's kernel sizes and strides and logged to both stdout and TensorBoard at training startup.

**Justification:** Avoids manually re-deriving the receptive field from kernel configs when reviewing experiment logs. Makes it immediately visible whether the receptive field exceeds the context window (which would mean every output position sees the entire input).

## 2026-04: Small-RF CNN Variant (CNNSmallRF, r0=27)

**Motivation:** Empirical measurement of the mask-sampling geometry in `scripts/experiments/ssl_26_cpg_contrastive/measure_mask_fractions.py` and the receptive-field comparison across exp 21/23/26 revealed a structural problem at ctx=32: the default CNN's receptive field (107) exceeds the context (32), so every CNN output latent is a function of the entire input. Masked-prediction SSL objectives (both contrastive and autoencoder) implicitly depend on per-latent locality — the "predict the missing position given the surrounding context" framing is degenerate when the context IS the entire input. This is the same architectural issue regardless of `p_mask`, `mask_size`, or the specific loss. Exp 25's autoencoder nonetheless reached ~66% probe and exp 27's fine-tune reached 79%, 3pp short of the supervised baseline (82%); the A1 hypothesis is that restoring locality closes that gap.

**Method:** `CNNSmallRF` (`smrt_foundation/model.py`) is a new CNN class with 4 ResBlocks (k=3, k=3, k=3 stride=2, k=3 stride=2) instead of the default's 11 blocks. Receptive field drops from 107 to 27, which fits comfortably inside ctx=32. The 4× downsampling ratio is preserved so latent counts and positional geometry match the default encoder at equal contexts — probe head, decoder, and SSL wrapper return signatures are all unchanged. Two companion classes (`SmrtEncoderSmallRF`, `SmrtAutoencoderSmallRF`) wrap the new CNN and inherit all forward-path methods from their non-variant parents by calling `nn.Module.__init__` to skip the parent constructor. State-dict keys are not interchangeable with the default classes because the ResBlock counts differ; these are fresh-training-only variants.

**Justification:** A1 is the minimum-viable architectural test of the RF hypothesis: one new encoder class, zero new data pipelines, zero new loss functions, zero new probe infrastructure. If exp 30 (autoencoder at ctx=32 with CNNSmallRF) matches or beats exp 25, the RF mismatch was a contributing factor. If exp 30 goes on to fine-tune ≥82%, the mismatch was the dominant remaining bottleneck and the foundation-model target is hit. If exp 30 matches exp 25 at ~66% and fine-tunes at ~79%, the RF was not dominant and the remaining gap must come from elsewhere (loss objective, optimizer schedule, or data scale). In all three cases the result is informative and bounds the next decision.

The RF=27 choice is conservative: 27/32 ≈ 0.84 of the context window, which leaves enough latent diversity for per-position representations to differ without losing so much kinetic context that the encoder starves. Smaller alternatives (RF=19 with 2 stride blocks, or patch embedding with RF=4) are available as follow-up variants if A1 partially resolves the gap.

## 2026-04: SimCLR Instability Diagnostics (grad_norm, embed_z_std, embed_z_norm)

**Motivation:** `ssl_51_simclr_grid_r1/size_d768_L8` ran on the full `ob007_raw.memmap` for 30 h / 55 epochs and collapsed: NT-Xent loss descended from 7.0 to 6.5, then spiked to ~7.8 (near the `log(2N-1) ≈ 8.32` upper bound for effective batch 2048) and plateaued there; probe_top1 decayed from 0.61 on the first eval to 0.50. The d128/d256/d512 points in the same grid trained cleanly under the reduced-data regime (0.63/0.65/0.675 best probe_top1) with identical code, so the failure was specific to the large-model + full-data combination. Loss trajectory (down then up, not monotone-down) ruled out the "shortcut / overfitting" label — under a shortcut the SSL objective gets easier, not harder. Without step-level instrumentation we cannot distinguish the two leading candidate mechanisms (mid-training gradient explosion versus dimensional collapse of the projection), and therefore cannot choose between the fixes that those mechanisms imply (gradient clipping + lower `max_lr` versus longer warmup / larger effective batch / architectural regularizer).

**Method:** Three scalars are logged per training step in `_shared_train.py`, alongside `train_loss`, `learning_rate`, and `epoch`:

- `grad_norm` — total L2 norm across all model parameters, returned by `accelerator.clip_grad_norm_(model.parameters(), max_norm=float('inf'))` immediately after `accelerator.backward(loss)` and before `optimizer.step()`. `max_norm=inf` means "report but do not clip." Swapping in a finite value at the same call-site enables clipping without further restructuring.
- `embed_z_std` — mean over the 128 projection channels of the per-channel standard deviation across the concatenated `[z1; z2]` batch, computed under `torch.no_grad()` on the raw projection outputs *before* NTXent's internal `F.normalize`. Approaches 0 under dimensional / representation collapse.
- `embed_z_norm` — mean L2 norm of the unnormalized projection vectors. Tracks absolute embedding scale; useful for distinguishing "representations collapsed to the origin" from "representations collapsed onto a low-dimensional subspace of the sphere."

All three are reduced across ranks via `accelerator.reduce(..., reduction='mean')` so they report global state, not a single rank's slice.

**Justification:** The two mechanisms have distinguishable signatures in these scalars. A grad-norm that spikes or climbs monotonically in the window just before the loss blow-up points at a training-instability / no-grad-clip failure and argues for `clip_grad_norm_(..., 1.0)` plus a lower `max_lr` for the largest model. A flat or falling grad-norm while `embed_z_std` decays toward zero points at dimensional collapse and argues for longer warmup, larger effective batch (gradient accumulation), or a method with an explicit decorrelation term (VICReg / Barlow Twins). The metrics cost one extra `accelerator.reduce` + `.item()` sync per step on three scalars, which is negligible against the per-step compute of a 102M-parameter model. They are purely observational — no training math changes — so they can be added mid-sweep without invalidating comparisons against earlier runs from the same shared loop (comparisons on `train_loss` / `probe_top1` remain apples-to-apples; the new scalars are simply absent from prior TB event files).

## 2026-04: Persistent Normalization Stats in Checkpoints

**Motivation:** Every training script that uses `KineticsNorm` samples ~1M data points to compute mean/std statistics at startup. Those stats are what the model is actually trained on — a `(log1p(x) - mean) / std` transform on kinetics channels [1, 2]. At inference time a new input has to be normalized with the *same* stats or the model sees a distribution shift it was never trained to handle. Two prior patterns each had a failure mode:

1. **Recompute stats at inference time.** Re-instantiating `KineticsNorm(inference_data)` fits fresh statistics from whatever data is on hand. Only safe if that data exactly matches the training distribution, and even in that case the random sampling adds ~1e-4 of noise vs the training stats. On a novel BAM (different sample, different polymerase conditions), this normalizes out the exact distribution shift the model needs to see.
2. **Reimplement the save boilerplate in every train.py.** Before this change, any experiment that persisted stats wrote `{'norm_means': norm_fn.means.detach().cpu(), 'norm_stds': norm_fn.stds.detach().cpu(), 'norm_log_transform': norm_fn.log_transform}` by hand. Three lines of duplicated boilerplate, with no canonical key schema kept in sync between training scripts and inference code.

**Method:** `KineticsNorm` now exposes a three-method API for stats persistence (`smrt_foundation/normalization.py`):

- `norm_fn.save_stats()` — instance method. Returns `{'norm_means', 'norm_stds', 'norm_log_transform'}` with tensors detached to CPU. Ready to merge into a `torch.save(...)` dict via `**` unpacking.
- `KineticsNorm.load_stats(state)` — classmethod. Extracts the three `norm_*` keys from either a full checkpoint dict or a bare stats dict (other keys are ignored). Defaults `norm_log_transform=True` when absent. Delegates internally to `from_stats`.
- `KineticsNorm.from_stats(means, stds, log_transform=True, eps=1e-8)` — classmethod. Low-level constructor that takes raw tensors directly. Kept for callers that already have stats in hand; most code with a checkpoint should use `load_stats` instead so the key schema stays in one place.

**Integration template for a new train.py:**

```python
from smrt_foundation.normalization import KineticsNorm

# ... dataset setup ...
train_norm_fn = KineticsNorm(tmp_ds, log_transform=True)

# ... training loop ...

# Per-epoch (or final) checkpoint save:
torch.save({
    'model_state_dict': unwrapped.state_dict(),
    'encoder_state_dict': unwrapped.encoder.state_dict(),
    'config': config,
    'epoch': epoch,
    'metrics': metrics,
    **train_norm_fn.save_stats(),   # drops in norm_means, norm_stds, norm_log_transform
}, save_path)
```

**Inference template:**

```python
import torch
from smrt_foundation.normalization import KineticsNorm

ckpt = torch.load('scripts/experiments/<name>/checkpoints/epoch_NN.pt', map_location='cpu')
# ... reconstruct model from ckpt['config'] and load ckpt['model_state_dict'] ...
norm_fn = KineticsNorm.load_stats(ckpt)   # reads norm_* keys, ignores everything else

with torch.no_grad():
    logits = model(norm_fn(x))
```

**When to apply this:** any new training script under `scripts/experiments/` that uses `KineticsNorm` — which should be all of them going forward, since `ZNorm` is deprecated in favor of `KineticsNorm` (they are equivalent on no-padding CpG data; `KineticsNorm` additionally handles padded SSL data correctly). The only line to add is `**norm_fn.save_stats()` inside the `torch.save(...)` dict. No per-script key management, no reimplementation of the detach/CPU incantation.

**When NOT to apply this:** if a script's checkpoint is purely a throwaway (smoke-test runs, benchmark scripts that never load back), skipping `save_stats` is fine. Any checkpoint that might be loaded for inference, fine-tuning, or probe evaluation should include the stats.

**Justification:** 
- *Single source of truth.* The `norm_*` key schema lives in exactly one place (`save_stats` / `load_stats`). If it ever changes — say, to include an `eps` field or version number — only those two methods need updating, with no audit across every train.py and inference script.
- *Bit-exact reproducibility.* Verified end-to-end: `train_norm_fn(x)` and `KineticsNorm.load_stats(ckpt)(x)` produce identical outputs at float32 precision, not "within sampling noise". This matters because a 1e-4 shift in normalization is enough to nudge near-boundary CpG classifications across threshold.
- *No training data required at inference.* Re-instantiating `KineticsNorm(ds)` needs the full training dataset on disk to sample from. `load_stats(ckpt)` only needs the checkpoint file, which is useful for deployment, cross-cluster reproducibility, and sharing a model without shipping the corpus.
- *Symmetric API.* `save_stats()` on the training side pairs 1:1 with `load_stats()` on the inference side. Any code reviewer can read a single line (`**norm_fn.save_stats()`) and immediately know what's in the checkpoint and how to load it back.
