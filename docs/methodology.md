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
