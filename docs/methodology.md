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

## 2026-04-28: Contrastive SSL experiments named for the invariant encoded

**Decision:** Contrastive (SimCLR-family) SSL experiments are named for the *invariant the encoder is being asked to learn*, not the architectural mechanism. The invariant is the property of the input that the encoder is trained to ignore — equivalently, the property that defines what makes two views a positive pair.

Examples:
- `ssl_56_simclr_neighbor_invariance` — positives are nearby windows on the same molecule; encoder asked to be invariant to small position shifts.
- A future `ssl_NN_simclr_strand_invariance` — positives are fwd and rev kinetic passes of the same molecule; encoder asked to be invariant to which strand the polymerase traversed.
- *Not:* `ssl_56_simclr_localpair` (mechanism description), `ssl_56_simclr_lnhead_v2` (architectural revision label).

Sub-experiment subdirectories within a single contrastive experiment can describe the *parameter being varied* (e.g., `gap_16/`, `gap_32/` in ssl_56) — that's a property of the sweep design and doesn't violate the convention.

Architectural tweaks orthogonal to the invariant (e.g., `ssl_55_simclr_grid_lnhead` — a LayerNorm fix to the projection head) keep the mechanism-based naming, since they're testing a fix to the architecture given the invariant, not exploring a different invariant.

**Justification:**
- *Experiment log readability.* The log becomes a record of which invariants did or didn't transfer to the downstream task (CpG methylation classification), not a list of loss formulations or architectural revisions. Two contrastive runs with the same architecture but different positive-pair definitions are testing fundamentally different hypotheses about what features transfer; the directory name should make that explicit.
- *Future-proofing.* When the project moves on from contrastive SSL or adds non-SimCLR contrastive methods (e.g., MoCo, DINO), the invariant-named directories still describe what was learned. A method-named directory (`ssl_NN_simclr_*`) becomes meaningless if the method is later replaced.
- *Discoverability.* When asking "have we already tested whether the encoder should be position-invariant within a read?", the answer is in the directory name, not in the README of one of N similarly-named runs.

**Scope:** This convention applies to *contrastive* (positive/negative pair) SSL only. Masked-prediction SSL (Smrt2Vec, autoencoder lineages) names by mechanism since the invariant framing doesn't directly apply — there's no positive/negative pair, only "predict the masked thing." Supervised experiments name by what they're testing (data scaling, baseline control, etc.).

## 2026-05: Single-Pass BAM → Labeled Memmap for Tissue Classification

**Motivation:** The tissue-provenance classification task needs windowed reads with per-read tissue labels. The existing `yoran.zarr` does not store `read.query_name`, so it cannot be joined to `yoran_read_labels.txt` post-hoc without a second BAM pass to recover names. Any drift between the two passes would silently shift labels relative to data and produce plausible-looking but wrong accuracy. For a feasibility study, that's the worst possible failure mode.

**Method:** `scripts/bam_to_labeled_memmap.py` walks the BAM once and writes data shards, a labels sidecar, and a `manifest.parquet` in the same iteration loop. Label assignment, kinetics extraction, crop, and shard write all happen at the same point in the same loop, so the row at `(shard_idx, row_idx)` and the label at `labels_NNNNN.npy[row_idx]` are produced together. Misattribution is structurally impossible. Per-tissue read names are pre-sampled from the labels file (not from the BAM) so the cap is unbiased w.r.t. BAM read order.

**Justification:** Skips a 1.6 TB intermediate zarr we don't need for this task and removes the names-out-of-sync silent-failure mode. The manifest is the load-bearing artifact for tests, downstream cell-level splitting, and per-tissue diagnostics; keeping it as a separate parquet (rather than baking labels into shard filenames or the data tensor) lets the same memmap be split read-level OR held-out-cell without rebuilding.

**Manifest as the single source of truth for labels.** An earlier revision also wrote a `labels_NNNNN.npy` sidecar parallel to each data shard, carrying `(tissue_id, cell_id)` per row. That redundancy was dropped: the manifest already has those columns plus `read_name`, `cell_str`, `tissue_str`, `crop_start`, and `read_length`, so a separate sidecar adds maintenance burden without adding information. `TissueMemmapDataset` (in `smrt_foundation/dataset.py`) reads the manifest at construction time, applies an optional polars `filter_expr` for train/val splitting, and indexes shard slices on demand.

## 2026-05: Raw uint8 on Disk; Online Normalization in the Dataloader

**Motivation:** Different normalization schemes (per-read MAD on the cropped window, per-batch z-score, no normalization, learned normalization) each have different transfer-time properties. Baking one scheme into the persisted artifact (as `zarr_to_memmap_instanceNorm.py` does) locks future experiments into that one choice.

**Method:** `bam_to_labeled_memmap.py` writes `dtype=uint8` shards holding the raw BAM tag values (sequence tokens, fi/fp/ri/rp, optional sm/sx) with no log1p or MAD transform. `schema.json` records `"normalize": "none"`. The dataset class is responsible for casting to float and applying whatever normalization the experiment wants.

**Justification:** Storage halves vs float16, IO is faster, and normalization choice becomes an experimental knob rather than a one-way commitment. Matches the precedent already in `bam_to_zarr.py` and `zarr_to_methyl_memmap_v2.py`, which store raw bytes; the older `zarr_to_memmap_instanceNorm.py` is a legacy pattern not propagated forward.

## 2026-05: Forward + Forward-Aligned-Reverse Kinetics in Each Window

**Motivation:** PacBio CCS reads have kinetics from two passes of the polymerase: `fi/fp` from the forward strand and `ri/rp` from the reverse. Both carry independent information about base modifications and polymerase damage. The methyl-classification path (v2 pipeline) already uses both via separate forward and reverse views per CpG window; for tissue classification a single per-read window is enough, but discarding half the kinetics signal would be a needless loss.

**Method:** Each output row of `bam_to_labeled_memmap.py` carries both kinetics views aligned to the same forward-strand position. Channels are `[seq, fi, fp, ri, rp, *optional_tags, mask]`. PacBio stores `ri[i] / rp[i]` in reverse order (i.e., `ri[i]` is the kinetics measured at forward-strand position `L-1-i`); we reverse the per-read array (`ri[::-1].copy()`) before slicing so column `j` of the output row corresponds to forward-strand position `crop_start + j` for all four kinetics columns.

**Justification:** Aligning all kinetics to a single forward-strand frame lets a position-aware encoder consume both signals at the same coordinate without a second view, and it sidesteps the v1 archived-script bug that flipped only the sequence and kinetics jointly with `np.flip` on the whole read (which silently misaligned `ri/rp` against the reverse-complemented sequence). Doing the alignment via explicit reverse-indexing matches `zarr_to_methyl_memmap_v2.py:136-139` and keeps the cropping operation a single contiguous slice across all columns.

## 2026-05: Random Crop at Build Time, Determinstic per Seed

**Motivation:** PacBio HiFi reads are typically 10-25 kb. A `context=4096` window covers a small fraction of any one read. Choosing which crop to keep on disk has consequences: deterministic center-crop biases away from start-of-read polymerase warmup; multiple non-overlapping crops per read inflate effective sample size and risk train/val leakage at the read level; random crop at training time precludes a fixed-shape memmap.

**Method:** One random crop per accepted read, drawn at build time from a single `np.random.RandomState(seed + 1)` advancing once per accepted read in BAM iteration order. A separate `RandomState(seed)` drives per-tissue read-name sampling so changing `--tissues` or `--max_reads_per_tissue` doesn't shift crop draws via shared RNG state. The crop start is recorded in `manifest.parquet` so tests can reconstruct the exact slice from the BAM.

**Justification:** Matches the existing pattern in `zarr_to_methyl_memmap_v2.py:207`. Avoids `hash(read.query_name)` for per-read seeding because Python's `hash()` is randomized across processes via `PYTHONHASHSEED` and would silently break determinism. BAM iteration order is itself deterministic, so a single advancing RNG suffices and the dataset is fully reproducible from `--seed`.

## 2026-05-12: Four-Way CpG Partition for Clean SSL → FT Comparison

**Motivation:** The in-training SSL probe and the supervised FT validation read the same files. Probe-eval and FT-val both point at `data/01_processed/val_sets/cpg_pos_v2.memmap/val` and `cpg_neg_v2.memmap/val`. The train/val split is deterministic (seed=42, 80/20, set in `scripts/zarr_to_methyl_memmap_v2.py:155-246`), so the same 20% of CpG-windowed reads is the probe target on every SSL checkpoint and the validation target on every FT run. `scripts/utils/select_best_ssl_checkpoint.py:187-204` then picks the SSL checkpoint with the highest smoothed `probe_top1` on that partition, and FT initializes from it. The val_top1 that FT reports is then a number measured on reads the SSL selection had already optimized for. This is implicit val-set selection across roughly 100 checkpoints per SSL run, and the resulting FT numbers are biased upward on the SSL-init arm. The random-init FT arm does not get this advantage, so the SSL → FT lift we have been reporting is asymmetrically inflated.

The bias is not catastrophic in magnitude. For N=100 checkpoints and probe noise around 0.5-1pp top-1, the expected over-selection from picking the max is in the 1-3pp range. But that is roughly the size of the LP→FT gap that supervised_53 is built to study, so the puzzle of "ssl_58 has a small LP→FT gap at scale" may partly be an artifact of this leakage rather than a real phenomenon about top-layer reconstruction-saturation. F1-F4 cannot cleanly resolve their target hypotheses while the comparison metric is biased.

There is a second, distinct selection step that also needs an unbiased partition: the FT-recipe comparison. supervised_53 runs four treatments (midlayer, lpft_lldr, decoder_init, recipe_match), and within each SSL experiment we want to pick the best treatment. If we use one held-out partition to do both the SSL ckpt selection *and* the FT recipe selection, then any cross-experiment comparison reads from a partition that has been selected on twice. The clean structure is one held-out partition per selection step, plus one for the final cross-experiment comparison.

**Probe mechanics for reference.** The probe is a single `nn.Linear(d_model, 1)` trained with Adam (lr=3e-3, 3 epochs, BCE-with-logits) on center-latent-pooled features from the frozen encoder; `scripts/experiments/ssl_58_autoencoder_grid/_shared_train.py:208-224`. It already has *separate* config-key pairs for probe-train (`probe_pos_train` / `probe_neg_train`) and probe-eval (`probe_pos_val` / `probe_neg_val`), so the four-way refit needs no changes to the probe's internals; it is purely a data-layout change.

**Method:** Regenerate the CpG memmaps with a four-way partition rather than the existing two-way.

- Extend `scripts/zarr_to_methyl_memmap_v2.py` to support a four-way split: `train` / `val1` / `val2` / `val3`, with distinct seeds per partition so that adding val2/val3 does not perturb the existing val1 (= existing val) reads by construction. Sketch sizing: `train` ~ 65-70%, `val1` ~ 10%, `val2` ~ 10%, `val3` ~ 10-15%. Existing `cpg_{pos,neg}_v2.memmap/val` is renamed in place to `cpg_{pos,neg}_v2.memmap/val1` (zero-cost rename), and the new `val2/` and `val3/` directories are carved out of the existing `train/`.
- Roles per partition:

  | Partition | Used by | Role |
  |---|---|---|
  | `train` | FT encoder+head training, probe-head training | training data |
  | `val1` | probe-eval, `select_best_ssl_checkpoint.py` | SSL checkpoint selection |
  | `val2` | FT in-training tracking, FT recipe selection within an experiment (e.g., which of F1-F4 wins for a given SSL ckpt) | within-experiment FT selection |
  | `val3` | cross-experiment final reporting, one shot per question | the only number that appears in the experiment log / README |

- Config plumbing on the SSL side: `probe_pos_train` / `probe_neg_train` point at new `train`; `probe_pos_val` / `probe_neg_val` point at `val1`. This is the same code path the probe already runs; only the directory paths change. SSL probe history from past runs remains comparable.
- Config plumbing on the supervised side: new keys `pos_data_train` / `neg_data_train` point at new `train`; existing `pos_data_val` / `neg_data_val` point at `val2`; new keys `pos_data_test` / `neg_data_test` point at `val3`. `scripts/ds_grid_v3.py:442-444` already constructs `val_ds` from the val keys; add a parallel `test_ds` construction and a single end-of-training forward pass that writes `test_top1` / `test_loss` to the run's metrics output.
- Reporting discipline: `test_top1` is the only number that gets read into the experiment log, the README, or any cross-experiment summary. `val_top1` (now measured on val2) is for picking the best step within a single FT run and for comparing recipes inside a single experiment. Probe `val1` accuracy stays as the SSL selection metric. If a future analysis loops over multiple FT recipes and picks "the best one on val3," val3 stops being clean for that report — discipline is to fix the recipe ahead of time or use val2 for the recipe selection, and reserve val3 for one shot.

**Dataflow and why shared `train` between the probe and FT is not contamination.** The four partitions are consumed in a specific temporal sequence per SSL experiment:

1. *SSL pretraining.* The encoder trains on `yoran_raw.memmap`, not on any CpG partition. Encoder checkpoints save every 10k steps to disk.
2. *In-training probe* (called every 10k steps during SSL). For each saved checkpoint, fit a fresh 1-layer linear head on `train` features (3 epochs, frozen encoder, BCE), measure top1 on val1, log `probe_top1` to `probe_history.csv`, throw the head away. The encoder on disk is byte-identical before and after.
3. *SSL checkpoint selection.* Across the roughly 100 checkpoints in an SSL run, `select_best_ssl_checkpoint.py` picks the step with the highest smoothed `probe_top1`. This is the only place val1 enters any selection decision.
4. *FT.* Load the selected encoder, initialize a new head, train the full encoder + head on `train`. Track in-training metrics on val2 every N steps. Save FT checkpoints.
5. *FT recipe selection* (across F1-F4 within one supervised_53 experiment). Whichever treatment scores highest on val2 is the winner for that SSL experiment.
6. *Cross-experiment final report.* The winning (SSL ckpt × FT recipe) configurations across SSL experiments are evaluated one shot each on val3. These are the only numbers that appear in the experiment log or README.

A natural concern looking at this flow: the probe head fits on `train` in step 2, and FT also trains on `train` in step 4. Both consume the same data. Is that a contamination path into the val3 number reported in step 6?

No. The two uses of `train` are structurally disconnected. The probe head's weights are discarded after each call, and the encoder is frozen during the probe (`with torch.no_grad()` in `_shared_train.py:218`), so probe gradients never touch the encoder. The checkpoint loaded in step 4 is byte-for-byte what step 1 produced. Whatever the probe did with `train` does not propagate into the model FT trains.

What the probe does propagate forward is a single scalar per checkpoint: `probe_top1` on val1, used in step 3 for selection. Both `probe_top1` (a function of train→val1 generalization) and the eventual `FT_val3` (a function of train→val3 generalization) are noisy estimates of the same underlying quantity, namely how well does this encoder's features support a CpG classifier trained on `train`. Because val1 and val3 are random IID subsets of the same population, the encoder that maximizes `probe_top1` on val1 is approximately the encoder that maximizes the underlying quantity, plus selection noise. The `FT_val3` measurement is on a different sample, so the selection inflation regresses to the mean. The 1-3pp bias from max-of-N selection lives on val1, not on val3.

A concrete way to feel this: imagine running the whole pipeline twice with the same `train` reads and the same SSL checkpoints, but different RNG seeds for the val1/val2/val3 carving. The checkpoint that wins `probe_top1` might differ between the two seeds (different selection noise drawing different "winners"), but the `FT_val3` numbers averaged across the two runs converge to the same underlying value. That is the property "val3 is unbiased" cashes out as.

Where shared `train` does matter is for the external validity of the result, not its leakage status. If `train`, val1, val2, and val3 are all carved from a single CpG corpus (one sequencing batch, one tissue, one instrument), every metric we measure is conditioned on that distribution. We have no answer to "how does the model generalize to data drawn from a different distribution." That is a real limitation, but it is not selection contamination, and adding a separate `probe_train` partition would not improve it. Only acquiring genuinely held-out external data would.

**Backwards compatibility.** Full backwards compat is not achievable because old `train` ∪ old `val` is the entire dataset, so any new partition is drawn from one of the two. Choosing existing val as the new val1 (zero role change for the probe) and carving val2/val3 out of existing train preserves the SSL probe history exactly and keeps `val3` disjoint from old FT *training* data. But old FT models were initialized from SSL checkpoints selected on the old probe-val set (which is the new val1), and old probe-val is a strict superset of new val3 reads' surrounding population, so old runs' val3 numbers are still SSL-selection-contaminated by the ~1-3pp described above. Practical implication:

- Past SSL probe trajectories (the `probe_history.csv` files across the ssl_58 grid) are byte-identical valid under the new scheme; the partition they were measured on is now named val1.
- Past FT runs (supervised_27, supervised_50, supervised_51) cannot produce truly unbiased numbers under the new scheme without re-training. Their existing val_top1 numbers (now measured on what is val2 + val3 reads combined) remain useful as upper bounds.
- supervised_53 (F1-F4) was built but had not yet submitted at the time the leakage was found, so it can be the first wave to produce unbiased val3 numbers.

**Justification:** The four-way partition is the minimum that supports the two distinct selection steps already in the pipeline (SSL ckpt selection on val1; FT recipe selection on val2) plus one held-out partition for the cross-experiment report (val3). Three partitions are not enough: pooling SSL ckpt selection and FT recipe selection on the same val partition reproduces the leakage one level up. Five would be overkill: there is no third selection step inside a single FT run that would need its own protected partition; in-training step selection within FT is what val2 already covers.

The cleanest property of this scheme is that the probe path needs no code change. Its internal train/eval separation already exists; we are only relabeling the directory it points at on the eval side. The behavioral change is entirely on the supervised side, where FT grows a second eval pass (`test_top1` on val3) at the end of training.

Compute cost is trivial. Regenerating the v2 memmaps is a few hours of single-node CPU work. The final `test_top1` eval at FT time is a single forward pass over roughly 10% of the val-equivalent data. No GPU training is repeated.

## 2026-05-19: Four-Way Partition Implementation — Equal Vals, Reuse v2 Path

**Decision:** The four-way partition (2026-05-12) is implemented in `scripts/zarr_to_methyl_memmap_v2.py` with two deviations from the original sketch.

1. **70 / 10 / 10 / 10** (equal val partitions), not 60 / 20 / 10 / 10. The original sketch reused the existing v2 `val/` (20%) as val1 so past probe trajectories would be measured on bit-identical reads. Equal vals sacrifice that bit-identity in exchange for ~12.5% more FT training data. At ~220M total CpG sites the probe-metric standard error at val1 = 10% (~22M sites) is `sqrt(0.5·0.5/22e6) ≈ 1e-4`, so the larger val1 was costing data without improving selection quality — the 1-3pp `argmax_k(probe_top1)` inflation it was meant to dampen is driven by inter-checkpoint training-stochasticity variance, not by val sampling noise at this N.

2. **Regenerate at the canonical `cpg_{pos,neg}_v2.memmap/` path; manually rename the old 2-way data to `cpg_{pos,neg}_v2_deprecated.memmap/` before regeneration.** An earlier draft proposed a fresh `cpg_{pos,neg}_v3.memmap/` path and a script-and-function rename to drop the `_v2` suffix entirely. Both were reverted in favor of the smaller diff. Reasons: (a) the gwf CONFIG registry and the active experiment configs already pin the `_v2` path string, so a v3 bump (or a rename to no-suffix) would cascade into multi-file path edits at the same time as the script's behavioral change, conflating two diffs that should be reversible separately; (b) the disambiguator that actually matters going forward is the new `schema.json` `splits` block (`scheme: "permutation_v1"`), not the filename — any future layout change can bump `scheme` in place without filename churn; (c) the `_v2` name in 8+ existing methodology log entries (2026-02 through 2026-05-12) is the unambiguous identifier of "the active CpG methyl pipeline at that time," and a rename would erode those breadcrumbs without replacing them with anything load-bearing. Cost: `cpg_*_v2.memmap/` now denotes two distinct layouts depending on time (pre-2026-05-19: 2-way; post: 4-way); the `schema.json` `splits.scheme` field is the disambiguator if a future loader ever needs to handle both, and on disk the `_deprecated` suffix on the old directories is the human-visible signal.

**Split scheme on disk:** single `np.random.RandomState(42).permutation(N_reads)`, then contiguous slice assignment `[0:0.1N]→val1`, `[0.1N:0.2N]→val2`, `[0.2N:0.3N]→val3`, `[0.3N:]→train`. Disjoint and exhaustive by construction (the four slices partition `range(N)` exactly when `val{1,2,3}_pct = 0.10`). The schema.json written into each split dir carries a `splits` block (`train_pct`, `val{1,2,3}_pct`, `seed`, `scheme: "permutation_v1"`) so the on-disk layout is self-documenting and any future re-split scheme can be distinguished by the `scheme` string without filename archaeology.

**Justification for the 70% train choice over 65%:** at this corpus scale FT runs are step- or epoch-limited rather than data-limited (the largest supervised runs use `ds_limit` well below the 80% old-train budget), so the 5pp more training data is not load-bearing for absolute accuracy. The reason to pick 70 over 65 anyway is symmetry — three equally-sized val partitions are interchangeable in role-assignment if a future analysis needs to rotate them (e.g., swap val2 ↔ val3 to spot-check whether a single FT-recipe winner is val2-specific). A 65/15/10/10 layout would not support that without re-balancing on disk.

**Status of the gwf wiring:** `workflow.py` `memmap_cpg_conversion` (lines 395-420) now passes `--val1_pct / --val2_pct / --val3_pct` (defaults 0.10 each) instead of the removed `--val_pct`. The CONFIG entries for `cpg_pos` / `cpg_neg` keep their existing `memmap` paths (`data/01_processed/val_sets/cpg_{pos,neg}_v2.memmap`). Operator step before triggering the DAG: rename the existing `cpg_{pos,neg}_v2.memmap/` directories on disk to `cpg_{pos,neg}_v2_deprecated.memmap/`. After that, `gwf run` on the `cpg_*_to_memmap` targets will regenerate the four-way layout at the canonical path.

**Follow-ups deferred:**
- Experiment configs (ssl_58 / ssl_59 / ssl_60 / supervised_53): `probe_pos_val` / `probe_neg_val` and `pos_data_val` / `neg_data_val` keys need their subdirectory swapped from `…/val` to `…/val1` and `…/val2` respectively, plus new `pos_data_test` / `neg_data_test` keys at `…/val3` on the supervised side. The memmap path string itself (`cpg_pos_v2.memmap`) does not change — only the trailing subdirectory. Done at launch time, one experiment at a time.
- `scripts/ds_grid_v3.py:431-445` — needs a parallel `test_ds` construction and an end-of-training forward pass that emits `test_top1` / `test_loss` to the run's metrics output. This is the only structural change on the supervised side; the existing `val_ds` path is unchanged.
- `tests/test_zarr_to_methyl_memmap_v2.py` — currently hardcodes `train`/`val` and asserts the 80/20 fraction. Tests fail under the new layout until reworked; the test fixtures also still pass the removed `val_pct=0.2` kwarg to the now-renamed signature, which is a separate import/call fix. Reworked after a first regeneration on Gefion confirms the four-way output is well-formed.
