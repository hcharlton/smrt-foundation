# Diagnostics manual

Reference for the scalar diagnostics logged across this project's training scripts. Manual-style: each section stands on its own and can be flipped to without reading sequentially. Add new sections here as new diagnostics enter the codebase.

## Contents

- [Embedding diagnostics: z_norm and z_std](#embedding-diagnostics-z_norm-and-z_std)

---

## Embedding diagnostics: z_norm and z_std

Two scalars logged from every contrastive SSL run in this project: `embed_z_norm` and `embed_z_std`. Both summarize the projection-head output `c_proj` (shape `[B, T, d_proj]` for input-mask runs, `[B, d_proj]` for SimCLR runs) before `F.normalize` enters the loss. Computation lives in `scripts/experiments/ssl_57_inputmask_grid_lnhead/_shared_train.py:555-561` (ssl_55 and ssl_56 have the same code). Treat them as stability sensors, not optimization targets.

### Definitions

Flatten `c_proj` to a 2D matrix `Z` of shape `[N, d_proj]` (N = B for SimCLR, N = B*T for input-mask). Then:

```
z_norm = mean over rows of  ||z_n||_2
z_std  = mean over columns of  std over rows ( z_{n,c} )
```

`z_norm` is the average length of an embedding vector. `z_std` is the average per-channel spread across the cloud of embeddings.

### Why these and not something else

Treat `Z` as N points in `d_proj`-dimensional Euclidean space. Any embedding cloud has two classical pathologies, both well-documented in the SSL literature:

1. **Magnitude runaway.** Embeddings drift to high-magnitude regions. `||z_n||` grows over training without bound. Wang and Isola 2020 frame this as the "uniformity" failure.
2. **Collapse.** All points end up at one location (full collapse) or on a low-dimensional subspace (dimensional collapse, Jing et al. 2022). Samples become indistinguishable in the embedding space.

`z_norm` catches (1). `z_std` catches (2), at the level of `trace(cov)`. Both are coarse but cheap to compute every step.

### Why the loss cares about magnitude even though it L2-normalizes

The loss does `F.normalize(z, dim=-1)` first, so cosine similarity only sees direction. So why monitor `z_norm` if the loss discards it? Because the gradient flowing back through `F.normalize` is gated by `1 / ||z||`. The derivation below makes this exact, then the next subsection shows how that factor closes a self-reinforcing loop on the projection-head weights.

#### Derivation of the Jacobian

Let `f(z) = z / ||z||` with `z` a column vector in `R^d`. The Jacobian `J = df/dz` is a `d x d` matrix with entries `J_{ij} = partial f_i / partial z_j`. Set `r = ||z||_2 = (sum_k z_k^2)^{1/2}` for compactness. Then

```
partial r / partial z_j = z_j / r
```

Apply the quotient rule componentwise:

```
partial f_i / partial z_j = partial (z_i / r) / partial z_j
                          = delta_ij / r - z_i (z_j / r) / r^2
                          = (1 / r) (delta_ij - z_i z_j / r^2)
```

Letting `hat z = z / r` denote the unit vector, this is `partial f_i / partial z_j = (1/r) (delta_ij - hat z_i hat z_j)`. In matrix form:

```
J = (1 / ||z||) * (I - hat z hat z^T)
```

The factor `(I - hat z hat z^T)` is a rank-`(d-1)` projector onto the hyperplane tangent to the unit sphere at `hat z`. Apply `J` to any vector `v`:

- The component of `v` along `hat z` (the radial direction) is annihilated. Moving along the ray through the origin doesn't change the unit vector.
- The component of `v` perpendicular to `hat z` (tangential to the sphere at `hat z`) is preserved and scaled by `1 / ||z||`.

So the spectral norm of `J` is exactly `1 / ||z||`. The whole map shrinks linearly as `||z||` grows.

#### The attractor

Backprop chains the upstream gradient through `J^T`:

```
dL / dz = J^T * (dL / d hat z) = (1 / ||z||) * (I - hat z hat z^T) * (dL / d hat z)
```

`dL/dz` inherits the `1 / ||z||` factor. Larger `||z||` produces smaller gradient on the projection-head weights that produced `z`. AdamW takes a smaller step. The next batch produces a similarly large `||z||` because the network barely moved. The loop closes.

Concretely this makes large `||z||` a stable fixed point of the optimization. The trap is hard to enter (the network has to drift to large `||z||` in the first place) but, once in, hard to leave. Two routes in:

- A handful of outlier batches early in training push `||z||` up; the resulting gradient suppression then prevents corrective updates.
- Wider projection heads have larger output magnitudes by default (the linear layer's output variance scales with the input dim under standard inits), so larger sizes drift in faster. The d=768 case in `ssl_54_simclr_grid_yoran/size_d768_L8` hit this: norm walked from O(10) to 3.6e6 over the first ~150k steps, gradients went non-finite, and the model was locked in for the rest of training. Full post-mortem at `docs/negative_results.md` 2026-04-27.

What breaks the trap: pin `||z||` analytically with LayerNorm at the head's output, so the magnitude can never grow regardless of what the optimizer wants. ssl_55's `MLPProjectionHeadLN` onward, including ssl_57's `Smrt2VecInputMaskLN` with the same final-LN structure.

`z_norm` is the early-warning sensor for the trap. If it walks upward steadily during training, the loop is engaging. If it sits flat near `sqrt(d_proj)` (next subsection), the LN is doing its job and the trap is unreachable.

### Why z_std catches collapse

If `z_std` drops to zero, every row of `Z` is identical. Every L2-normalized row points the same direction. Every cosine similarity is 1. Every InfoNCE or NT-Xent logit is identical. Cross-entropy is exactly `log(2N)` and constant. No gradient signal anywhere, no learning. This is what "collapse" means concretely.

Partial collapse is the more common failure: a fraction of channels die (their per-row variance goes to zero), the rest carry signal. `z_std` drops in proportion to the dead fraction, which is why it is a coarse but useful metric. A run with k healthy channels out of d_proj gives `z_std approx (k/d_proj) * sigma_healthy`.

### The LayerNorm head bounds z_norm by construction

The ssl_55 and ssl_57 projection heads end in a final `LayerNorm` (the `Linear -> LN -> GELU -> Linear -> LN` structure in `SimCLRSmrtLN` and `Smrt2VecInputMaskLN`). LN normalizes each token to mean 0, unit variance across the d_proj channels, then applies a learned affine `(gamma, beta)`. After LN with affine off, `||z|| = sqrt(d_proj)` exactly: d_proj unit-variance entries summed in quadrature.

With the learned affine `gamma` in play, the stable value is `z_norm approx mean(gamma) * sqrt(d_proj)`. The empirical mean of `gamma` is task-dependent and generally lives in `[0.7, 1.3]`. Observed at ssl_57's step 300k (masked-prediction, `gamma` mean ~0.81):

| size  | sqrt(d_proj) | observed z_norm | ratio |
|-------|--------------|-----------------|-------|
| d128  | 11.31        | 9.01            | 0.80  |
| d256  | 16.00        | 13.21           | 0.83  |
| d512  | 22.63        | 18.59           | 0.82  |
| d768  | 27.71        | 22.43           | 0.81  |

For comparison, ssl_55 (SimCLR contrastive) settles around `gamma` mean ~1.1 to ~1.24 — the contrastive objective pushes the affine *upward* despite the LN. Both objectives stay within an order of magnitude of `sqrt(d_proj)`, which is what the LN guarantees; the exact `gamma` is what the loss prefers.

Magnitude is pinned analytically. The optimizer cannot grow embeddings beyond what `gamma` allows even if the runaway feedback loop tries to engage. ssl_55 added the LN head specifically to retire this failure mode and ssl_57 inherits it; the curves are flat from step 0 onward.

`z_std` is harder to predict analytically. LN normalizes within a single token; `z_std` measures variability across tokens, which depends on how diverse the upstream representations are. Empirically it lives in [0.7, 1.0] for healthy runs and decays slightly during training as the head specializes. ssl_57 trends from ~0.95 to ~0.79 over 300k steps across all four sizes and stabilizes there.

### Reading the curves

| pattern | reading |
|---------|---------|
| `z_norm` flat at `mean(gamma) * sqrt(d_proj)` (typically within `[0.7, 1.3] * sqrt(d_proj)`) | LN head healthy. Magnitude bounded, no runaway risk. |
| `z_norm` walking upward without bound | Magnitude runaway. Check whether LN was bypassed somewhere or affine grew unrealistically. |
| `z_std` in [0.7, 1.0], slow decay across training | Healthy specialization. Encoder is using its capacity. |
| `z_std` collapsing toward zero | Dimensional collapse engaging. Embeddings are converging to a single direction; the contrastive loss is about to lose its signal. |
| `z_std` stable but per-channel histogram shows a dead fraction | Partial collapse. The mean hides it; check the per-channel std distribution explicitly. |
| Both flat AND `probe_top1` climbing | Best case. Encoder is learning useful features without straining the projection head. ssl_57 d256-d768 sits here. |
| Both flat AND `probe_top1` declining | Stability is fine but transfer is failing. The training task is learning something irrelevant to the probe. ssl_56 sits here. |

### Limits of z_std

`z_std` is a single number summarizing one feature of the covariance spectrum. It catches gross collapse but misses two patterns:

- A fraction of channels die while the survivors carry larger variance. `z_std` (mean) can stay constant while effective rank halves.
- The cloud lives on a curved low-dimensional manifold inside `R^{d_proj}`. The per-axis std does not see this. The manifold could be 1-D and `z_std` would look healthy if the axes happen to align with the manifold's tangent space.

The cleaner diagnostic is the eigenvalue spectrum of the empirical covariance `Sigma = Z^T Z / N - mu mu^T`. Two derived scalars worth adding when this becomes load-bearing:

- **Effective rank by participation ratio:** `(sum lambda_i)^2 / sum(lambda_i^2)`. Equals d_proj when all eigenvalues are equal, drops sharply when the spectrum is heavy-tailed.
- **Top-k variance fraction:** `sum_{i=1}^k lambda_i / sum lambda_i` for `k = d_proj // 4`. Near 1 means most variance is in a small subspace.

Neither is computed today. Both are cheap to add to the probe step (one snapshot per probe is enough, no need for every training step).

### Where these came from

- `ssl_54_simclr_grid_yoran/size_d768_L8` catastrophic failure: norm runaway to 3.6e6, gradients non-finite, model corrupted. Motivated the LN projection head. See `docs/negative_results.md` 2026-04-27.
- `ssl_55_simclr_grid_lnhead`: LN head added, ran to completion at d=768 with `z_norm` bounded at ~14. See `docs/experiment_log.md` 2026-04-27.
- `ssl_57_inputmask_grid_lnhead`: same harness, masked-prediction objective. `z_norm` analytically pinned at ~0.8 * sqrt(d_proj), `z_std` stable at ~0.78-0.81 across all four sizes. The metrics now serve as guardrails rather than alarms.

### Reminder: not optimization targets

These are the SSL equivalent of watching gradient norm and weight magnitude during supervised training. We do not optimize them. They tell us whether the optimizer is in a stable regime. The load-bearing transfer signal is `probe_top1` and the downstream fine-tune accuracy. These two scalars exist so we can rule out collapse and runaway cleanly when interpreting the transfer signal.
