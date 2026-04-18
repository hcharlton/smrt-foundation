"""Archived pre-cut NTXent with max_negatives subsampling branch.

This module is NOT imported anywhere. It's kept as a readable reference
in case a future training run ever has a global batch large enough that
the [2·local_B, 2·world_size·local_B] similarity matrix exceeds GPU memory.

Why it was removed from the shipping NTXent
  Ported originally from `smrt_foundation.loss.AgInfoNCE`, where the
  subsampling branch is load-bearing: masked-prediction SSL treats every
  CNN-output latent as a query/key pair, so at 99% masking the key pool
  reaches ~1M rows and the sim matrix easily exceeds 10 GB per rank.

  SimCLR-style NT-Xent contributes exactly one projected vector per
  augmented view per sample. For the R1 scoping grid's largest
  configuration (world_size=8, local_B=256, d_proj=128), the key pool is
  only 2·8·256 = 4096 rows × 128 dims = 2 MB, and the similarity matrix
  is [512, 4096] × 4 bytes = 8 MB — three orders of magnitude below any
  memory threshold that `max_negatives` would protect against. Every
  shipped config sets `max_negatives: null`, so the branch is never
  exercised but still pays code-review and test-coverage rent.

When this branch would become worth reviving
  - Similarity matrix per rank exceeds O(10 GB), i.e. the fp32
    similarity tensor is roughly 2·local_B × 2·world_size·local_B × 4
    bytes. At d_proj=128 projections, that crosses 10 GB when
    local_B · world_size ≳ 50 k — so concretely, local batch 1024 across
    a 64-rank job, or local batch 4096 across 16 ranks. Both are far
    beyond what the SimCLR pretraining plan (up to 8 ranks × 256–512
    local batch) uses.
  - Gradient-accumulation / local batch reduction is the first-line fix
    for OOM at this loss layer. `max_negatives` is strictly a fallback
    once those are also exhausted, since it defeats SimCLR's "bigger
    batch is better" scaling law.

Revival procedure
  1. Copy the class below into `smrt_foundation/loss.py` (overwrite the
     simpler shipping NTXent, or add alongside under a different name).
  2. Re-add `max_negatives: int | None = None` default to the relevant
     config.yaml files and thread it through the train script's DEFAULT
     dict and `NTXent(...)` call.
  3. Add a regression test that exercises the subsampling branch — the
     offset-rebuilding logic is subtle (must_keep slots, new_pos
     mapping, sorted sel ordering for stable indexing) and has no
     coverage outside manual inspection.

For the shipping (simplified) version see `smrt_foundation/loss.py:NTXent`.

------------------------------------------------------------------------

PRE-CUT SOURCE (reference only; unused).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn


class NTXent(nn.Module):
    """Symmetric NT-Xent loss (SimCLR v1).

    Takes two batches of projected embeddings `z1`, `z2` of shape `[B, d]`
    produced from two independent augmented views of the same underlying
    sample. L2-normalises both, computes the 2N x 2N cosine-similarity
    matrix across the concatenated batch, masks the diagonal (self-
    similarity), and applies temperature-scaled cross-entropy with the
    positive for each query being the other view of the same sample.

    With DDP, negatives are drawn from the union of all ranks' local
    batches via differentiable all-gather (same pattern as `AgInfoNCE`).
    Gradients flow only through this rank's local slice of the gathered
    tensors, so the loss remains per-rank while the negative pool grows
    with world_size.

    Temperature 0.1 matches SimCLR §5.1's near-optimal setting for
    L2-normalised logits. `max_negatives` (optional) subsamples the global
    key pool after all-gather to cap the similarity-matrix memory; the
    default of None uses the full pool (2 * world_size * local_B - 2
    negatives per positive), which is the SimCLR-canonical choice.
    """

    def __init__(self, temperature: float = 0.1, max_negatives: int | None = None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = float(temperature)
        self.max_negatives = max_negatives

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        local_B = z1.shape[0]
        device = z1.device

        if dist.is_initialized():
            z1_all = torch.cat(dist_nn.all_gather(z1), dim=0)
            z2_all = torch.cat(dist_nn.all_gather(z2), dim=0)
            rank = dist.get_rank()
        else:
            z1_all = z1
            z2_all = z2
            rank = 0

        global_B = z1_all.shape[0]

        # Queries are this rank's local views; keys are the global pool.
        queries = torch.cat([z1, z2], dim=0)          # [2 * local_B, d]
        keys = torch.cat([z1_all, z2_all], dim=0)     # [2 * global_B, d]

        # Optional subsample of the negative pool (kept rarely — only for very
        # large global batches where the [2*local_B, 2*global_B] matrix blows
        # past GPU memory).
        if self.max_negatives is not None and keys.shape[0] > self.max_negatives:
            # Always keep the slots that hold self and positive for each query.
            offset1_slot = rank * local_B + torch.arange(local_B, device=device)
            offset2_slot = global_B + rank * local_B + torch.arange(local_B, device=device)
            must_keep = torch.cat([offset1_slot, offset2_slot], dim=0)
            remaining = max(self.max_negatives - must_keep.shape[0], 0)
            if remaining > 0:
                all_idx = torch.arange(keys.shape[0], device=device)
                pool_mask = torch.ones(keys.shape[0], dtype=torch.bool, device=device)
                pool_mask[must_keep] = False
                pool = all_idx[pool_mask]
                sample = pool[torch.randperm(pool.shape[0], device=device)[:remaining]]
                sel = torch.cat([must_keep, sample], dim=0)
            else:
                sel = must_keep
            sort_order = torch.argsort(sel)
            sel = sel[sort_order]
            # Rebuild offsets into the subsampled key matrix.
            keys = keys[sel]
            # Map old global indices → new positions.
            new_pos = torch.full((2 * global_B,), -1, dtype=torch.long, device=device)
            new_pos[sel] = torch.arange(sel.shape[0], device=device)
            self_z1 = new_pos[rank * local_B + torch.arange(local_B, device=device)]
            self_z2 = new_pos[global_B + rank * local_B + torch.arange(local_B, device=device)]
            pos_z1 = self_z2
            pos_z2 = self_z1
        else:
            arange = torch.arange(local_B, device=device)
            self_z1 = rank * local_B + arange
            self_z2 = global_B + rank * local_B + arange
            pos_z1 = self_z2
            pos_z2 = self_z1

        sim = queries @ keys.T / self.temperature

        # Mask out each query's own row in the key pool (self-similarity).
        query_rows = torch.arange(2 * local_B, device=device)
        self_cols = torch.cat([self_z1, self_z2], dim=0)
        sim[query_rows, self_cols] = float('-inf')

        targets = torch.cat([pos_z1, pos_z2], dim=0)
        return self.cross_entropy(sim, targets)
