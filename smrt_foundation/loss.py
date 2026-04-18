# smrt_foundation/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as dist_nn

class InfoNCE(nn.Module):
  def __init__(self, temperature=0.1):
    super().__init__()
    self.cross_entropy = nn.CrossEntropyLoss()
    self.temperature = temperature
  def forward(self, c_proj, targets, mask_idx):
    # gather the predictions and truth vectors
    preds = c_proj[mask_idx]
    truth = targets[mask_idx]
    # normalize for cosine similarity
    # last dim (embedding dim)
    preds = F.normalize(preds, dim=-1)
    truth = F.normalize(truth, dim=-1)
    # print(truth.shape,preds.shape)
    logits = torch.mm(preds, truth.permute(1,0)) / self.temperature
    labels = torch.arange(truth.shape[0], device=truth.device)
    loss = self.cross_entropy(logits, labels)
    return loss

class AgInfoNCE(nn.Module):
    def __init__(self, temperature=0.1, max_negatives=None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.max_negatives = max_negatives

    def forward(self, c_proj, targets, mask_idx):
        preds = F.normalize(c_proj[mask_idx], dim=-1)
        truth = F.normalize(targets[mask_idx], dim=-1)

        # Subsample masked positions to cap similarity matrix size.
        # With input masking + CNN downsampling, ~99% of latents are masked,
        # making the full [N_local, N_total] matrix too large for GPU memory.
        if self.max_negatives and preds.shape[0] > self.max_negatives:
            idx = torch.randperm(preds.shape[0], device=preds.device)[:self.max_negatives]
            preds = preds[idx]
            truth = truth[idx]

        if dist.is_initialized():
            truth_gathered = torch.cat(dist_nn.all_gather(truth), dim=0)
            labels = torch.arange(truth.shape[0], device=truth.device) + (dist.get_rank() * truth.shape[0])
        else:
            truth_gathered = truth
            labels = torch.arange(truth.shape[0], device=truth.device)

        logits = torch.mm(preds, truth_gathered.T) / self.temperature
        return self.cross_entropy(logits, labels)

class MaskedReconstructionLoss(nn.Module):
    """MSE loss on masked kinetics positions only."""
    def forward(self, kin_recon, kin_target, mask):
        # kin_recon: [B, T, 2], kin_target: [B, T, 2], mask: [B, T] bool
        return F.mse_loss(kin_recon[mask], kin_target[mask])


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
    with world_size. Each query sees 2 · world_size · B − 2 negatives,
    the SimCLR-canonical count.

    Temperature 0.1 matches SimCLR §5.1's near-optimal setting for
    L2-normalised logits. A previous version of this class supported an
    optional `max_negatives` subsample of the key pool after all-gather;
    that path never ran (the full [2B, 2·world_size·B] matrix is ~8 MB
    at our largest R1 size) and has been archived in
    `archive/ntxent_max_negatives.py` for reference.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = float(temperature)

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

        # Each local query's own slot in the global key pool (to mask out),
        # and its positive — the matching view of the same sample.
        arange = torch.arange(local_B, device=device)
        self_z1 = rank * local_B + arange               # z1-side self
        self_z2 = global_B + rank * local_B + arange    # z2-side self
        pos_z1 = self_z2                                # z1 query's positive is z2 slot
        pos_z2 = self_z1                                # z2 query's positive is z1 slot

        sim = queries @ keys.T / self.temperature

        # Mask out each query's own row in the key pool (self-similarity).
        query_rows = torch.arange(2 * local_B, device=device)
        self_cols = torch.cat([self_z1, self_z2], dim=0)
        sim[query_rows, self_cols] = float('-inf')

        targets = torch.cat([pos_z1, pos_z2], dim=0)
        return self.cross_entropy(sim, targets)
