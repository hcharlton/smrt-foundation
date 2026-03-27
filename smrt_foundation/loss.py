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
