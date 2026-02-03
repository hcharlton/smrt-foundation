# smrt_foundation/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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
