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

class AgInfoNCE3(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, c_proj, targets, mask_idx):
        preds = F.normalize(c_proj[mask_idx], dim=-1)
        truth = F.normalize(targets[mask_idx], dim=-1)
        
        if dist.is_initialized():
            truth_gathered = torch.cat(dist_nn.all_gather(truth), dim=0)
            labels = torch.arange(truth.shape[0], device=truth.device) + (dist.get_rank() * truth.shape[0])
        else:
            truth_gathered = truth
            labels = torch.arange(truth.shape[0], device=truth.device)
            
        logits = torch.mm(preds, truth_gathered.T) / self.temperature
        return self.cross_entropy(logits, labels)




class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if not dist.is_initialized():
            return x
        gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if not dist.is_initialized():
            return grad_output
        grad_input = torch.empty_like(x)
        dist.reduce_scatter_tensor(grad_input, grad_output.contiguous())
        return grad_input

class AgInfoNCE1 (nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, c_proj, targets, mask_idx):
        preds = F.normalize(c_proj[mask_idx], dim=-1)
        truth = F.normalize(targets[mask_idx], dim=-1)
        
        if dist.is_initialized():
            truth_gathered = GatherLayer.apply(truth)
            labels = torch.arange(truth.shape[0], device=truth.device) + (dist.get_rank() * truth.shape[0])
        else:
            truth_gathered = truth
            labels = torch.arange(truth.shape[0], device=truth.device)
            
        logits = torch.mm(preds, truth_gathered.T) / self.temperature
        return self.cross_entropy(logits, labels)


class AgInfoNCE2(nn.Module):
    def __init__(self, temperature=0.1, accelerator=None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.accelerator = accelerator

    def forward(self, c_proj, targets, mask_idx):
        preds = F.normalize(c_proj[mask_idx], dim=-1)
        truth = F.normalize(targets[mask_idx], dim=-1)
        
        if self.accelerator is not None and self.accelerator.state.num_processes > 1:
            truth_gathered = self.accelerator.gather(truth)
            rank = self.accelerator.process_index
            labels = torch.arange(truth.shape[0], device=truth.device) + (rank * truth.shape[0])
        else:
            truth_gathered = truth
            labels = torch.arange(truth.shape[0], device=truth.device)
            
        logits = torch.mm(preds, truth_gathered.T) / self.temperature
        return self.cross_entropy(logits, labels)
