import torch
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, total_steps, pct_start, min_lr_ratio = 0.05, num_cycles=0.5):
    num_warmup_steps = int(total_steps * pct_start)

    def lr_lambda(current_step):
        if current_step > total_steps:
            return min_lr_ratio

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
            
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return (1.0 - min_lr_ratio) * cosine_val + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)

def get_noam_schedule(optimizer, d_model, n_warmup_steps):
    """
    Attention is all you need classic
    """
    def lr_lambda(step):
        step += 1 
        return (d_model ** -0.5) * min(step ** -0.5, step * n_warmup_steps ** -1.5)
    
    return LambdaLR(optimizer, lr_lambda)