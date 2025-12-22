import torch
import torch.nn.functional as F

def cpc_loss(c_target, predictions):
    """
    Computes the InfoNCE loss for CPC.
    
    Args:
        c_target: [B, L, D] tensor of *detached* target vectors (z_t, which is c_t)
        predictions: List of K tensors [B, L, D], (p_{t,k} = W_k(c_{t}))
    """
    total_loss = 0
    B, L, D = c_target.shape
    K = len(predictions)

    for k in range(1, K + 1):
        k_idx = k - 1
        pred_k = predictions[k_idx] # [B, L, D]
        
        # predict up to L-k steps ahead
        L_valid = L - k
        if L_valid <= 0:
            continue
            
        # Get contexts c_t for t = 0 to L-k-1
        # these are the inputs for the linear predictors for the k steps
        c_t = pred_k[:, :L_valid, :] # [B, L_valid, D]
        
        # Get targets z_{t+k} for t = 0 to L-k-1 (which is z_k to z_{L-1})
        z_target_k = c_target[:, k:, :] # [B, L_valid, D]

        # Reshape for matrix multiplication
        # [B * L_valid, D]
        c_t_flat = c_t.reshape(B * L_valid, D)
        z_target_k_flat = z_target_k.reshape(B * L_valid, D)

        # get dot product similarities
        # contrast c_t's prediction against all  z_j in the batch
        scores = torch.matmul(c_t_flat, z_target_k_flat.T)
        
        # the positive sample is the one at the same (b, t) index
        # which corresponds to the diagonal in the scores  matrix
        labels = torch.arange(B * L_valid, device=scores.device)
        
        # get ce loss
        loss = F.cross_entropy(scores, labels)
        total_loss += loss

    if K == 0:
        return torch.tensor(0.0, device=c_target.device)
        
    return total_loss / K
