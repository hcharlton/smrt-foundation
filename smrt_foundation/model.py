import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import math

class PositionalEncoding(nn.Module):
    """Standard positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class CPCDNA(nn.Module):
    """
    CPC on PacBio SMRT data.
    
    This model uses a single Causal Transformer to generate context vectors c_t.
    It then tries to predict future context vectors c_{t+k} using c_t.
    The "latent space" is the output space of the Causal Transformer.
    """
    def __init__(self, 
                 d_model = 128, 
                 nhead = 4, 
                 num_layers= 4,
                 dim_feedforward = 512,
                 predict_steps = 5,
                 vocab_size = 5, 
                 kinetics_dim = 4, 
                 dropout = 0.1
                 ):
        super().__init__()
        self.d_model = d_model
        self.predict_steps = predict_steps

        # 1. Input Embeddings
        self.seq_embedding = nn.Embedding(vocab_size, d_model, padding_idx=4)
        # We project kinetics to d_model. We'll add this to the seq embedding.
        self.kinetics_proj = nn.Linear(kinetics_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Autoregressive Transformer Encoder (Causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 4. Projection Heads (W_k in the CPC paper)
        # One simple linear predictor for each step k
        self.predictors = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(predict_steps)]
        )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a causal mask for the transformer."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, seq_ids, kinetics):
        """
        Args:
            seq_ids: [B, L] tensor of nucleotide indices (0-3)
            kinetics: [B, L, 4] or [B, L, 0] tensor of kinetics
            
        Returns:
            c: [B, L, d_model] tensor of context vectors (c_t)
            predictions: List of K tensors, each [B, L, d_model],
                         representing p_{t,k} = W_k(c_t)
        """
        B, L = seq_ids.shape
        device = seq_ids.device

        # 1. Project inputs
        seq_emb = self.seq_embedding(seq_ids) # [B, L, d_model]
        
        # Add kinetics if they exist
        if kinetics.shape[-1] > 0:
            # Normalize kinetics (0-255 -> 0-1)
            kin_norm = kinetics / 255.0
            kin_emb = self.kinetics_proj(kin_norm) # [B, L, d_model]
            x = seq_emb + kin_emb
        else:
            x = seq_emb
            
        # 2. Add positional encoding
        x_pos = self.pos_encoder(x) # [B, L, d_model]
        
        # 3. Pass through Causal Transformer
        mask = self._generate_square_subsequent_mask(L).to(device)
        c = self.transformer(x_pos, mask) # [B, L, d_model]
        
        # 4. Generate predictions for k steps ahead
        predictions = []
        for predictor in self.predictors:
            predictions.append(predictor(c))
            
        # c = context vectors c_t
        # predictions = list of p_{t,k}
        return c, predictions



MODEL_REGISTRY = {
    'CPCDNA': CPCDNA,
}