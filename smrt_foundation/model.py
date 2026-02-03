# smrt_foundation/model.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
  """
  Generates positional encodings based on a given sequence length and
  d_model. Follows Vaswani et al (2017).
  """
  def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model) # initialize vector
        position = torch.arange(0, max_len, dtype=torch.bfloat16).unsqueeze(1) # make an index vector
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
      return x + self.pe[:, :x.size(1)]

class MLP(nn.Module):
  """
  Simple MLP for composition in the TransformerBlock
  """
  def __init__(self, d_model, expansion=4):
      super().__init__()
      self.c_fc = nn.Linear(d_model, d_model * expansion)
      self.gelu = nn.GELU() # maybe ReLU would be fine?
      self.c_proj = nn.Linear(d_model * expansion, d_model)

  def forward(self, x):
      x = self.c_fc(x)
      x = self.gelu(x)
      x = self.c_proj(x)
      return x

class SmrtEmbedding(nn.Module):
  """
  Hybrid embeddings for the SMRT data

  Nucleotides are in encoded as float representations of integers in the dataset
  so that they can be used to index into an embedding table

  Kinetics (ipd, pw) come as floats and are embedded with a linear projection

  Issue:
  n_nucleotides is set to 5, but in reality we don't use a padding token in the
  data preprocessing. Instead, the padded sections of the data are set to 0.0
  with the exception of the padding mask channel, which is set to 1.0
  """
  def __init__(self, d_model, n_nucleotides=5, n_continuous=2):
    super().__init__()
    self.nuc_embed = nn.Embedding(n_nucleotides, d_model//2)
    self.kin_embed = nn.Linear(n_continuous, d_model//2, dtype=torch.bfloat16)
    self.layernorm = nn.LayerNorm(d_model)
    self.d_model = d_model
  def forward(self, x_nuc, x_kin, is_padding):
    scale = math.sqrt(self.d_model)
    seq_emb = self.nuc_embed(x_nuc.int())*scale
    kin_emb = self.kin_embed(x_kin)*scale
    x = torch.concat((seq_emb,kin_emb),dim=-1)
    x = self.layernorm(x)
    return x

class BidirectionalSelfAttention(nn.Module):
  """
  Implementation of bidirectional self attention.

  Uses a GPT-style single matrix multiplication for computing QKV

  Forward pass uses the padding mask (provided in the data as the
  last channel) as an attn mask. Pytorch has competing standards on whether
  1 should correspond to "attend" or "ignore". The mask in the data is a
  "padding mask" and so 1 corresponds to "pad" and 0 to "active data".
  """
  def __init__(self, d_model, n_head=4, max_len=4096):
      super().__init__()
      assert d_model % n_head == 0
      self.n_head = n_head
      self.head_dim = d_model // n_head
      # produces qkv, so we output 3*d_model
      self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
      self.c_proj = nn.Linear(d_model, d_model, bias=False)

  def forward(self, x, x_pad, pad_val=1):
      B, T, C = x.size()
      # use one big matmul and split
      qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim)
      qkv = qkv.permute(2, 0, 3, 1, 4) # -> (3, B, n_head, T, head_dim)
      q, k, v = qkv[0], qkv[1], qkv[2] # -> 3 x (B, n_head, T, head_dim)

      # broadcast across the head and query dims
      # given alignment right to left, we need to reshape to match B,H,T,T
      attn_mask = ~x_pad.view(B, 1, 1, T)
      output = F.scaled_dot_product_attention(
          q, k, v,
          attn_mask=attn_mask,
          dropout_p=0.0 if not self.training else 0.05,
          is_causal=False # since we attend to everything outside the att_mask
      )

      output = output.transpose(1, 2).contiguous().view(B, T, C)
      return self.c_proj(output)

class TransformerBlock(nn.Module):
  """
  Ties together BidirectionalSelfAttention, MLP, and LayerNorm to form
  a layerable transformer block

  Issue:
  Layernorm does not accept a pad mask
  https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  """
  def __init__(self, d_model, n_head, max_len):
      super().__init__()
      self.ln1 = nn.LayerNorm(d_model)
      self.attn = BidirectionalSelfAttention(d_model, n_head, max_len)
      self.ln2 = nn.LayerNorm(d_model)
      self.mlp = MLP(d_model)

  def forward(self, x, x_pad): # includes unscaled residuals
      x = x + self.attn(self.ln1(x), x_pad)
      x = x + self.mlp(self.ln2(x))
      return x

class ResBlock(nn.Module):
  """
  Convolutional block with residual connections

  Issue:
  Uses batch norm in between layers. This is dubious. Though it's was
  approrpriate for the purely convolutional 1d methylation classifier, for
  this sequential data it could cause problems
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1):
    super(ResBlock, self).__init__()

    self.padding = (kernel_size - 1) // 2
    self.kernel_size = kernel_size

    self.bn1 = nn.BatchNorm1d(in_channels)
    self.conv1 = nn.Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=self.padding,
                           bias=False)
    self.bn2 = nn.BatchNorm1d(out_channels)
    self.conv2 = nn.Conv1d(in_channels=out_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=1,
                           padding=self.padding,
                           bias=False)

    self.relu = nn.ReLU(inplace=True)
    self.stride = stride
    # projection residual
    if any([in_channels != out_channels, stride != 1]):
      self.residual = nn.Sequential(
          nn.Conv1d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1, stride=stride,
                    bias=False)
          )
    # identity residual
    else:
      self.residual = nn.Sequential()
  def _resize_mask(self, mask, pad_val=1):
    if mask.dtype == torch.bool:
      mask = mask.float()
    if pad_val == 0:
      mask = F.max_pool1d(mask,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          padding=self.padding)
    elif pad_val == 1:
      mask = 1 - F.max_pool1d(1 - mask,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding)
    else:
      raise ValueError("Invalid pad value: Pad value must be 0 or 1")
    return mask.bool()

  def forward(self, x, mask):
    out = self.relu(self.bn1(x))
    out = self.conv1(out)
    out = self.relu(self.bn2(out))
    out = self.conv2(out)
    out += self.residual(x)
    mask = self._resize_mask(mask)
    return out, mask

class CNN(nn.Module):
  """
  Composes the ResBlocks into a downsampler, bringing the padding mask
  along so that that it also gets downsampled
  """
  def __init__(self, d_model, max_len, dropout_p):
    super().__init__()
    self.max_len = max_len
    self.in_channels = d_model
    # extractor
    self.extractor = nn.ModuleList([
          ResBlock(self.in_channels, self.in_channels, kernel_size=7),            # (B, C, T)   -> (B, C, T)

          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T)   -> (B, C, T)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T)   -> (B, C, T)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T)   -> (B, C, T)

          ResBlock(self.in_channels, self.in_channels, kernel_size=3, stride=2),  # (B, C, T)   -> (B, C, T/2)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T/2) -> (B, C, T/2)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T/2) -> (B, C, T/2)

          ResBlock(self.in_channels, self.in_channels, kernel_size=3, stride=2),  # (B, C, T/2) -> (B, C, T/4)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T/2) -> (B, C, T/4)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T/2) -> (B, C, T/4)
          ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T/2) -> (B, C, T/4)
          ])
    self.dropout = nn.Dropout(p=dropout_p)
    # calculate fc layer input with dummy passthrough
    self.output_shapes = self._get_output_shape()

  def forward(self, x, mask):
    for block in self.extractor:
      x, mask= block(x,mask)
    return x, mask

  def _get_output_shape(self):
      """
      Returns output shapes for the data and mask
      """
      dummy_x = torch.randn(1, self.in_channels, self.max_len)
      dummy_mask = torch.randn(1, self.max_len)

      # get outputshapes
      output, mask = self.forward(dummy_x, dummy_mask)
      return output.shape, mask.shape


### Encoder Class

class SmrtEncoder(nn.Module):
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=4096, dropout_p=0.01):
    super().__init__()
    self.d_model = d_model
    self.embed = SmrtEmbedding(d_model)
    self.pe = PositionalEncoding(d_model, max_len=max_len)
    self.downsample = CNN(d_model, max_len=max_len, dropout_p=dropout_p)
    self.layer_norm_target = nn.LayerNorm(d_model)
    self.blocks = nn.ModuleList([
        TransformerBlock(d_model=d_model, n_head=n_head, max_len=max_len) for _ in range(n_layers)
        ])
  def get_latents(self, x):
    """
    Runs [x -> Embedding -> CNN -> out] stack (for training)
    Returns:
      z (downsampled latents with PE)
      z_pad (dowsampled padding mask)
      targets (latents without PE)
    """
    # separate into features and padding
    x_nuc = x[...,0]
    x_kin = x[...,1:3]
    x_pad = x[...,3]
    # generate hybrid embedding
    x = self.embed(x_nuc, x_kin, x_pad)
    # featurize the emmbeddings (cnn expect BCT)
    z, z_pad = self.downsample(x.permute(0,2,1), x_pad)
    # permute back to BTC
    z = z.permute(0,2,1)
    targets = self.layer_norm_target(z.clone())
    return z, z_pad, targets

  def add_pe(self, z):
      return self.pe(z)

  def forward_transformer(self, z, z_pad):
    """
    Runs the transformer blocks on the downsampled latents
    Returns:
      c (context aware latents)
    """
    c = z
    for block in self.blocks:
      c = block(c, z_pad)
    return c
  def forward(self, x):
    z, z_pad, _ = self.get_latents(x)
    z = self.add_pe(z)
    c = self.forward_transformer(z, z_pad)
    return c

### Main Model
class Smrt2Vec(nn.Module):
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=4096):
    super().__init__()
    self.d_model = d_model
    self.encoder = SmrtEncoder(d_model, n_layers, n_head, max_len)

    # components specific to pretraining
    self.mask_vec = nn.Parameter(torch.randn(d_model))
    self.project =  nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(), # avoid negative values being ignored with ReLU
        nn.Linear(d_model, d_model)
        )
  def apply_mask(self, x_emb, pad, prob=0.05, size=6):
    B, T, C = x_emb.shape
    mask_idx_centers = (torch.rand(B, T, device=x_emb.device) < prob) & ~(pad.bool())
    mask_idx_full = F.max_pool1d(
        mask_idx_centers.bfloat16(),
        kernel_size=size, stride=1, # hyperparameter here...
        padding=size//2
      ).bool()[:, :T] & (~pad.bool())
    x_masked = x_emb.clone()
    x_masked[mask_idx_full] = self.mask_vec.to(dtype=x_emb.dtype, device=x_emb.device)
    return x_masked, mask_idx_full
  def forward(self, x):
    # dowsampled latents with pe (no transormer block yet)
    z, z_pad, targets = self.encoder.get_latents(x)
    # mask indices for loss
    z_masked, z_masked_bool = self.apply_mask(z, z_pad)
    z_masked_pe = self.encoder.add_pe(z_masked)
    # run through transformer
    c = self.encoder.forward_transformer(z_masked_pe, z_pad)
    # project
    c_proj = self.project(c)
    return c_proj, targets.detach(), z_masked_bool # projected transformer output, detached unmasked downsampled latents (not transfomer applied), boolean matrix of where the targets are




