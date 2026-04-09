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
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # make an index vector
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
    self.kin_embed = nn.Linear(n_continuous, d_model//2, dtype=torch.float32)
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
      attn_mask = ~x_pad.bool().view(B, 1, 1, T)
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
    groups1 = min(in_channels // 4, 32) 
    self.gn1 = nn.GroupNorm(num_groups=groups1, num_channels=in_channels)
    self.conv1 = nn.Conv1d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=self.padding,
                           bias=False)
    groups2 = min(out_channels // 4, 32) 
    self.gn2 = nn.GroupNorm(num_groups=groups2, num_channels=out_channels)
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
    if mask.dtype != torch.float:
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
    return mask

  def forward(self, x, mask):
    out = self.relu(self.gn1(x))
    out = self.conv1(out)
    out = self.relu(self.gn2(out))
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
    self.r0 = self._compute_r0()

  def _compute_r0(self):
    cum_rf = 1
    cum_stride = 1
    for block in self.extractor:
      cum_rf += (block.kernel_size - 1) * cum_stride
      cum_stride *= block.stride # only the first layer in each block can be != 1
      cum_rf += (block.kernel_size - 1) * cum_stride 
    return cum_rf

  def forward(self, x, mask):
    for block in self.extractor:
      x, mask= block(x,mask)
    x = self.dropout(x)
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
    self.cnn = CNN(d_model, max_len=max_len, dropout_p=dropout_p)
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
    z, z_pad = self.cnn(x.permute(0,2,1), x_pad)
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
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=4096, p_mask=0.05, mask_size=6):
    super().__init__()
    self.d_model = d_model
    self.p_mask = p_mask
    self.mask_size = mask_size
    self.encoder = SmrtEncoder(d_model, n_layers, n_head, max_len)

    # components specific to pretraining
    self.mask_vec = nn.Parameter(torch.randn(d_model))
    self.project =  nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(), # avoid negative values being ignored with ReLU
        nn.Linear(d_model, d_model)
        )
  def apply_mask(self, x_emb, pad, p_mask, mask_size):
    B, T, C = x_emb.shape
    mask_idx_centers = (torch.rand(B, T, device=x_emb.device) < p_mask) & ~(pad.bool())
    mask_idx_full = F.max_pool1d(
        mask_idx_centers.float(),
        kernel_size=mask_size, stride=1, # hyperparameter here...
        padding=mask_size//2
      ).bool()[:, :T] & (~pad.bool())
    x_masked = x_emb.clone()
    x_masked[mask_idx_full] = self.mask_vec.to(dtype=x_emb.dtype, device=x_emb.device)
    return x_masked, mask_idx_full
  def forward(self, x):
    # dowsampled latents with pe (no transormer block yet)
    z, z_pad, targets = self.encoder.get_latents(x)
    # mask indices for loss
    z_masked, z_masked_bool = self.apply_mask(z, z_pad, self.p_mask, self.mask_size)
    z_masked_pe = self.encoder.add_pe(z_masked)
    # run through transformer
    c = self.encoder.forward_transformer(z_masked_pe, z_pad)
    # project
    c_proj = self.project(c)
    return c_proj, targets.detach(), z_masked_bool # projected transformer output, detached unmasked downsampled latents (not transfomer applied), boolean matrix of where the targets are



class Smrt2VecInputMask(nn.Module):
  """
  SSL model with input-level masking (wav2vec 2.0 style).

  Masks raw input kinetics BEFORE the CNN, forcing the encoder to learn
  representations without information at masked positions. The CNN's
  107-base receptive field means latent masking (Smrt2Vec) leaks information
  through overlapping receptive fields — this class fixes that.

  Same return signature as Smrt2Vec: (c_proj, targets, mask_idx)
  Uses the same SmrtEncoder, so pretrained weights transfer to DirectClassifier.
  """
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=4096, p_mask=0.15, mask_size=10):
    super().__init__()
    self.d_model = d_model
    self.p_mask = p_mask
    self.mask_size = mask_size
    self.encoder = SmrtEncoder(d_model, n_layers, n_head, max_len)

    self.project = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model)
    )

  def apply_input_mask(self, x, p_mask, mask_size):
    """Mask kinetics channels at the input level (before CNN).

    Zeros out kinetics (channels 1, 2) at random contiguous spans.
    Keeps sequence tokens (channel 0) and padding mask (channel 3) intact.

    Returns:
      x_masked: input with kinetics zeroed at masked positions
      mask_bool: boolean mask at input resolution [B, T]
    """
    B, T, C = x.shape
    pad = x[..., 3]

    # Sample mask centers, exclude padding
    mask_centers = (torch.rand(B, T, device=x.device) < p_mask) & ~(pad.bool())

    # Expand centers into contiguous spans
    mask_full = F.max_pool1d(
        mask_centers.float().unsqueeze(1),
        kernel_size=mask_size, stride=1,
        padding=mask_size // 2
    ).squeeze(1)[:, :T].bool() & ~(pad.bool())

    # Zero out kinetics at masked positions, keep seq and pad intact
    x_masked = x.clone()
    x_masked[mask_full, 1] = 0.0  # fi / IPD
    x_masked[mask_full, 2] = 0.0  # fp / pulse width

    return x_masked, mask_full

  def _downsample_mask(self, mask, target_len):
    """Downsample input-resolution mask to CNN output resolution.

    A downsampled position is masked if ANY of its corresponding
    input positions were masked (conservative — ensures we predict
    at every position that lost information).
    """
    m = mask.float().unsqueeze(1)  # [B, 1, T]
    # Match the CNN's two stride-2 downsampling blocks
    m = F.max_pool1d(m, kernel_size=2, stride=2)  # T → T/2
    m = F.max_pool1d(m, kernel_size=2, stride=2)  # T/2 → T/4
    return m.squeeze(1).bool()[:, :target_len]

  def forward(self, x):
    # Targets: unmasked CNN features (no grad needed for targets)
    with torch.no_grad():
      _, _, targets = self.encoder.get_latents(x)

    # Masked path: mask input kinetics, then run through full encoder
    x_masked, input_mask = self.apply_input_mask(x, self.p_mask, self.mask_size)
    z, z_pad, _ = self.encoder.get_latents(x_masked)
    z = self.encoder.add_pe(z)
    c = self.encoder.forward_transformer(z, z_pad)
    c_proj = self.project(c)

    # Downsample mask to match CNN output resolution
    ds_mask = self._downsample_mask(input_mask, z.shape[1])

    return c_proj, targets.detach(), ds_mask


class Smrt2VecInputMaskToken(nn.Module):
  """
  SSL model with input-level mask-token injection (wav2vec 2.0 style, mask-token variant).

  Variant of Smrt2VecInputMask. Instead of zeroing kinetics channels at the
  raw input, this class lets the embedding run normally, then overwrites the
  embedding output at masked positions with a learnable d_model mask token
  before the CNN. Compared to:

  - Smrt2Vec (latent-level masking after the CNN): the original kinetics
    never reach the CNN through the masked path, so the CNN's 107-base
    receptive field cannot leak the masked values through neighboring latents.

  - Smrt2VecInputMask (zeroing channels 1, 2 at the raw input): zeros are
    indistinguishable from "real value at the mean of the normalized
    distribution" because KineticsNorm mean-centers the kinetics, so the
    encoder gets no signal that a position is masked. The learnable token
    here gives the encoder an explicit, distinct mask signal at the same
    representational level as the embedding output.

  Architectural differences from Smrt2VecInputMask:
    - Adds self.mask_vec: nn.Parameter(d_model), same shape as Smrt2Vec.mask_vec.
    - apply_input_mask returns only the boolean mask, not a masked input tensor.
    - forward() runs encoder.embed -> inject mask_vec -> encoder.cnn -> add_pe ->
      forward_transformer manually (cannot use encoder.get_latents on the
      masked path because it has no mid-call hook).

  Same return signature as Smrt2VecInputMask: (c_proj, targets, mask_idx).
  Uses the same SmrtEncoder, so pretrained encoder weights remain
  interchangeable with DirectClassifier for fine-tuning.
  """
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=4096, p_mask=0.15, mask_size=10):
    super().__init__()
    self.d_model = d_model
    self.p_mask = p_mask
    self.mask_size = mask_size
    self.encoder = SmrtEncoder(d_model, n_layers, n_head, max_len)

    # Learnable mask token, injected at the embedding output before the CNN.
    # Same shape as Smrt2Vec.mask_vec so the parameter has equivalent capacity.
    self.mask_vec = nn.Parameter(torch.randn(d_model))

    self.project = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, d_model)
    )

  def apply_input_mask(self, x, p_mask, mask_size):
    """Sample a contiguous-span boolean mask at input resolution.

    Returns:
      mask_full: boolean mask at input resolution [B, T]. Padded positions
                 are excluded from the mask.

    The actual mask token injection is performed at the embedding output
    in forward(); this method only samples which positions to mask.
    """
    B, T, _ = x.shape
    pad = x[..., 3]

    # Sample mask centers, exclude padding
    mask_centers = (torch.rand(B, T, device=x.device) < p_mask) & ~(pad.bool())

    # Expand centers into contiguous spans
    mask_full = F.max_pool1d(
        mask_centers.float().unsqueeze(1),
        kernel_size=mask_size, stride=1,
        padding=mask_size // 2
    ).squeeze(1)[:, :T].bool() & ~(pad.bool())

    return mask_full

  def _downsample_mask(self, mask, target_len):
    """Downsample input-resolution mask to CNN output resolution.

    A downsampled position is masked if ANY of its corresponding
    input positions were masked (conservative -- ensures we predict
    at every position that lost information).
    """
    m = mask.float().unsqueeze(1)  # [B, 1, T]
    # Match the CNN's two stride-2 downsampling blocks
    m = F.max_pool1d(m, kernel_size=2, stride=2)  # T -> T/2
    m = F.max_pool1d(m, kernel_size=2, stride=2)  # T/2 -> T/4
    return m.squeeze(1).bool()[:, :target_len]

  def forward(self, x):
    # Targets: unmasked CNN features (no grad needed for targets)
    with torch.no_grad():
      _, _, targets = self.encoder.get_latents(x)

    # Sample the input-resolution mask
    input_mask = self.apply_input_mask(x, self.p_mask, self.mask_size)

    # Run the embedding manually so the learnable mask token can be injected
    # between embed and CNN. encoder.get_latents() is not used here because
    # it has no mid-call hook for swapping in the mask token.
    x_nuc = x[..., 0]
    x_kin = x[..., 1:3]
    x_pad = x[..., 3]
    x_emb = self.encoder.embed(x_nuc, x_kin, x_pad)  # [B, T, d_model]

    # Replace the embedding at masked positions with the learnable mask token
    x_emb = x_emb.clone()
    x_emb[input_mask] = self.mask_vec.to(dtype=x_emb.dtype, device=x_emb.device)

    # Continue through CNN, PE, transformer
    z, z_pad = self.encoder.cnn(x_emb.permute(0, 2, 1), x_pad)
    z = z.permute(0, 2, 1)
    z = self.encoder.add_pe(z)
    c = self.encoder.forward_transformer(z, z_pad)
    c_proj = self.project(c)

    # Downsample mask to match CNN output resolution
    ds_mask = self._downsample_mask(input_mask, z.shape[1])

    return c_proj, targets.detach(), ds_mask


class SmrtDecoder(nn.Module):
  """Lightweight decoder that upsamples transformer output back to input resolution."""
  def __init__(self, d_model):
    super().__init__()
    self.upsample = nn.Sequential(
      nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
      nn.GELU(),
      nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
      nn.GELU(),
    )
    self.head = nn.Linear(d_model, 2)  # predict kinetics (IPD, PW)

  def forward(self, z):
    # z: [B, T/4, d_model]
    z = z.permute(0, 2, 1)       # [B, d_model, T/4]
    z = self.upsample(z)         # [B, d_model, T]
    z = z.permute(0, 2, 1)       # [B, T, d_model]
    return self.head(z)          # [B, T, 2]


class SmrtAutoencoder(nn.Module):
  """Masked autoencoder for kinetics reconstruction.

  Masks input kinetics at random spans, encodes through CNN + transformer,
  decodes back to input resolution, and reconstructs the masked kinetics.
  """
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=128, p_mask=0.15, mask_size=10):
    super().__init__()
    self.encoder = SmrtEncoder(d_model, n_layers, n_head, max_len)
    self.decoder = SmrtDecoder(d_model)
    self.p_mask = p_mask
    self.mask_size = mask_size

  def apply_input_mask(self, x, p_mask, mask_size):
    """Mask kinetics channels at the input level (before CNN).

    Zeros out kinetics (channels 1, 2) at random contiguous spans.
    Keeps sequence tokens (channel 0) and padding mask (channel 3) intact.
    """
    B, T, C = x.shape
    pad = x[..., 3]

    mask_centers = (torch.rand(B, T, device=x.device) < p_mask) & ~(pad.bool())

    mask_full = F.max_pool1d(
        mask_centers.float().unsqueeze(1),
        kernel_size=mask_size, stride=1,
        padding=mask_size // 2
    ).squeeze(1)[:, :T].bool() & ~(pad.bool())

    x_masked = x.clone()
    x_masked[mask_full, 1] = 0.0
    x_masked[mask_full, 2] = 0.0

    return x_masked, mask_full

  def forward(self, x):
    x_orig = x.clone()
    x_masked, mask = self.apply_input_mask(x, self.p_mask, self.mask_size)
    c = self.encoder(x_masked)       # [B, T/4, d_model]
    kin_recon = self.decoder(c)       # [B, T, 2]
    return kin_recon, x_orig[..., 1:3], mask


class DirectClassifier(nn.Module):
  def __init__(self, d_model, n_layers, n_head, max_len):
    super().__init__()
    self.encoder = SmrtEncoder(d_model, n_layers, n_head, max_len)
    self.head = nn.Sequential(
      nn.Linear(d_model, d_model//2),
      nn.GELU(),
      nn.Linear(d_model//2, 1)
    )

  def forward(self, x):
    c = self.encoder.forward(x)
    logits = self.head(c[:, c.shape[1]//2, :])
    return logits


class DirectClassifierNoTransformer(nn.Module):
  """CNN-only ablation of DirectClassifier.

  Same SmrtEmbedding and CNN as the default encoder, but no PositionalEncoding
  and no TransformerBlocks. The classification head (center latent -> d/2 ->
  GELU -> 1) is identical to DirectClassifier.

  Intended as a capacity ablation of the supervised baseline at ctx=32, where
  the default CNN's receptive field (107) already covers the full 32-base
  input. Under that condition every CNN output latent is a function of the
  entire input, so the transformer's bidirectional attention has less to
  contribute than it would at longer contexts — its job is just to re-mix
  latents that each already see everything. This class tests how much of the
  supervised baseline's accuracy actually depends on that re-mixing.

  State dict keys are NOT interchangeable with DirectClassifier: the
  submodules (embed, cnn, head) sit directly on self instead of underneath
  self.encoder, so weights cannot be transferred between the two classes.
  This is fresh-training only.
  """
  def __init__(self, d_model, max_len, dropout_p=0.01):
    super().__init__()
    self.d_model = d_model
    self.embed = SmrtEmbedding(d_model)
    self.cnn = CNN(d_model, max_len=max_len, dropout_p=dropout_p)
    self.head = nn.Sequential(
      nn.Linear(d_model, d_model // 2),
      nn.GELU(),
      nn.Linear(d_model // 2, 1),
    )

  def forward(self, x):
    x_nuc = x[..., 0]
    x_kin = x[..., 1:3]
    x_pad = x[..., 3]
    z = self.embed(x_nuc, x_kin, x_pad)          # [B, T, d]
    z, _ = self.cnn(z.permute(0, 2, 1), x_pad)   # [B, d, T/4]
    z = z.permute(0, 2, 1)                        # [B, T/4, d]
    return self.head(z[:, z.shape[1] // 2, :])    # [B, 1]


### Small-receptive-field variants
#
# The default CNN has 11 ResBlocks and a receptive field of ~107 bases. At
# short downstream contexts (e.g., the 32-base CpG windows used by exp
# 25/26/27), RF (107) >> context (32), which means every CNN output latent
# is a function of the entire input — there is no "local" region of the
# input that a latent specifically represents. This erases the locality
# that masked-prediction SSL objectives (both contrastive InfoNCE and the
# masked autoencoder) implicitly depend on: the model is supposed to
# "predict the missing position given the surrounding context", but with
# RF > context the surrounding context IS the entire input, including the
# position being predicted.
#
# The Small-RF variants below rebuild the CNN with 4 ResBlocks and a
# receptive field of 27 bases, which fits comfortably inside a 32-base
# context window. The downsampling ratio (4x) is preserved so latent counts
# and positional geometry match the default encoder at equal contexts,
# which keeps the probe head, the decoder, and the shape contracts of the
# SSL wrappers unchanged.
#
# Each variant is implemented by inheriting from its non-variant parent
# and explicitly calling nn.Module.__init__() to skip the parent's __init__
# (which would build the wrong CNN). All forward-path methods are inherited
# unchanged, so behavior matches the base class except for the CNN stack.
# State-dict keys are NOT interchangeable with the base classes because the
# ResBlock counts differ — these are fresh-training-only variants.


class CNNSmallRF(CNN):
  """
  CNN variant with 4 ResBlocks (k=3, k=3, k=3 stride=2, k=3 stride=2) and
  receptive field 27. Same 4x total downsampling as the default CNN.

  Intended for ctx <= 32 tasks where the default CNN's RF=107 exceeds the
  input and erases per-latent locality. At RF=27 in a 32-base input, each
  latent depends on a distinct ~27-base sub-window, restoring the locality
  that masked-prediction objectives rely on.

  Inherits forward, _compute_r0, and _get_output_shape from CNN.
  """
  def __init__(self, d_model, max_len, dropout_p):
    # Skip CNN.__init__ (which builds the default 11-block extractor).
    nn.Module.__init__(self)
    self.max_len = max_len
    self.in_channels = d_model
    self.extractor = nn.ModuleList([
        ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T)   -> (B, C, T)
        ResBlock(self.in_channels, self.in_channels, kernel_size=3),            # (B, C, T)   -> (B, C, T)
        ResBlock(self.in_channels, self.in_channels, kernel_size=3, stride=2),  # (B, C, T)   -> (B, C, T/2)
        ResBlock(self.in_channels, self.in_channels, kernel_size=3, stride=2),  # (B, C, T/2) -> (B, C, T/4)
    ])
    self.dropout = nn.Dropout(p=dropout_p)
    # calculate fc layer input with dummy passthrough
    self.output_shapes = self._get_output_shape()
    self.r0 = self._compute_r0()


class SmrtEncoderSmallRF(SmrtEncoder):
  """
  SmrtEncoder variant using CNNSmallRF (RF=27) instead of CNN (RF=107).
  See the Small-RF section comment for motivation.

  Inherits get_latents, add_pe, forward_transformer, and forward from
  SmrtEncoder. Only the CNN differs.
  """
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=32, dropout_p=0.01):
    # Skip SmrtEncoder.__init__ (which would instantiate the default CNN).
    nn.Module.__init__(self)
    self.d_model = d_model
    self.embed = SmrtEmbedding(d_model)
    self.pe = PositionalEncoding(d_model, max_len=max_len)
    self.cnn = CNNSmallRF(d_model, max_len=max_len, dropout_p=dropout_p)
    self.layer_norm_target = nn.LayerNorm(d_model)
    self.blocks = nn.ModuleList([
        TransformerBlock(d_model=d_model, n_head=n_head, max_len=max_len) for _ in range(n_layers)
    ])


class SmrtAutoencoderSmallRF(SmrtAutoencoder):
  """
  SmrtAutoencoder variant using SmrtEncoderSmallRF (CNN RF=27) instead of
  SmrtEncoder (CNN RF=107). Intended for short-context autoencoder
  pretraining (ctx <= 32) where the default encoder's RF would cover the
  entire input and defeat the "reconstruct the masked kinetics from the
  surrounding context" framing.

  Same masking, decoder, and return signature as SmrtAutoencoder. Inherits
  apply_input_mask and forward from SmrtAutoencoder.
  """
  def __init__(self, d_model=128, n_layers=4, n_head=4, max_len=32, p_mask=0.15, mask_size=10):
    # Skip SmrtAutoencoder.__init__ (which would build the default encoder).
    nn.Module.__init__(self)
    self.encoder = SmrtEncoderSmallRF(d_model, n_layers, n_head, max_len)
    self.decoder = SmrtDecoder(d_model)
    self.p_mask = p_mask
    self.mask_size = mask_size
