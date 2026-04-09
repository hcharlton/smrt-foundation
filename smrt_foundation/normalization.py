# smrt_foundation/normalization.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

module_path = os.path.abspath("/dcai/projects/cu_0030/smrt-foundation")
if module_path not in sys.path:
    sys.path.append(module_path)

from smrt_foundation.dataset import ChunkedRandomSampler

def build_rc_lookup(config):
    """
    Creates a numpy lookup table for RC conversion based on config maps.
    Returns: np.array where index=input_token, value=rc_token (requires ints)
    """
    token_map = config['data']['token_map']
    rc_map = config['data']['rc_map']
    
    max_token = max(token_map.values())
    
    lookup = np.arange(max_token + 1, dtype=np.int8)
    
    for base, idx in token_map.items():
        if base in rc_map:
            comp_base = rc_map[base]
            if comp_base in token_map:
                lookup[idx] = token_map[comp_base]
                
    return lookup
    
### MAD normalization
def normalize_read_mad(read_data, is_continuous_mask, eps=1e-6):
    """
    MAD normalization of a single read on the continous features
    
    read_data: array of data for one read
    is_continuous_mask: boolean mask to index only the continuous features 
                        (masks out the categorical features)
    eps: Description
    """
    np.log1p(read_data, out=read_data, where=is_continuous_mask)
    x = read_data[:, is_continuous_mask]
    x_median = np.median(x, axis=0)
    mad = np.median(np.abs(x - x_median), axis=0)
    mad = np.where(mad < eps, 1.0, mad)
    x_norm = (x - x_median) / (mad * 1.4826)
    read_data[:, is_continuous_mask] = x_norm

    return read_data

########### Normalization Classes #############

class ZNorm:
    def __init__(self, ds, eps=1e-8, log_transform=True):
        self.log_transform = log_transform
        self.sampler = ChunkedRandomSampler(ds, 2048, shuffle_within=True)
        x, _ = next(iter(DataLoader(ds, batch_size=1048576, sampler = self.sampler)))
        if self.log_transform:
            x = x.clone()
            x[..., [1, 2]] = torch.log1p(x[..., [1, 2]])
        self.means = torch.mean(x, dim=(0,1))
        self.stds = torch.std(x, dim=(0,1))
        self.eps = eps

    def __call__(self, x):
        if self.log_transform:
            x[..., [1, 2]] = torch.log1p(x[..., [1, 2]])
        x[..., [1, 2]] -= self.means[[1, 2]]
        x[..., [1, 2]] /= (self.stds[[1, 2]] + self.eps)
        return x


class KineticsNorm:
    """Unified log1p + z-score normalization for kinetics channels [1, 2].

    Replaces both ZNorm (for LabeledMemmapDataset) and SSLNorm (for
    ShardedMemmapDataset). Handles both dataset types automatically and
    excludes padded positions when computing statistics.

    Args:
        ds: Any Dataset. If __getitem__ returns a tuple, the first element
            is used. If it returns a tensor, used directly.
        eps: Epsilon for numerical stability in division.
        log_transform: Whether to apply log1p before z-scoring.
        max_samples: Maximum samples to load for computing statistics.
    """
    def __init__(self, ds, eps=1e-8, log_transform=True, max_samples=1_048_576):
        self.log_transform = log_transform
        self.eps = eps

        sampler = ChunkedRandomSampler(ds, 2048, shuffle_within=True)
        batch = next(iter(DataLoader(ds, batch_size=max_samples, sampler=sampler)))

        # Handle both (x, y) tuples and raw tensors
        x = batch[0] if isinstance(batch, (tuple, list)) else batch

        if self.log_transform:
            x = x.clone()
            x[..., [1, 2]] = torch.log1p(x[..., [1, 2]])

        # Exclude padded positions (mask channel is last: 0.0 = real, 1.0 = pad)
        active = x[..., -1] == 0.0

        self.means = torch.zeros(x.shape[-1])
        self.stds = torch.ones(x.shape[-1])
        for c in [1, 2]:
            vals = x[..., c][active]
            if vals.numel() > 0:
                self.means[c] = vals.mean()
                self.stds[c] = vals.std()

    @classmethod
    def from_stats(cls, means, stds, log_transform=True, eps=1e-8):
        """Construct a KineticsNorm from pre-computed statistics.

        Bypasses the dataset-sampling path in __init__. Low-level constructor
        used internally by `load_stats`; callers persisting state into a
        training checkpoint should pair `save_stats` / `load_stats`
        instead so the key schema stays in one place.

        Args:
            means: 1-D tensor of shape (C,) with per-channel means. Only
                   channels [1, 2] are read by __call__; other channel values
                   are accepted for layout compatibility with the __init__
                   path (which stores a full-width vector).
            stds: 1-D tensor of shape (C,), same layout as means.
            log_transform: Must match the value used at training time. True
                           for any checkpoint produced by the standard
                           training flow (ZNorm or KineticsNorm with default
                           log_transform=True).
            eps: Division-by-zero guard. Default matches __init__.

        Returns:
            A KineticsNorm instance with .means, .stds, .log_transform, .eps
            set directly. No DataLoader is instantiated.
        """
        means_t = means.clone().float() if torch.is_tensor(means) else torch.tensor(means, dtype=torch.float32)
        stds_t = stds.clone().float() if torch.is_tensor(stds) else torch.tensor(stds, dtype=torch.float32)
        assert means_t.ndim == 1, f"means must be 1-D, got shape {tuple(means_t.shape)}"
        assert stds_t.shape == means_t.shape, (
            f"stds shape {tuple(stds_t.shape)} must match means shape {tuple(means_t.shape)}"
        )

        instance = cls.__new__(cls)
        instance.log_transform = log_transform
        instance.eps = eps
        instance.means = means_t
        instance.stds = stds_t
        return instance

    def save_stats(self):
        """Return the minimal state needed to reconstruct this normalizer.

        Produces a dict with `norm_`-prefixed keys that a training script
        can merge directly into its `torch.save(...)` payload. Tensors are
        detached and moved to CPU so the saved file is portable across
        devices. The `norm_` prefix keeps these fields from colliding with
        model state dict keys or other checkpoint metadata.

        Example:
            >>> torch.save({
            ...     'model_state_dict': model.state_dict(),
            ...     'config': config,
            ...     **norm_fn.save_stats(),   # merges norm_means, norm_stds, norm_log_transform
            ... }, path)
        """
        return {
            'norm_means': self.means.detach().cpu(),
            'norm_stds': self.stds.detach().cpu(),
            'norm_log_transform': self.log_transform,
        }

    @classmethod
    def load_stats(cls, state, eps=1e-8):
        """Inverse of `save_stats`: rebuild a KineticsNorm from a saved dict.

        Accepts either the dict returned by `save_stats` or a full checkpoint
        dict that contains the `norm_*` keys alongside model weights, config,
        etc. Extra keys are ignored. `norm_log_transform` defaults to True
        when absent, which matches the standard training flow.

        Example:
            >>> ckpt = torch.load('checkpoints/epoch_20.pt', map_location='cpu')
            >>> norm_fn = KineticsNorm.load_stats(ckpt)
            >>> x_normed = norm_fn(x)
        """
        assert 'norm_means' in state and 'norm_stds' in state, (
            "state must contain 'norm_means' and 'norm_stds' (produced by save_stats)"
        )
        return cls.from_stats(
            state['norm_means'],
            state['norm_stds'],
            log_transform=state.get('norm_log_transform', True),
            eps=eps,
        )

    def __call__(self, x):
        if self.log_transform:
            x[..., [1, 2]] = torch.log1p(x[..., [1, 2]])
        x[..., [1, 2]] -= self.means[[1, 2]]
        x[..., [1, 2]] /= (self.stds[[1, 2]] + self.eps)
        return x
