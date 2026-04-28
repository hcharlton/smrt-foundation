"""Augmentation primitives for SimCLR-style pair-of-views SSL.

All primitives operate on the canonical SMRT tensor layout:

    x: [T, C]   where C = [seq_token, fi, fp, pad_mask]
                pad_mask = 1.0 at padded positions, 0.0 at real bases

Both the v2 CpG pipeline (`scripts/zarr_to_methyl_memmap_v2.py`) and the
OB007 SSL pipeline (`scripts/zarr_to_memmap_instanceNorm.py`) write two rows
per read/window: one "fwd view" with kinetics = fi/fp, one "rev view" with
kinetics = ri/rp (already reverse-indexed and flipped to read in the same
5'→3' direction as the RC'd seq). Each individual row therefore carries
kinetics from a single strand; a tensor never contains fwd and rev
kinetics simultaneously. That fact shapes the `reverse_complement`
augmentation below — see its docstring for the resulting caveat.

Convention: each primitive returns a *new* tensor. The composing policy calls
primitives twice independently to produce (view1, view2); ordering and
per-augmentation probabilities are configurable.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# Channel indices (explicit constants rather than magic numbers).
CH_SEQ = 0
CH_FI = 1
CH_FP = 2
CH_PAD = 3
KIN_CHANNELS = (CH_FI, CH_FP)


def _unpadded_length(x: torch.Tensor) -> int:
    """Length of the prefix of x that is not padded.

    Reads in the OB007 memmap are right-padded: pad channel is 0.0 over the
    real bases, 1.0 over the trailing padded region. This returns the count
    of real-base positions at the front. Falls back to x.shape[0] if the
    pad channel is absent or identically zero.
    """
    pad = x[..., CH_PAD]
    not_pad = (pad == 0.0)
    if bool(not_pad.all()):
        return int(x.shape[0])
    # First padded index from the left.
    # argmax over a bool returns the first True, which here is the first pad.
    first_pad = int((pad == 1.0).to(torch.int32).argmax().item())
    return first_pad


def random_subcrop(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Return a `target_len`-long sub-window of x sampled uniformly from the
    unpadded prefix.

    Three regimes:
      - unpadded_len >= target_len: pick a random start in [0, unpadded_len - target_len]
        and return the contiguous slice.
      - unpadded_len in [1, target_len): return the whole prefix, right-pad with
        zeros over real channels and 1.0 on the pad channel.
      - unpadded_len == 0: return an all-zero tensor with the pad channel set to 1.0.

    The geometric backbone of the SimCLR analogue. Two independent calls on
    the same read produce two different crops, which is what gives the
    contrastive task its predictive signal.
    """
    _, C = x.shape
    unpadded = _unpadded_length(x)

    if unpadded >= target_len:
        max_start = unpadded - target_len
        start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
        return x[start:start + target_len, :].clone()

    out = torch.zeros(target_len, C, dtype=x.dtype)
    out[:, CH_PAD] = 1.0
    if unpadded > 0:
        out[:unpadded] = x[:unpadded]
    return out


def reverse_complement(x: torch.Tensor, rc_lookup: torch.Tensor) -> torch.Tensor:
    """Reverse the time axis and complement the sequence channel.

    CAVEAT — applied to a stored fwd row (kinetics = fi/fp) this produces a
    tensor whose sequence channel matches the complementary strand but whose
    kinetic channels are still the original strand's polymerase dynamics.
    That combination does not correspond to any real observation; the
    genuine rev-view is already a separate row in the memmap (kinetics
    = ri/rp). For CpG methylation the signal is strand-asymmetric, so this
    synthetic view can destroy the downstream label signal. Keep it OFF by
    default; enable only as an ablation, and consider the dataset-level
    fwd↔rev row pairing as the honest alternative for SimCLR-positives.

    Args:
      x: [T, C] tensor, pad channel last.
      rc_lookup: 1-D long tensor of length max_token+1 where
                 rc_lookup[t] = complement_token(t). Built from
                 `smrt_foundation.normalization.build_rc_lookup`.
    """
    out = torch.flip(x, dims=[0]).clone()
    seq = out[..., CH_SEQ].long()
    out[..., CH_SEQ] = rc_lookup.to(device=x.device)[seq].to(x.dtype)
    return out


def kinetics_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add iid Gaussian noise (in z-scored units) to kinetic channels.

    Padded positions are left untouched — we only perturb real data so the
    downstream mean/variance statistics on the non-pad region are the thing
    the encoder learns to be invariant to. Sigma ranges of ~0.05–0.15 are
    reasonable for z-scored kinetics.
    """
    out = x.clone()
    noise = torch.randn(out.shape[0], len(KIN_CHANNELS), dtype=out.dtype, device=out.device) * sigma
    pad = (out[..., CH_PAD] == 1.0).unsqueeze(-1)
    noise = noise.masked_fill(pad, 0.0)
    out[..., CH_FI:CH_FP + 1] = out[..., CH_FI:CH_FP + 1] + noise
    return out


def kinetics_channel_dropout(x: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Independently zero out each kinetic channel with probability `p_drop`.

    The two kinetic channels (fi, fp) encode different aspects of
    polymerase behaviour; forcing the encoder to cope when either is
    missing is analogous to color-channel dropout in SimCLR. Zeroing in
    z-scored space is equivalent to substituting the per-channel mean,
    which is a minimally biased fill value.
    """
    out = x.clone()
    for c in KIN_CHANNELS:
        if float(torch.rand(1).item()) < p_drop:
            out[..., c] = 0.0
    return out


def kinetics_temporal_blur(x: torch.Tensor, sigma_range=(0.2, 2.0)) -> torch.Tensor:
    """Gaussian-smooth the kinetic channels along the time axis.

    Discrete 1-D Gaussian kernel, `sigma` sampled uniformly from
    `sigma_range` per call, radius = ceil(3 * sigma) truncated. Kernel is
    normalised to sum to 1. Seq and pad channels are left untouched.
    After smoothing we re-zero the kinetics at padded positions so pad
    leakage across the smoothing kernel does not pollute the real region.
    """
    sigma = float(torch.empty(1).uniform_(*sigma_range).item())
    radius = max(1, int(math.ceil(3.0 * sigma)))
    ks = 2 * radius + 1
    t = torch.arange(ks, dtype=x.dtype, device=x.device) - radius
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()

    out = x.clone()
    # [T, 2] -> [1, 2, T] for conv1d with groups=2.
    kin = out[..., CH_FI:CH_FP + 1].transpose(0, 1).unsqueeze(0).contiguous()  # [1, 2, T]
    depthwise = kernel.view(1, 1, -1).expand(len(KIN_CHANNELS), 1, ks).contiguous()
    smoothed = F.conv1d(kin, depthwise, padding=radius, groups=len(KIN_CHANNELS))
    out[..., CH_FI:CH_FP + 1] = smoothed.squeeze(0).transpose(0, 1)

    pad = (out[..., CH_PAD] == 1.0)
    out[..., CH_FI] = out[..., CH_FI].masked_fill(pad, 0.0)
    out[..., CH_FP] = out[..., CH_FP].masked_fill(pad, 0.0)
    return out


class AugmentationPolicy:
    """Composes augmentation primitives into a two-view SimCLR policy.

    Calling the policy on a tensor `x` returns `(view1, view2)` where each
    view is generated by an independent random application of the enabled
    primitives. The subcrop is always applied (geometric backbone); every
    other primitive is Bernoulli-gated by its per-aug probability.

    The default configuration mirrors the SimCLR v1 recipe scaled to this
    domain: subcrop always, kinetic jitter most of the time, moderate
    blur, occasional channel dropout. Reverse-complement is off by default
    (see `reverse_complement` docstring).
    """

    def __init__(
        self,
        target_len: int,
        *,
        rc_lookup=None,
        revcomp_p: float = 0.0,
        channel_dropout_p: float = 0.2,
        gaussian_noise_p: float = 0.8,
        gaussian_noise_sigma: float = 0.1,
        blur_p: float = 0.5,
        blur_sigma_range=(0.2, 2.0),
    ):
        self.target_len = int(target_len)
        self.rc_lookup = rc_lookup if rc_lookup is None or torch.is_tensor(rc_lookup) else torch.as_tensor(rc_lookup, dtype=torch.long)
        self.revcomp_p = float(revcomp_p)
        self.channel_dropout_p = float(channel_dropout_p)
        self.gaussian_noise_p = float(gaussian_noise_p)
        self.gaussian_noise_sigma = float(gaussian_noise_sigma)
        self.blur_p = float(blur_p)
        self.blur_sigma_range = tuple(blur_sigma_range)

    def _one_view(self, x: torch.Tensor) -> torch.Tensor:
        v = random_subcrop(x, self.target_len)
        if self.rc_lookup is not None and self.revcomp_p > 0 and float(torch.rand(1).item()) < self.revcomp_p:
            v = reverse_complement(v, self.rc_lookup)
        if self.channel_dropout_p > 0 and float(torch.rand(1).item()) < self.channel_dropout_p:
            # Under the p_drop gate, force at least some dropout; actual per-channel
            # gating is handled inside the primitive at probability 1.0 for clarity.
            v = kinetics_channel_dropout(v, p_drop=1.0)
        if self.gaussian_noise_p > 0 and float(torch.rand(1).item()) < self.gaussian_noise_p:
            v = kinetics_gaussian_noise(v, self.gaussian_noise_sigma)
        if self.blur_p > 0 and float(torch.rand(1).item()) < self.blur_p:
            v = kinetics_temporal_blur(v, self.blur_sigma_range)
        return v

    def __call__(self, x: torch.Tensor):
        return self._one_view(x), self._one_view(x)


### Experimental (ssl_56) — neighbor-invariance augmentation
#
# Sibling primitive and policy for ssl_56_simclr_neighbor_invariance.
# Diagnoses the ssl_54/55 failure where the encoder learned same-read
# membership (global polymerase / kinetic statistics) as its discrimination
# signal because random_subcrop pairs arbitrary distant windows from the
# same read. neighbor_pair_subcrop forces positives to share local context:
# two non-overlapping target_len windows separated by gap_bp real bases on
# the same molecule. The invariant the encoder is asked to learn becomes
# short-range neighbor invariance over local kinetic dynamics — aligned
# with the locality the methylation signal lives at.


def neighbor_pair_subcrop(x: torch.Tensor, target_len: int, gap_bp: int):
    """Sample a pair of non-overlapping `target_len` windows from x's
    unpadded prefix, with `gap_bp` real (non-pad) bases between view1's end
    and view2's start.

    Both views are returned together (sharing one anchor draw), which is
    what makes them a positive in the neighbor-invariance contrastive task.

    Three regimes mirror `random_subcrop`'s behaviour:
      - unpadded >= 2*target_len + gap_bp: pick a uniform anchor in
        [0, unpadded - 2*target_len - gap_bp], return view1 from
        [anchor, anchor + target_len) and view2 from
        [anchor + target_len + gap_bp, anchor + 2*target_len + gap_bp).
      - unpadded in [1, 2*target_len + gap_bp): the read is too short for
        a valid neighbor pair; return two all-pad views (zeros over real
        channels, 1.0 on the pad channel) so the policy is robust at the
        short tail of the distribution. This matches the spirit of
        `random_subcrop`'s short-read fallback.
      - unpadded == 0: same all-pad fallback.
    """
    _, C = x.shape
    unpadded = _unpadded_length(x)
    required = 2 * target_len + gap_bp

    if unpadded >= required:
        max_start = unpadded - required
        anchor = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
        v1_start = anchor
        v2_start = anchor + target_len + gap_bp
        view1 = x[v1_start:v1_start + target_len, :].clone()
        view2 = x[v2_start:v2_start + target_len, :].clone()
        return view1, view2

    # Short-read fallback: two all-pad views.
    out1 = torch.zeros(target_len, C, dtype=x.dtype)
    out1[:, CH_PAD] = 1.0
    out2 = torch.zeros(target_len, C, dtype=x.dtype)
    out2[:, CH_PAD] = 1.0
    return out1, out2


class NeighborPairAugmentationPolicy:
    """Two-view SimCLR policy where positives are non-overlapping
    `target_len`-base windows separated by `gap_bp` real bases on the same
    molecule.

    The invariant the encoder is asked to learn: nearby local kinetic
    contexts on the same polymerase trace should embed similarly. Per-view
    perturbations (kinetic channel dropout, Gaussian noise, temporal blur,
    optional reverse-complement) are applied independently to view1 and
    view2 with the same per-primitive probabilities as
    `AugmentationPolicy`. Only the *pair-selection* differs — instead of
    two independent `random_subcrop` calls, one `neighbor_pair_subcrop`
    call samples both views together with a shared anchor.

    Constructor signature mirrors `AugmentationPolicy` plus one extra
    required kwarg `gap_bp`.
    """

    def __init__(
        self,
        target_len: int,
        gap_bp: int,
        *,
        rc_lookup=None,
        revcomp_p: float = 0.0,
        channel_dropout_p: float = 0.2,
        gaussian_noise_p: float = 0.8,
        gaussian_noise_sigma: float = 0.1,
        blur_p: float = 0.5,
        blur_sigma_range=(0.2, 2.0),
    ):
        self.target_len = int(target_len)
        self.gap_bp = int(gap_bp)
        self.rc_lookup = rc_lookup if rc_lookup is None or torch.is_tensor(rc_lookup) else torch.as_tensor(rc_lookup, dtype=torch.long)
        self.revcomp_p = float(revcomp_p)
        self.channel_dropout_p = float(channel_dropout_p)
        self.gaussian_noise_p = float(gaussian_noise_p)
        self.gaussian_noise_sigma = float(gaussian_noise_sigma)
        self.blur_p = float(blur_p)
        self.blur_sigma_range = tuple(blur_sigma_range)

    def _apply_perview(self, v: torch.Tensor) -> torch.Tensor:
        if self.rc_lookup is not None and self.revcomp_p > 0 and float(torch.rand(1).item()) < self.revcomp_p:
            v = reverse_complement(v, self.rc_lookup)
        if self.channel_dropout_p > 0 and float(torch.rand(1).item()) < self.channel_dropout_p:
            v = kinetics_channel_dropout(v, p_drop=1.0)
        if self.gaussian_noise_p > 0 and float(torch.rand(1).item()) < self.gaussian_noise_p:
            v = kinetics_gaussian_noise(v, self.gaussian_noise_sigma)
        if self.blur_p > 0 and float(torch.rand(1).item()) < self.blur_p:
            v = kinetics_temporal_blur(v, self.blur_sigma_range)
        return v

    def __call__(self, x: torch.Tensor):
        v1, v2 = neighbor_pair_subcrop(x, self.target_len, self.gap_bp)
        return self._apply_perview(v1), self._apply_perview(v2)
