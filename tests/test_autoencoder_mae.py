"""Tests for SmrtAutoencoderMAE (true MAE sparse encoder, ssl_59 lineage).

Companion to tests/test_autoencoder.py — same fixture style, same shape
contracts, but exercises the asymmetric encoder/decoder structure.
"""

import torch
import pytest

from smrt_foundation.model import SmrtAutoencoderMAE, SmrtEncoderSmallRF, SmrtDecoder
from smrt_foundation.loss import MaskedReconstructionLoss


D_MODEL = 128
N_LAYERS = 2
N_HEAD = 4
CONTEXT = 128
BATCH = 4


@pytest.fixture
def sample_input():
    """[B, T, 4] = [nucleotide_token, ipd, pw, pad]; no padding for the base case."""
    x = torch.zeros(BATCH, CONTEXT, 4)
    x[..., 0] = torch.randint(0, 5, (BATCH, CONTEXT)).float()
    x[..., 1] = torch.randn(BATCH, CONTEXT)
    x[..., 2] = torch.randn(BATCH, CONTEXT)
    return x


@pytest.fixture
def padded_sample_input():
    """[B, T, 4] with the last 16 positions padded for half the batch — exercises the
    pad-aware mask path."""
    x = torch.zeros(BATCH, CONTEXT, 4)
    x[..., 0] = torch.randint(0, 5, (BATCH, CONTEXT)).float()
    x[..., 1] = torch.randn(BATCH, CONTEXT)
    x[..., 2] = torch.randn(BATCH, CONTEXT)
    x[: BATCH // 2, -16:, 3] = 1.0
    return x


def _build_model(mask_ratio=0.75, decoder_n_layers=2):
    return SmrtAutoencoderMAE(
        d_model=D_MODEL, n_layers=N_LAYERS, n_head=N_HEAD,
        max_len=CONTEXT, mask_ratio=mask_ratio,
        decoder_n_layers=decoder_n_layers,
    )


class TestSmrtAutoencoderMAEForward:
    def test_output_shapes(self, sample_input):
        model = _build_model()
        kin_recon, kin_target, mask = model(sample_input)
        assert kin_recon.shape == (BATCH, CONTEXT, 2)
        assert kin_target.shape == (BATCH, CONTEXT, 2)
        assert mask.shape == (BATCH, CONTEXT)
        assert mask.dtype == torch.bool

    def test_kin_target_is_input_kinetics(self, sample_input):
        model = _build_model()
        _, kin_target, _ = model(sample_input)
        assert torch.allclose(kin_target, sample_input[..., 1:3])

    def test_mask_coverage_ratio(self, sample_input):
        """Mask should cover ~ mask_ratio of non-pad input positions.

        Each masked latent covers 4 input positions (4x downsample), so the
        coverage in input space matches the latent-space mask ratio (modulo
        rounding to the integer keep_count).
        """
        torch.manual_seed(0)
        model = _build_model(mask_ratio=0.75)
        _, _, mask = model(sample_input)
        non_pad = (sample_input[..., 3] == 0)
        coverage = (mask & non_pad).float().sum() / non_pad.float().sum()
        assert 0.65 <= coverage.item() <= 0.85, f"got {coverage.item():.3f}"

    def test_mask_excludes_padding(self, padded_sample_input):
        """Padding positions must never be marked masked (loss target should not
        include them). The random_mask helper pushes pad to the back of the
        keep-shuffle so padded latents are dropped, but the input-resolution
        mask is then & with ~pad to be defensive."""
        model = _build_model()
        _, _, mask = model(padded_sample_input)
        pad_positions = padded_sample_input[..., 3].bool()
        assert not (mask & pad_positions).any(), \
            "MAE marked padding positions as masked; they should be excluded."


class TestSmrtAutoencoderMAEGradient:
    def test_backward_populates_grads(self, sample_input):
        """Backward through MAE forward + reconstruction loss must populate grads
        on encoder, mask_token, decoder_blocks, and decoder_upsample.

        `encoder.layer_norm_target` is the one allowed exception: it's only used
        by Smrt2Vec's contrastive target path (`get_latents` returns it as
        `targets`), not the autoencoder/MAE path. ssl_58 sets
        `find_unused_parameters=True` on DDP for the same reason.
        """
        model = _build_model()
        criterion = MaskedReconstructionLoss()
        kin_recon, kin_target, mask = model(sample_input)
        loss = criterion(kin_recon, kin_target, mask)
        loss.backward()

        ALLOW_NO_GRAD = {'encoder.layer_norm_target.weight',
                          'encoder.layer_norm_target.bias'}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in ALLOW_NO_GRAD:
                continue
            assert p.grad is not None, f"{name} has no grad"
            assert torch.isfinite(p.grad).all(), f"{name} grad not finite"

    def test_loss_is_finite_and_positive(self, sample_input):
        model = _build_model()
        criterion = MaskedReconstructionLoss()
        kin_recon, kin_target, mask = model(sample_input)
        loss = criterion(kin_recon, kin_target, mask)
        assert torch.isfinite(loss)
        # MSE on standard-normal kinetics minus a randomly-init MAE recon should
        # be O(1); zero would mean the loss isn't actually computed on the
        # masked positions (or the recon is perfectly initialised, unlikely).
        assert loss.item() > 0.0


class TestSmrtAutoencoderMAEDeterminism:
    def test_seeded_mask_is_reproducible(self, sample_input):
        """Two forward passes with the same seed should produce the same mask."""
        model = _build_model()
        torch.manual_seed(123)
        _, _, mask_a = model(sample_input)
        torch.manual_seed(123)
        _, _, mask_b = model(sample_input)
        assert torch.equal(mask_a, mask_b)

    def test_different_seeds_produce_different_masks(self, sample_input):
        model = _build_model()
        torch.manual_seed(1)
        _, _, mask_a = model(sample_input)
        torch.manual_seed(2)
        _, _, mask_b = model(sample_input)
        assert not torch.equal(mask_a, mask_b)


class TestSmrtAutoencoderMAEStructure:
    def test_encoder_is_smrtencoder_smallrf(self):
        model = _build_model()
        assert isinstance(model.encoder, SmrtEncoderSmallRF)
        # CNN receptive field should be 27 (small-RF default for ssl_58/59).
        assert model.encoder.cnn.r0 == 27

    def test_decoder_components_exist(self):
        model = _build_model()
        assert hasattr(model, 'decoder_blocks')
        assert hasattr(model, 'decoder_pe')
        assert hasattr(model, 'decoder_upsample')
        assert isinstance(model.decoder_upsample, SmrtDecoder)

    def test_mask_token_shape_and_init(self):
        model = _build_model()
        assert model.mask_token.shape == (1, 1, D_MODEL)
        # Init std=0.02 per MAE convention; loose bound.
        assert 0.005 < model.mask_token.std().item() < 0.05
