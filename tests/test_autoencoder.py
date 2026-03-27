"""Tests for SmrtAutoencoder, SmrtDecoder, and MaskedReconstructionLoss."""

import torch
import pytest
from smrt_foundation.model import SmrtAutoencoder, SmrtDecoder, SmrtEncoder, DirectClassifier
from smrt_foundation.loss import MaskedReconstructionLoss


D_MODEL = 128
N_LAYERS = 2  # fewer layers for fast tests
N_HEAD = 4
CONTEXT = 128
BATCH = 4


@pytest.fixture
def sample_input():
    """Fake input matching CpG data layout: [B, T, 4] = [seq, ipd, pw, mask]."""
    x = torch.zeros(BATCH, CONTEXT, 4)
    x[..., 0] = torch.randint(0, 5, (BATCH, CONTEXT)).float()  # nucleotides
    x[..., 1] = torch.randn(BATCH, CONTEXT)  # IPD (normalized)
    x[..., 2] = torch.randn(BATCH, CONTEXT)  # PW (normalized)
    # channel 3 = 0 (no padding)
    return x


class TestSmrtDecoder:
    def test_output_shape(self):
        decoder = SmrtDecoder(D_MODEL)
        z = torch.randn(BATCH, CONTEXT // 4, D_MODEL)
        out = decoder(z)
        assert out.shape == (BATCH, CONTEXT, 2)

    def test_different_lengths(self):
        decoder = SmrtDecoder(D_MODEL)
        for ctx in [32, 64, 128, 256]:
            z = torch.randn(2, ctx // 4, D_MODEL)
            out = decoder(z)
            assert out.shape == (2, ctx, 2), f"Failed for context={ctx}"


class TestSmrtAutoencoder:
    def test_forward_shapes(self, sample_input):
        model = SmrtAutoencoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        kin_recon, kin_target, mask = model(sample_input)
        assert kin_recon.shape == (BATCH, CONTEXT, 2)
        assert kin_target.shape == (BATCH, CONTEXT, 2)
        assert mask.shape == (BATCH, CONTEXT)
        assert mask.dtype == torch.bool

    def test_masking_zeros_kinetics(self, sample_input):
        model = SmrtAutoencoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT, p_mask=0.5, mask_size=5)
        x_masked, mask = model.apply_input_mask(sample_input, 0.5, 5)

        # Kinetics should be zeroed at masked positions
        assert (x_masked[mask, 1] == 0).all()
        assert (x_masked[mask, 2] == 0).all()

    def test_masking_preserves_sequence(self, sample_input):
        model = SmrtAutoencoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT, p_mask=0.5, mask_size=5)
        x_masked, mask = model.apply_input_mask(sample_input, 0.5, 5)

        # Sequence tokens and padding mask should be unchanged
        assert (x_masked[..., 0] == sample_input[..., 0]).all()
        assert (x_masked[..., 3] == sample_input[..., 3]).all()

    def test_masking_produces_masked_positions(self, sample_input):
        model = SmrtAutoencoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT, p_mask=0.5, mask_size=5)
        _, mask = model.apply_input_mask(sample_input, 0.5, 5)
        assert mask.any(), "No positions were masked"

    def test_gradient_flows(self, sample_input):
        model = SmrtAutoencoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        kin_recon, kin_target, mask = model(sample_input)
        loss = MaskedReconstructionLoss()(kin_recon, kin_target, mask)
        loss.backward()
        # Check encoder and decoder both receive gradients
        enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.encoder.parameters() if p.requires_grad)
        dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.decoder.parameters() if p.requires_grad)
        assert enc_grad, "Encoder received no gradients"
        assert dec_grad, "Decoder received no gradients"


class TestMaskedReconstructionLoss:
    def test_basic(self):
        loss_fn = MaskedReconstructionLoss()
        recon = torch.randn(BATCH, CONTEXT, 2)
        target = torch.randn(BATCH, CONTEXT, 2)
        mask = torch.rand(BATCH, CONTEXT) > 0.5
        loss = loss_fn(recon, target, mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_reconstruction(self):
        loss_fn = MaskedReconstructionLoss()
        target = torch.randn(BATCH, CONTEXT, 2)
        mask = torch.rand(BATCH, CONTEXT) > 0.5
        loss = loss_fn(target.clone(), target, mask)
        assert loss.item() < 1e-6

    def test_only_masked_positions(self):
        """Loss should only use masked positions."""
        loss_fn = MaskedReconstructionLoss()
        target = torch.zeros(BATCH, CONTEXT, 2)
        recon = torch.ones(BATCH, CONTEXT, 2)
        mask = torch.zeros(BATCH, CONTEXT, dtype=torch.bool)
        mask[:, 0] = True  # only first position masked

        loss = loss_fn(recon, target, mask)
        expected = 1.0  # MSE of (1-0)^2 = 1, averaged over 2 channels
        assert abs(loss.item() - expected) < 1e-5


class TestEncoderCompatibility:
    def test_encoder_weights_transfer(self):
        """Autoencoder's encoder weights should load into DirectClassifier."""
        ae = SmrtAutoencoder(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)
        clf = DirectClassifier(D_MODEL, N_LAYERS, N_HEAD, CONTEXT)

        encoder_sd = ae.encoder.state_dict()
        missing, unexpected = clf.encoder.load_state_dict(encoder_sd, strict=False)

        # Only PE buffer size mismatch is acceptable
        for key in missing:
            assert 'pe' in key, f"Unexpected missing key: {key}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
