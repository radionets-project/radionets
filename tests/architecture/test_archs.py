"""Tests for src/radionets/architecture/archs.py"""

import pytest
import torch
import torch.nn as nn

from radionets.architecture.archs import (
    SRResNet,
    SRResNet18,
    SRResNet18AmpPhase,
    SRResNet18Complex,
    SRResNet34,
    SRResNet34_unc,
    SRResNet34_unc_no_grad,
    SRResNet34AmpPhase,
    SRResNetComplex,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture(
    scope="module",
    params=[8, pytest.param(32, marks=cuda), pytest.param(64, marks=cuda)],
)
def input_tensor(request):
    size = request.param
    return torch.randn(2, 2, size, size)


class TestInstantiation:
    @pytest.mark.parametrize(
        "cls",
        [
            SRResNet,
            SRResNetComplex,
            SRResNet18,
            SRResNet18Complex,
            SRResNet18AmpPhase,
            SRResNet34,
            SRResNet34AmpPhase,
            SRResNet34_unc,
            SRResNet34_unc_no_grad,
        ],
    )
    def test_instantiate_all(self, cls):
        model = cls()
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize(
        "cls",
        [
            SRResNet,
            SRResNetComplex,
            SRResNet18,
            SRResNet18Complex,
            SRResNet18AmpPhase,
            SRResNet34,
            SRResNet34AmpPhase,
            SRResNet34_unc,
            SRResNet34_unc_no_grad,
        ],
    )
    def test_model_parameters_exist(self, cls):
        model = cls()
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_srrresnet_channels(self):
        model = SRResNet()
        assert model.channels == 64

    def test_srrresnet_complex_channels(self):
        model = SRResNetComplex()
        assert model.channels == 128


class TestSRResNetForward:
    def test_srrresnet_has_pre_block(self):
        model = SRResNet()

        assert hasattr(model, "pre_block")
        assert isinstance(model.pre_block, nn.Sequential)

    def test_srrresnet_has_post_block(self):
        model = SRResNet()

        assert hasattr(model, "post_block")
        assert isinstance(model.post_block, nn.Sequential)

    def test_srrresnet_has_final(self):
        model = SRResNet()

        assert hasattr(model, "final")
        assert isinstance(model.final, nn.Sequential)

    def test_srrresnet_no_blocks_by_default(self):
        """Check that SRResNet base class does *not* call _create_blocks."""
        model = SRResNet()

        assert not hasattr(model, "blocks")


class TestSRResNet18Forward:
    def test_forward_returns_dict_with_pred(self, input_tensor):
        model = SRResNet18()
        output = model(input_tensor)

        assert isinstance(output, dict)
        assert "pred" in output

    def test_forward_shape(self, input_tensor):
        model = SRResNet18()
        output = model(input_tensor)

        assert output["pred"].shape == input_tensor.shape

    def test_forward_finite(self, input_tensor):
        model = SRResNet18()
        output = model(input_tensor)

        assert torch.isfinite(output["pred"]).all()


class TestSRResNet34Forward:
    def test_forward_returns_dict_with_pred(self, input_tensor):
        model = SRResNet34()
        output = model(input_tensor)

        assert isinstance(output, dict)
        assert "pred" in output

    def test_forward_shape(self, input_tensor):
        model = SRResNet34()
        output = model(input_tensor)

        assert output["pred"].shape == input_tensor.shape

    def test_forward_finite(self, input_tensor):
        model = SRResNet34()
        output = model(input_tensor)

        assert torch.isfinite(output["pred"]).all()


class TestSRResNetComplexForward:
    def test_has_no_blocks_by_default(self):
        """Check that SRResNetComplex base class does *not* call _create_blocks."""
        model = SRResNetComplex()
        assert not hasattr(model, "blocks")

    def test_srrresnet18complex_has_blocks(self):
        model = SRResNet18Complex()

        assert hasattr(model, "blocks")
        assert len(model.blocks) == 8


class TestAmpPhaseArchitecture:
    def test_srrresnet18_amp_phase_has_restriction_layers(self):
        model = SRResNet18AmpPhase()

        assert hasattr(model, "relu")
        assert hasattr(model, "hardtanh")
        assert isinstance(model.relu, nn.ReLU)

    def test_srrresnet34_amp_phase_has_restriction_layers(self):
        model = SRResNet34AmpPhase()

        assert hasattr(model, "relu")
        assert hasattr(model, "hardtanh")

    def test_srrresnet18_amp_phase_has_blocks(self):
        model = SRResNet18AmpPhase()

        assert hasattr(model, "blocks")
        assert len(model.blocks) == 8

    def test_srrresnet34_amp_phase_has_blocks(self):
        model = SRResNet34AmpPhase()

        assert hasattr(model, "blocks")
        assert len(model.blocks) == 16


@cuda
class TestGPUForwardPasses:
    def test_srrresnet18_gpu(self, input_tensor):
        model = SRResNet18().cuda()
        inp = input_tensor.cuda()
        output = model(inp)
        assert output["pred"].shape == inp.shape
        assert torch.isfinite(output["pred"]).all()

    def test_srrresnet34_gpu(self, input_tensor):
        model = SRResNet34().cuda()
        inp = input_tensor.cuda()
        output = model(inp)
        assert output["pred"].shape == inp.shape

    def test_srrresnet18_complex_gpu(self, input_tensor):
        model = SRResNet18Complex().cuda()
        inp = input_tensor.cuda()
        output = model(inp)
        assert output["pred"].shape == inp.shape
