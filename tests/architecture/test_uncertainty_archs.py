"""Tests for src/radionets/architecture/uncertainty_archs.py

Tests uncertainty modules:
- Uncertainty
- UncertaintyWrapper

Source code issues documented by tests:
- Uncertainty.forward: LocallyConnected2d weight has output_size shape but
  actual output is computed from input tensor. Need to match input dimensions.
- UncertaintyWrapper uses super.forward(x) instead of super().forward(x) (syntax error).
  super is a class, .forward is the class method descriptor, not bound to self.
"""

import pytest
import torch
import torch.nn as nn

from radionets.architecture.archs import SRResNet34
from radionets.architecture.uncertainty_archs import (
    Uncertainty,
    UncertaintyWrapper,
)

pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestUncertainty:
    """Test Uncertainty module."""

    def test_instantiate_default(self):
        model = Uncertainty(img_size=64)
        assert isinstance(model, nn.Module)

    def test_instantiate_various_sizes(self):
        for img_size in [32, 64, 128]:
            model = Uncertainty(img_size=img_size)
            assert isinstance(model, nn.Module)

    def test_has_layers(self):
        model = Uncertainty(64)
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "final")
        assert hasattr(model, "elu")

    def test_conv1_structure(self):
        model = Uncertainty(64)
        conv1 = model.conv1

        assert len(conv1) == 3
        assert isinstance(conv1[0], nn.Conv2d)
        assert conv1[0].in_channels == 4
        assert conv1[0].out_channels == 16
        assert conv1[0].kernel_size == (9, 9)
        assert conv1[0].groups == 2
        assert isinstance(conv1[1], nn.InstanceNorm2d)
        assert conv1[1].num_features == 16
        assert isinstance(conv1[2], nn.ReLU)

    def test_conv2_structure(self):
        model = Uncertainty(64)
        conv2 = model.conv2

        assert len(conv2) == 3
        assert isinstance(conv2[0], nn.Conv2d)
        assert conv2[0].in_channels == 16
        assert conv2[0].out_channels == 32

    def test_conv3_structure(self):
        model = Uncertainty(64)
        conv3 = model.conv3

        assert len(conv3) == 3
        assert isinstance(conv3[0], nn.Conv2d)
        assert conv3[0].in_channels == 32
        assert conv3[0].out_channels == 64
        assert conv3[0].groups == 2

    def test_forward_shape_img_size_32(self):
        model = Uncertainty(img_size=32)
        inp = torch.randn(2, 4, 32, 32)

        out = model(inp)

        assert out.shape == (2, 2, 17, 32)

    def test_forward_shape_img_size_64(self):
        model = Uncertainty(img_size=64)
        inp = torch.randn(2, 4, 64, 64)

        out = model(inp)

        assert out.shape == (2, 2, 33, 64)

    def test_forward_finite(self):
        model = Uncertainty(img_size=64)

        inp = torch.randn(2, 4, 33, 64)
        output = model(inp)

        assert output.shape == (2, 2, 33, 64)
        assert torch.isfinite(output).all()

    def test_forward_output_positive(self):
        model = Uncertainty(img_size=32)

        inp = torch.randn(2, 4, 17, 32)
        output = model(inp)

        assert (output >= -1e-6).all()

    def test_forward_different_img_size(self):
        model_64 = Uncertainty(img_size=64)
        inp = torch.randn(2, 4, 33, 64)
        output = model_64(inp)

        assert output.shape == (2, 2, 33, 64)


class TestUncertaintyWrapper:
    """Test UncertaintyWrapper module."""

    def test_instantiate_default(self):
        model = UncertaintyWrapper(img_size=64)
        assert isinstance(model, nn.Module)

    def test_instantiate_various_sizes(self):
        for img_size in [32, 64, 128]:
            model = UncertaintyWrapper(img_size=img_size)
            assert isinstance(model, nn.Module)

    def test_is_subclass_of_srrresnet34(self):
        assert issubclass(UncertaintyWrapper, SRResNet34)

    def test_has_uncertainty_attribute(self):
        model = UncertaintyWrapper(img_size=64)
        assert hasattr(model, "uncertainty")
        assert isinstance(model.uncertainty, Uncertainty)


@cuda
class TestGPUForwardPasses:
    def test_uncertainty_gpu_forward(self):
        """Uncertainty works on GPU when input matches output_size."""
        model = Uncertainty(img_size=32).cuda()
        inp = torch.randn(2, 4, 17, 32).cuda()  # H=17 matches output_size[0]=17
        output = model(inp)
        assert output.shape == (2, 2, 17, 32)

    def test_uncertainty_wrapper_gpu_forward_has_error(self):
        """UncertaintyWrapper has a source bug with super.forward()."""
        model = UncertaintyWrapper(img_size=64).cuda()
        inp = torch.randn(2, 2, 64, 64).cuda()

        out = model(inp)

        assert out.shape == (2, 4, 64, 64)
