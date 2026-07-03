"""Tests for src/radionets/architecture/layers.py"""

import pytest
import torch

from radionets.architecture.layers import (
    ComplexConv2d,
    ComplexInstanceNorm2d,
    ComplexPReLU,
    LocallyConnected2d,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestLocallyConnected2d:
    def test_weight_shape(self):
        layer = LocallyConnected2d(
            in_channels=2,
            out_channels=4,
            output_size=(8, 8),
            kernel_size=3,
            stride=1,
        )
        expected_shape = (1, 4, 2, 8, 8, 9)  # kernel ** 2 = 3 * 3 = 9

        assert layer.weight.shape == expected_shape

    def test_weight_shape_large_kernel(self):
        layer = LocallyConnected2d(2, 8, (16, 16), 5, 2)
        expected_shape = (1, 8, 2, 16, 16, 25)

        assert layer.weight.shape == expected_shape

    def test_bias_shape(self):
        layer = LocallyConnected2d(2, 4, (8, 8), 3, 1, bias=True)

        assert layer.bias is not None
        assert layer.bias.shape == (1, 4, 8, 8)

    def test_bias_false(self):
        layer = LocallyConnected2d(2, 4, (8, 8), 3, 1, bias=False)

        assert layer.bias is None

    def test_forward_shape_input_output(self):
        layer = LocallyConnected2d(2, 4, (8, 8), 3, 1, bias=True)

        # Input (2, 2, 10, 10) -> output (2, 4, 8, 8)
        inp = torch.randn(2, 2, 10, 10)
        output = layer(inp)

        assert output.shape == (2, 4, 8, 8)

    def test_forward_no_bias(self):
        layer = LocallyConnected2d(2, 4, (8, 8), 3, 1, bias=False)
        inp = torch.randn(2, 2, 10, 10)
        output = layer(inp)

        assert output.shape == (2, 4, 8, 8)

    def test_forward_stride(self):
        layer = LocallyConnected2d(2, 4, (7, 7), 3, 2)
        inp = torch.randn(2, 2, 15, 15)
        output = layer(inp)

        assert output.shape == (2, 4, 7, 7)

    def test_forward_finite(self):
        layer = LocallyConnected2d(2, 4, (8, 8), 3, 1)
        inp = torch.randn(2, 2, 10, 10)
        output = layer(inp)

        assert torch.isfinite(output).all()

    def test_forward_size_matches_output(self):
        layer = LocallyConnected2d(2, 4, (10, 10), 1, 1, bias=True)
        inp = torch.randn(2, 2, 10, 10)
        output = layer(inp)

        assert output.shape == (2, 4, 10, 10)


class TestComplexConv2d:
    def test_conv_real_channel_split(self):
        layer = ComplexConv2d(4, 8, 3)

        assert layer.conv_real.in_channels == 2
        assert layer.conv_real.out_channels == 4

    def test_conv_imag_channel_split(self):
        layer = ComplexConv2d(4, 8, 3)

        assert layer.conv_imag.in_channels == 2
        assert layer.conv_imag.out_channels == 4

    def test_conv_bias_true(self):
        layer = ComplexConv2d(4, 8, 3, bias=True)

        assert layer.conv_real.bias is not None
        assert layer.conv_imag.bias is not None

    def test_conv_bias_false(self):
        layer = ComplexConv2d(4, 8, 3, bias=False)

        assert layer.conv_real.bias is None
        assert layer.conv_imag.bias is None

    def test_forward_shape(self):
        layer = ComplexConv2d(2, 4, 3, bias=True)
        inp = torch.randn(2, 2, 16, 16)
        output = layer(inp)

        assert output.shape == (2, 4, 16, 16)

    def test_forward_shape_same_padding(self):
        layer = ComplexConv2d(4, 6, 3, stride=1, padding="same")
        inp = torch.randn(2, 4, 16, 16)
        output = layer(inp)

        assert output.shape == (2, 6, 16, 16)

    def test_forward_shape_stride(self):
        layer = ComplexConv2d(4, 8, 3, stride=2, padding=1)
        inp = torch.randn(2, 4, 16, 16)
        output = layer(inp)

        assert output.shape == (2, 8, 8, 8)

    def test_forward_finite(self):
        layer = ComplexConv2d(2, 4, 3)
        inp = torch.randn(2, 2, 16, 16)
        output = layer(inp)

        assert torch.isfinite(output).all()

    def test_forward_real_imag_split(self):
        inp = torch.zeros(1, 2, 4, 4)
        inp[:, 0, :, :] = 1.0
        inp[:, 1, :, :] = 1.0

        layer = ComplexConv2d(2, 2, 3, bias=False)
        with torch.no_grad():
            layer.conv_real.weight.fill_(0)
            layer.conv_imag.weight.fill_(0)

        output = layer(inp)
        assert output.shape == (1, 2, 4, 4)


class TestComplexInstanceNorm2d:
    def test_affine_params_exist(self):
        layer = ComplexInstanceNorm2d(4, eps=1e-5, affine=True)

        assert hasattr(layer, "weight_real")
        assert hasattr(layer, "weight_imag")
        assert hasattr(layer, "bias_real")
        assert hasattr(layer, "bias_imag")

    def test_affine_param_shapes(self):
        layer = ComplexInstanceNorm2d(8, affine=True)

        assert layer.weight_real.shape == (4,)
        assert layer.weight_imag.shape == (4,)
        assert layer.bias_real.shape == (4,)
        assert layer.bias_imag.shape == (4,)

    def test_affine_param_initialization(self):
        layer = ComplexInstanceNorm2d(4, affine=True)

        assert torch.allclose(layer.weight_real, torch.ones(2))
        assert torch.allclose(layer.weight_imag, torch.ones(2))
        assert torch.allclose(layer.bias_real, torch.zeros(2))
        assert torch.allclose(layer.bias_imag, torch.zeros(2))

    def test_no_affine_params(self):
        layer = ComplexInstanceNorm2d(4, affine=False)

        assert not hasattr(layer, "weight_real")
        assert not hasattr(layer, "weight_imag")
        assert not hasattr(layer, "bias_real")
        assert not hasattr(layer, "bias_imag")

    def test_num_features_stored(self):
        layer = ComplexInstanceNorm2d(8, affine=True)
        assert layer.num_features == 4

    def test_eps_stored(self):
        layer = ComplexInstanceNorm2d(4, eps=1e-3)
        assert layer.eps == 1e-3

    def test_forward_shape(self):
        layer = ComplexInstanceNorm2d(2, affine=True)
        inp = torch.randn(2, 2, 16, 16)
        output = layer(inp)

        assert output.shape == inp.shape

    def test_forward_normalization(self):
        """Check that normalized output has zero mean per channel."""
        inp = torch.full((2, 4, 8, 8), 3.0)
        layer = ComplexInstanceNorm2d(4, affine=False)

        output = layer(inp)

        real_mean = output[:, :2].mean(dim=[0, 2, 3])
        imag_mean = output[:, 2:].mean(dim=[0, 2, 3])

        assert torch.allclose(real_mean, torch.zeros_like(real_mean), atol=1e-4)
        assert torch.allclose(imag_mean, torch.zeros_like(imag_mean), atol=1e-4)

    def test_forward_no_affine(self):
        layer = ComplexInstanceNorm2d(2, affine=False)
        inp = torch.full((2, 2, 8, 8), 100.0)
        output = layer(inp)

        assert output.shape == (2, 2, 8, 8)


class TestComplexPReLU:
    def test_shared_parameter(self):
        layer = ComplexPReLU(num_parameters=1)

        assert layer.num_parameters == 1
        assert layer.weight_real.numel() == 1
        assert layer.weight_imag.numel() == 1

    def test_per_channel_parameter(self):
        layer = ComplexPReLU(num_parameters=4)

        assert layer.num_parameters == 4
        assert layer.weight_real.numel() == 2
        assert layer.weight_imag.numel() == 2

    @pytest.mark.parametrize("init", [0.1, 0.25, 0.5, 1.0])
    def test_init_value(self, init):
        layer = ComplexPReLU(num_parameters=1, init=init)

        assert torch.allclose(layer.weight_real, torch.full((1,), init))

    def test_custom_init_per_channel(self):
        layer = ComplexPReLU(num_parameters=6, init=0.5)

        assert torch.allclose(layer.weight_real, torch.full((3,), 0.5))
        assert torch.allclose(layer.weight_imag, torch.full((3,), 0.5))

    def test_forward_positive_values_unchanged(self):
        layer = ComplexPReLU(num_parameters=1, init=0.0)
        inp = torch.full((2, 2, 4, 4), 10.0)
        output = layer(inp)

        assert torch.allclose(output, inp, atol=1e-4)

    def test_forward_negative_values_scaled(self):
        layer = ComplexPReLU(num_parameters=1, init=0.5)
        inp = torch.full((2, 2, 4, 4), -1.0)
        output = layer(inp)

        assert torch.allclose(output, torch.full_like(inp, -0.5), atol=1e-4)

    def test_forward_finite(self):
        layer = ComplexPReLU(num_parameters=1)
        inp = torch.randn(2, 2, 16, 16)
        output = layer(inp)

        assert torch.isfinite(output).all()


@cuda
class TestGPUForwardPasses:
    def test_locally_connected_gpu(self):
        layer = LocallyConnected2d(2, 4, (8, 8), 3, 1).cuda()
        inp = torch.randn(2, 2, 10, 10).cuda()
        output = layer(inp)

        assert output.shape == (2, 4, 8, 8)
        assert torch.isfinite(output).all()

    def test_complex_conv_gpu(self):
        layer = ComplexConv2d(2, 4, 3).cuda()
        inp = torch.randn(2, 2, 16, 16).cuda()
        output = layer(inp)

        assert output.shape == (2, 4, 16, 16)
        assert torch.isfinite(output).all()

    def test_complex_instance_norm_gpu(self):
        layer = ComplexInstanceNorm2d(2, affine=True).cuda()
        inp = torch.randn(2, 2, 16, 16).cuda()
        output = layer(inp)

        assert output.shape == inp.shape
        assert torch.isfinite(output).all()

    def test_complex_prelu_gpu(self):
        layer = ComplexPReLU(num_parameters=1).cuda()
        inp = torch.randn(2, 2, 16, 16).cuda()
        output = layer(inp)

        assert output.shape == inp.shape
        assert torch.isfinite(output).all()
