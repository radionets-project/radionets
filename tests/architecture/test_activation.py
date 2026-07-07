"""Tests for src/radionets/architecture/activation.py"""

import torch
import torch.nn.functional as F

from radionets.architecture.activation import (
    GeneralELU,
    GeneralReLU,
)


class TestGeneralReLU:
    """Test GeneralReLU activation function."""

    def test_default_is_standard_relu(self):
        """No parameters should behave like standard ReLU."""
        inp = torch.randn(2, 4, 8, 8)

        act = GeneralReLU()
        output = act(inp)

        expected = F.relu(inp)

        assert torch.allclose(output, expected)

    def test_leak_param(self):
        """leak parameter should produce LeakyReLU behavior."""
        inp = torch.randn(2, 4, 8, 8)

        act = GeneralReLU(leak=0.01)
        output = act(inp)

        expected = F.leaky_relu(inp, negative_slope=0.01)

        assert torch.allclose(output, expected)

    def test_sub_param(self):
        """sub parameter should subtract from output."""
        inp = torch.full((2, 4, 8, 8), 10.0)

        act = GeneralReLU(sub=0.5)
        output = act(inp)

        expected = F.relu(inp) - 0.5

        assert torch.allclose(output, expected)

    def test_sub_with_leak(self):
        """sub should apply after leaky_relu."""
        inp = torch.full((2, 4, 8, 8), 10.0)

        act = GeneralReLU(leak=0.1, sub=1.0)
        output = act(inp)

        expected = F.leaky_relu(inp, negative_slope=0.1) - 1.0

        assert torch.allclose(output, expected)

    def test_maxv_param(self):
        """maxv parameter should clamp maximum values."""
        inp = torch.full((2, 4, 8, 8), 100.0)

        act = GeneralReLU(maxv=5.0)
        output = act(inp)

        assert (output <= 5.0).all()
        assert (output[:, :1, :1, :1] == 5.0).all()

    def test_maxv_with_positive(self):
        """maxv should not affect values below the max."""
        inp = torch.full((2, 4, 8, 8), 3.0)

        act = GeneralReLU(maxv=5.0)
        output = act(inp)

        assert torch.allclose(output, inp)

    def test_all_params_combined(self):
        """All parameters combined: leak -> sub -> maxv."""
        inp = torch.randn(2, 4, 8, 8) * 10.0

        act = GeneralReLU(leak=0.1, sub=1.0, maxv=5.0)
        output = act(inp)

        assert (output <= 5.0).all()

    def test_forward_shape(self):
        """Output shape should match input shape."""
        act = GeneralReLU(leak=0.01, sub=0.5, maxv=10.0)

        for shape in [(2, 2, 16, 16), (4, 8, 8, 8), (1, 1, 64, 64)]:
            inp = torch.randn(*shape)
            output = act(inp)

            assert output.shape == shape


class TestGeneralELU:
    """Test GeneralELU activation function."""

    def test_default_is_standard_elu(self):
        """No parameters should behave like standard ELU."""
        inp = torch.randn(2, 4, 8, 8)

        act = GeneralELU()
        output = act(inp)

        expected = F.elu(inp)

        assert torch.allclose(output, expected)

    def test_add(self):
        """add parameter should add to ELU output."""
        inp = torch.full((2, 4, 8, 8), 10.0)

        act = GeneralELU(add=0.5)
        output = act(inp)

        expected = F.elu(inp) + 0.5

        assert torch.allclose(output, expected)

    def test_maxv(self):
        """maxv parameter should clamp maximum values."""
        inp = torch.full((2, 4, 8, 8), 100.0)

        layer = GeneralELU(maxv=5.0)
        output = layer(inp)

        assert (output <= 5.0).all()

    def test_maxv_with_elu_negative(self):
        inp = torch.ones(2, 4, 8, 8) * -5.0

        act = GeneralELU(maxv=5.0)
        output = act(inp)

        expected = F.elu(inp)

        assert torch.allclose(output, expected)

    def test_add_and_maxv(self):
        inp = torch.randn(2, 4, 8, 8) * 10.0

        act = GeneralELU(add=1.0, maxv=5.0)
        output = act(inp)

        assert (output <= 5.0).all()

    def test_forward_shape(self):
        """Output shape should match input shape."""
        act = GeneralELU(add=0.5, maxv=10.0)

        for shape in [(2, 2, 16, 16), (4, 8, 8, 8), (1, 1, 64, 64)]:
            inp = torch.randn(*shape)
            output = act(inp)

            assert output.shape == shape
