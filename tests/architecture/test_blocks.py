"""Tests for src/radionets/architecture/blocks.py"""

import pytest
import torch
import torch.nn as nn

from radionets.architecture.blocks import (
    ComplexSRBlock,
    NNBlock,
    SRBlock,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def two_channel_tensor():
    return torch.randn(2, 2, 16, 16)


class TestInstantiation:
    @pytest.mark.parametrize(
        "cls, in_ch, out_ch",
        [
            (SRBlock, 64, 64),
            (SRBlock, 64, 128),
            (ComplexSRBlock, 64, 64),
            (ComplexSRBlock, 64, 128),
        ],
    )
    def test_instantiate(self, cls, in_ch, out_ch):
        block = cls(in_ch, out_ch)
        assert isinstance(block, nn.Module)

    @pytest.mark.parametrize(
        "cls, in_ch, out_ch",
        [
            (SRBlock, 64, 64),
            (ComplexSRBlock, 64, 64),
        ],
    )
    def test_instantiate_with_dropout(self, cls, in_ch, out_ch):
        block = cls(in_ch, out_ch, dropout=0.1)
        assert isinstance(block, nn.Module)


class TestSRBlock:
    def test_srrresnet_block_has_convs_sequential(self):
        block = SRBlock(64, 64)

        assert hasattr(block, "convs")
        assert isinstance(block.convs, nn.Sequential)

    def test_srrresnet_block_convs_count(self):
        block = SRBlock(64, 64)
        assert len(block.convs) == 7

    def test_srrresnet_block_convs_types(self):
        block = SRBlock(64, 64)
        convs = block.convs

        assert isinstance(convs[0], nn.Conv2d)
        assert isinstance(convs[1], nn.Dropout)
        assert isinstance(convs[2], nn.InstanceNorm2d)
        assert isinstance(convs[3], nn.PReLU)
        assert isinstance(convs[4], nn.Conv2d)
        assert isinstance(convs[5], nn.Dropout)
        assert isinstance(convs[6], nn.InstanceNorm2d)

    def test_srrresnet_block_convs_channel_count_same(self):
        block = SRBlock(64, 64)

        convs = block.convs
        assert convs[0].in_channels == 64
        assert convs[0].out_channels == 64
        assert convs[2].num_features == 64
        assert convs[4].in_channels == 64
        assert convs[4].out_channels == 64
        assert convs[6].num_features == 64

    def test_srrresnet_block_convs_channel_count_diff(self):
        block = SRBlock(32, 64)
        convs = block.convs

        assert convs[0].in_channels == 32
        assert convs[0].out_channels == 64

    def test_srrresnet_block_dropout_config(self):
        block = SRBlock(64, 64, dropout=0.1)

        assert isinstance(block.convs[1], nn.Dropout)
        assert block.convs[1].p == 0.1
        assert isinstance(block.convs[5], nn.Dropout)
        assert block.convs[5].p == 0.1

    def test_srrresnet_block_dropout_zero(self):
        block = SRBlock(64, 64, dropout=0)

        assert isinstance(block.convs[1], nn.Dropout)
        assert block.convs[1].p == 0

    def test_srrresnet_block_conv_padding_mode(self):
        block = SRBlock(64, 64)

        assert block.convs[0].padding_mode == "reflect"
        assert block.convs[4].padding_mode == "reflect"

    def test_srrresnet_block_conv_bias_false(self):
        block = SRBlock(64, 64)

        assert block.convs[0].bias is None
        assert block.convs[4].bias is None

    def test_srrresnet_block_forward_shape_equal_channels(self, two_channel_tensor):
        block = SRBlock(2, 2)
        output = block(two_channel_tensor)
        assert output.shape == two_channel_tensor.shape

    def test_srrresnet_block_forward_finite(self, two_channel_tensor):
        block = SRBlock(2, 2)
        output = block(two_channel_tensor)

        assert torch.isfinite(output).all()

    def test_srrresnet_block_forward_with_dropout(self):
        block = SRBlock(64, 64, dropout=0.5)
        inp = torch.randn(2, 64, 8, 8)
        output = block(inp)

        assert output.shape == inp.shape
        assert torch.isfinite(output).all()

    def test_srrresnet_block_idconv_identity(self):
        block = SRBlock(64, 64)
        assert isinstance(block.idconv, nn.Identity)

    def test_srrresnet_block_idconv_conv(self):
        block = SRBlock(64, 128)

        assert isinstance(block.idconv, nn.Conv2d)
        assert block.idconv.in_channels == 64
        assert block.idconv.out_channels == 128


class TestComplexSRBlock:
    def test_complex_srrresnet_block_has_convs_sequential(self):
        block = ComplexSRBlock(64, 64)

        assert hasattr(block, "convs")
        assert isinstance(block.convs, nn.Sequential)

    def test_complex_srrresnet_block_convs_count(self):
        block = ComplexSRBlock(64, 64)
        assert len(block.convs) == 7

    def test_complex_srrresnet_block_convs_types(self):
        from radionets.architecture.layers import (
            ComplexConv2d,
            ComplexInstanceNorm2d,
            ComplexPReLU,
        )

        block = ComplexSRBlock(64, 64)
        convs = block.convs

        assert isinstance(convs[0], ComplexConv2d)
        assert isinstance(convs[1], nn.Dropout)
        assert isinstance(convs[2], ComplexInstanceNorm2d)
        assert isinstance(convs[3], ComplexPReLU)
        assert isinstance(convs[4], ComplexConv2d)
        assert isinstance(convs[5], nn.Dropout)
        assert isinstance(convs[6], ComplexInstanceNorm2d)

    def test_complex_srrresnet_block_forward_shape_equal_channels(
        self, two_channel_tensor
    ):
        block = ComplexSRBlock(2, 2)
        output = block(two_channel_tensor)

        assert output.shape == two_channel_tensor.shape

    def test_complex_srrresnet_block_forward_finite(self, two_channel_tensor):
        block = ComplexSRBlock(2, 2)
        output = block(two_channel_tensor)

        assert torch.isfinite(output).all()


class TestNNBlock:
    def test_nnb_stores_attributes(self):
        block = NNBlock(
            64, 128, kernel_size=5, stride=2, padding=3, groups=4, dropout=0.5
        )

        assert block.in_channels == 64
        assert block.out_channels == 128
        assert block.kernel_size == 5
        assert block.stride == 2
        assert block.padding == 3
        assert block.groups == 4
        assert block.dropout == 0.5

    def test_nnb_idconv_identity(self):
        block = NNBlock(64, 64)
        assert isinstance(block.idconv, nn.Identity)

    def test_nnb_idconv_conv(self):
        block = NNBlock(64, 128)
        assert isinstance(block.idconv, nn.Conv2d)

    def test_nnb_pool_identity(self):
        block = NNBlock(64, 64, stride=1)
        assert isinstance(block.pool, nn.Identity)

    def test_nnb_pool_avgpool(self):
        block = NNBlock(64, 64, stride=2)
        assert isinstance(block.pool, nn.AvgPool2d)


@cuda
class TestGPUForwardPasses:
    def test_srrresnet_block_gpu(self, two_channel_tensor):
        block = SRBlock(2, 2).cuda()
        output = block(two_channel_tensor.cuda())

        assert output.shape == two_channel_tensor.shape
        assert torch.isfinite(output).all()

    def test_complex_srrresnet_block_gpu(self, two_channel_tensor):
        block = ComplexSRBlock(2, 2).cuda()
        output = block(two_channel_tensor.cuda())

        assert output.shape == two_channel_tensor.shape
        assert torch.isfinite(output).all()
