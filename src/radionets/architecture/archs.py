from math import pi

import torch
from torch import nn

from .activation import GeneralELU, GeneralReLU
from .blocks import ComplexSRBlock, SRBlock
from .layers import (
    ComplexConv2d,
    ComplexInstanceNorm2d,
    ComplexPReLU,
)

__all__ = [
    "SRResNet",
    "SRResNetComplex",
    "SRResNet18",
    "SRResNet18Complex",
    "SRResNet18AmpPhase",
    "SRResNet34",
    "SRResNet34AmpPhase",
    "SRResNet34_unc",
    "SRResNet34_unc_no_grad",
]


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 64

        self.pre_block = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=self.channels,
                kernel_size=9,
                stride=1,
                padding=4,
                groups=2,
            ),
            nn.PReLU(),
        )

        self.post_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(self.channels),
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=2,
                kernel_size=9,
                stride=1,
                padding=4,
                groups=2,
            ),
        )

    def _create_blocks(self, n_blocks, **kwargs):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(SRBlock(64, 64, **kwargs))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.pre_block(input)
        x = x + self.post_block(self.blocks(x))
        x = self.final(x)

        return {"pred": x}


class SRResNetComplex(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 128

        self.pre_block = nn.Sequential(
            ComplexConv2d(
                in_channels=2,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
            ),
            ComplexPReLU(num_parameters=2),
        )

        self.post_block = nn.Sequential(
            ComplexConv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ComplexInstanceNorm2d(self.channels),
        )

        self.final = nn.Sequential(
            ComplexConv2d(
                in_channels=self.channels,
                out_channels=2,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
        )

    def _create_blocks(self, n_blocks, **kwargs):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ComplexSRBlock(self.channels, self.channels, **kwargs))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.pre_block(input)
        x = x + self.post_block(self.blocks(x))
        x = self.final(x)

        return {"pred": x}


class SRResNet18(SRResNet):
    def __init__(self, **kwargs):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8, **kwargs)


class SRResNet18Complex(SRResNetComplex):
    def __init__(self, **kwargs):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8, **kwargs)


class SRResNet18AmpPhase(SRResNet):
    def __init__(self, **kwargs):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8, **kwargs)

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = super().forward(input)["pred"]

        amp = self.relu(out[:, 0].unsqueeze(1))
        phase = self.hardtanh(out[:, 1].unsqueeze(1))

        return {"pred": torch.cat([amp, phase], dim=1)}


class SRResNet34(SRResNet):
    def __init__(self, **kwargs):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
        self._create_blocks(16, **kwargs)


class SRResNet34AmpPhase(SRResNet):
    def __init__(self, **kwargs):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
        self._create_blocks(16, **kwargs)

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = super().forward(input)["pred"]

        amp = self.relu(out[:, 0].unsqueeze(1))
        phase = self.hardtanh(out[:, 1].unsqueeze(1))

        return {"pred": torch.cat([amp, phase], dim=1)}


class SRResNet34_unc(SRResNet):
    def __init__(self, **kwargs):
        super().__init__()

        self._create_blocks(16, **kwargs)

        self.post_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(self.channels),
        )

        self.grelu = GeneralReLU(sub=-1e-10)

    def forward(self, input):
        s = input.shape[-1]

        x = self.pre_block(input)

        x = x + self.post_block(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        x3 = self.grelu(x3)
        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        x4 = self.grelu(x4)

        return {"pred": torch.cat([x0, x3, x1, x4], dim=1)}


class SRResNet34_unc_no_grad(SRResNet34_unc):
    def __init__(self):
        super().__init__()

        self.elu = GeneralELU()

    def forward(self, input):
        s = input.shape[-1]

        x = self.pre_block(input)

        x = x + self.post_block(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x3 = self.elu(x3)

        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x4 = self.elu(x4)

        return {"pred": torch.cat([x0, x3, x1, x4], dim=1)}
