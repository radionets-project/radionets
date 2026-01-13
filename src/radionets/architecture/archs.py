from math import pi

import torch
from torch import nn

from radionets.architecture.activation import GeneralReLU
from radionets.architecture.blocks import ComplexSRBlock, SRBlock
from radionets.architecture.layers import (
    ComplexConv2d,
    ComplexInstanceNorm2d,
    ComplexPReLU,
)

__all__ = [
    "SRResNet",
    "SRResNetComplex",
    "SRResNetAmp18",
    "SRResNetPhase18",
    "SRResNetImag10",
    "SRResNetImag",
    "SRResNet8",
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

        self.preBlock = nn.Sequential(
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

        self.postBlock = nn.Sequential(
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

    def _create_blocks(self, n_blocks):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(SRBlock(64, 64))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.preBlock(input)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)

        return {"pred": x}


class SRResNetSingleChannel(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 64

        self.preBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.channels,
                kernel_size=9,
                stride=1,
                padding=4,
                groups=1,
            ),
            nn.PReLU(),
        )

        self.postBlock = nn.Sequential(
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
                out_channels=1,
                kernel_size=9,
                stride=1,
                padding=4,
                groups=1,
            ),
        )

        self.prelu = nn.PReLU()
        self.hardtanh = nn.Hardtanh(-pi, pi)

    def _create_blocks(self, n_blocks):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(SRBlock(64, 64, groups=1))

        self.blocks = nn.Sequential(*blocks)


class SRResNetAmp18(SRResNetSingleChannel):
    def __init__(self):
        super().__init__()
        self._create_blocks(8)

    def forward(self, inputs):
        x = inputs[:, 0, ...][:, None]
        x = self.preBlock(x)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)
        x = self.prelu(x)

        x = torch.cat((x, inputs[:, 1, ...][:, None]), dim=1)

        return {"pred": x}


class SRResNetPhase18(SRResNetSingleChannel):
    def __init__(self):
        super().__init__()
        self._create_blocks(8)

    def forward(self, inputs):
        x = inputs[:, 1, ...][:, None]
        x = self.preBlock(x)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)
        x = self.hardtanh(x)

        x = torch.cat((inputs[:, 0, ...][:, None], x), dim=1)

        return {"pred": x}


class SRResNetImag10(SRResNetSingleChannel):
    def __init__(self):
        super().__init__()
        self._create_blocks(4)

    def forward(self, inputs):
        x = inputs[:, 1, ...][:, None]
        x = self.preBlock(x)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)
        x = self.tanhshrink(x)

        x = torch.cat((inputs[:, 0, ...][:, None], x), dim=1)

        return {"pred": x}


class SRResNetImag(SRResNetSingleChannel):
    def __init__(self):
        super().__init__()
        self._create_blocks(8)
        self.tanhshrink = nn.Tanhshrink()

    def forward(self, inputs):
        x = inputs[:, 1, ...][:, None]
        x = self.preBlock(x)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)
        x = self.tanhshrink(x)

        x = torch.cat((inputs[:, 0, ...][:, None], x), dim=1)

        return {"pred": x}


class SRResNetComplex(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 128

        self.preBlock = nn.Sequential(
            ComplexConv2d(
                in_channels=2,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
            ),
            ComplexPReLU(num_parameters=2),
        )

        self.postBlock = nn.Sequential(
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

    def _create_blocks(self, n_blocks):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ComplexSRBlock(64, 64))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.preBlock(input)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)

        return {"pred": x}


class SRResNet8(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 3 ResBlocks to build a SRResNet8
        self._create_blocks(3)


class SRResNet18(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8)


class SRResNet18Complex(SRResNetComplex):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8)


class SRResNet18AmpPhase(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8)

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super().forward(x)

        x_amp = self.relu(x[:, 0].unsqueeze(1))
        x_phase = self.hardtanh(x[:, 1].unsqueeze(1))

        return {"pred": torch.cat([x_amp, x_phase], dim=1)}


class SRResNet34(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
        self._create_blocks(16)


class SRResNet34AmpPhase(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
        self._create_blocks(16)

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super().forward(x)

        x_amp = self.relu(x[:, 0].unsqueeze(1))
        x_phase = self.hardtanh(x[:, 1].unsqueeze(1))

        return {"pred": torch.cat([x_amp, x_phase], dim=1)}


class SRResNet34_unc(SRResNet):
    def __init__(self):
        super().__init__()

        self._create_blocks(16)

        self.postBlock = nn.Sequential(
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

        self.elu = GeneralReLU(sub=-1e-10)

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        x3 = self.elu(x3)
        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        x4 = self.elu(x4)

        return {"pred": torch.cat([x0, x3, x1, x4], dim=1)}


class SRResNet34_unc_no_grad(SRResNet34_unc):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

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
