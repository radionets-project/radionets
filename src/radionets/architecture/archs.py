from math import pi

import torch
from torch import nn

from radionets.architecture.activation import GeneralReLU
from radionets.architecture.blocks import SRBlock


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
            nn.BatchNorm2d(self.channels),
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

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def _create_blocks(self, n_blocks):
        blocks = []
        for i in range(n_blocks):
            blocks.append(SRBlock(64, 64))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.relu(x[:, 0].unsqueeze(1))
        x1 = self.hardtanh(x[:, 1].unsqueeze(1))

        return torch.cat([x0, x1], dim=1)


class SRResNet_18(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8)


class SRResNet_34(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
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


class SRResNet_34_unc(SRResNet):
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

        return torch.cat([x0, x3, x1, x4], dim=1)


class SRResNet_34_unc_no_grad(SRResNet):
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
        with torch.no_grad():
            x3 = self.elu(x3)

        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x4 = self.elu(x4)

        return torch.cat([x0, x3, x1, x4], dim=1)
