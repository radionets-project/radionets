import torch
from torch import nn
from radionets.dl_framework.model import (
    SRBlock,
    Lambda,
    symmetry,
)
from functools import partial


class SRResNet_small(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 4
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, 63, 63)
        x1 = self.symmetry_imag(x[:, 1]).reshape(-1, 1, 63, 63)

        return torch.cat([x0, x1], dim=1)


class SRResNet_bigger(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 4
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, 63, 63)
        x1 = self.symmetry_imag(x[:, 1]).reshape(-1, 1, 63, 63)

        return torch.cat([x0, x1], dim=1)
