from math import pi

import torch
from torch import nn

from radionets.dl_framework.model import GeneralELU, SRBlock


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 8
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
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),
        )

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s, s)
        x0 = self.relu(x0)
        x1 = self.hardtanh(x[:, 1]).reshape(-1, 1, s, s)

        return torch.cat([x0, x1], dim=1)


class SRResNet_16(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1), nn.ReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
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
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 64, 9, stride=1, padding=4, groups=1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # s = x.shape[-1]
        print("\nAnfang", (torch.isnan(x).sum() / x.reshape(-1).shape[0]).item())
        x = x[:, 1].unsqueeze(1)

        x = self.preBlock(x)
        print("\nPre Block", (torch.isnan(x).sum() / x.reshape(-1).shape[0]).item())

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1])

        x = self.softmax(x)

        return x


class SRResNet_16_unc(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
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
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 4, 9, stride=1, padding=4, groups=2))

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()
        self.elu = GeneralELU(add=+(1 + 1e-7))

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        # x0 = self.relu(x0)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        # x1 = self.hardtanh(x1)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        x3 = self.elu(x3)
        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        x4 = self.elu(x4)

        return torch.cat([x0, x3, x1, x4], dim=1)
