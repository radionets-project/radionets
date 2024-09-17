from math import pi

import torch
from torch import nn

from radionets.dl_framework.model import GeneralRelu, SRBlock


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

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2))
        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        # s = x.shape[-1]
        means = x.mean(axis=-1).mean(axis=-1).reshape(x.shape[0], x.shape[1], 1, 1)
        stds = x.std(axis=-1).std(axis=-1).reshape(x.shape[0], x.shape[1], 1, 1)

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        # x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        # x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        # x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        # x0 = self.relu(x0)
        # x1 = self.hardtanh(x[:, 1]).reshape(-1, 1, s // 2 + 1, s)

        return (x, means, stds)  # torch.cat([x0, x1], dim=1)


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
        self.elu = GeneralRelu(sub=-1e-10)

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
