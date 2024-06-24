from math import pi

import torch
from torch import nn

from radionets.dl_framework.model import GeneralELU, SRBlock, SRBlock13, SRBlock21, SRBlock29, SRBlock37


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()

        num_in = 32
        self.pre = nn.Conv2d(2, num_in, 3, padding=1, groups=2)

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
            SRBlock(num_in, num_in),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(num_in, num_in, 3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(num_in), nn.PReLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(num_in, 2, 3, stride=1, padding=1, groups=2),
        )


    def forward(self, x):
        x = self.pre(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        return x  # torch.cat([x0, x1], dim=1)


class SRResNet_scales(nn.Module):
    def __init__(self):
        super().__init__()

        num_in = 16
        self.pre = nn.Conv2d(2, num_in, 3, padding=1, groups=2)

        self.preBlock3 = nn.Sequential(
            nn.Conv2d(num_in, num_in, 5, stride=1, padding=2, groups=1), nn.PReLU(), nn.InstanceNorm2d(num_in),
            SRBlock13(num_in,  num_in),
            SRBlock13(num_in,  num_in),
        )
        self.preBlock5 = nn.Sequential(
            nn.Conv2d(num_in, num_in, 11, stride=1, padding=5, groups=1), nn.PReLU(), nn.InstanceNorm2d(num_in),
            SRBlock21(num_in,  num_in),
            SRBlock21(num_in,  num_in),
        )
        self.preBlock7 = nn.Sequential(
            nn.Conv2d(num_in, num_in, 15, stride=1, padding=7, groups=1), nn.PReLU(), nn.InstanceNorm2d(num_in),
            SRBlock29(num_in,  num_in),
            SRBlock29(num_in,  num_in),
        )
        self.preBlock9 = nn.Sequential(
            nn.Conv2d(num_in, num_in, 21, stride=1, padding=10, groups=1), nn.PReLU(), nn.InstanceNorm2d(num_in),
            SRBlock37(num_in,  num_in),
            SRBlock37(num_in,  num_in),
        )

        # ResBlock 4
        self.blocks = nn.Sequential(
            SRBlock(num_in*4, num_in*4),
            SRBlock(num_in*4, num_in*4),
            SRBlock(num_in*4, num_in*4),
            SRBlock(num_in*4, num_in*4),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(num_in*4, num_in*4, 3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(num_in*4), nn.PReLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(num_in*4, 2, 3, stride=1, padding=1, groups=2),
        )

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        inp = x
        x = self.pre(x)

        x1 = self.preBlock3(x)
        x2 = self.preBlock5(x)
        x3 = self.preBlock7(x)
        x4 = self.preBlock9(x)

        x_pre = torch.repeat_interleave(x, 4, dim=1) + torch.cat([x1, x2, x3, x4], dim=1)

        x = x_pre + self.postBlock(self.blocks(x_pre))

        x = self.final(x)

        x0 = self.relu(x[:, 0])
        x1 = self.hardtanh(x[:, 1])
        return x + inp   #torch.cat([x0[:, None], x1[:, None]], dim=1) + inp

class SRResNet_encoded(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(64, 128, 9, stride=1, padding=4, groups=1), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
        )

        self.final = nn.Sequential(nn.Conv2d(128, 64, 9, stride=1, padding=4, groups=1))

        # self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        x = x[:, 64:]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        # x0 = x[:, :128].reshape(-1, 128, s, s)
        #print("x0_max", x0.max())
        #print("x0_min", x0.min())
        # x1 = self.hardtanh(x[:, 128:]).reshape(-1, 128, s, s)
        #print("x1_max", x1.max())
        #print("x1_min", x1.min())

        return x  # torch.cat([x0, x1], dim=1)


class SRResNet_16_unc(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 128, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
            SRBlock(128, 128),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
        )

        self.final = nn.Sequential(nn.Conv2d(128, 4, 9, stride=1, padding=4, groups=2))

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
