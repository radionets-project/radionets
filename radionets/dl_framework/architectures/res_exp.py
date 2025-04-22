from math import pi

import torch
from torch import nn

from radionets.dl_framework.model import GeneralRelu, SRBlock, BottleneckResBlock


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
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.relu(x[:, 0].unsqueeze(1))
        x1 = self.hardtanh(x[:, 1].unsqueeze(1))

        return torch.cat([x0, x1], dim=1)


class SRResNetAmp(nn.Module):
    def __init__(self, dropout=False, eval=False, dropout_p=0.5):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
        )

        # ResBlock 8
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4, groups=1),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x[:, 0].unsqueeze(1)

        x = self.preBlock(x0)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)
        x = x0 + x

        # x = self.relu(x)

        return x


class UNetSRResNetAmp_32(nn.Module):
    def __init__(self):
        super(UNetSRResNetAmp_32, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
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
            nn.BatchNorm2d(64),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=4, groups=1),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x  # [:, 0].unsqueeze(1)

        x = self.preBlock(x0)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.relu(x)

        return x


class UNetSRResNetAmp(nn.Module):
    def __init__(self):
        super(UNetSRResNetAmp, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
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
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=4, groups=1),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x  # [:, 0].unsqueeze(1)

        x = self.preBlock(x0)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)
        x = x0 + x

        x = self.relu(x)

        return x


class SRResNetPhase(nn.Module):
    def __init__(self, dropout=True, eval=False, dropout_p=0.2):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1), nn.PReLU()
        )

        # ResBlock 8
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=0.1),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4, groups=1),
        )

        # self.hardtanh = nn.Hardtanh(-pi, pi)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x[:, 1].unsqueeze(1)
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        # x = self.hardtanh(x)
        x = self.tanh(x) * pi

        return x


class SRResNet_16(nn.Module):
    def __init__(self, dropout=False, eval=False, dropout_p=0.5):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2))

        self.prelu = nn.PReLU()
        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.prelu(x[:, 0].unsqueeze(1))
        x1 = self.hardtanh(x[:, 1].unsqueeze(1))

        return torch.cat([x0, x1], dim=1)


class SRResNet_16_Legacy(nn.Module):
    def __init__(self, dropout=False, eval=False, dropout_p=0.5):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2))

        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.prelu(x[:, 0].unsqueeze(1))
        x1 = self.prelu(x[:, 1].unsqueeze(1))

        return torch.cat([x0, x1], dim=1)


class SRResNetAmp_16(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1), nn.PReLU()
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
            nn.Conv2d(64, 64, 9, stride=1, padding=4, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, stride=1, padding=1, groups=1))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.relu(x)

        return x


class SRResNetAmp_32(nn.Module):
    def __init__(self, dropout=False, eval=False, dropout_p=0.8):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=4, bias=False),
            nn.InstanceNorm2d(64, eps=0.1),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, stride=1, padding=1, groups=1))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.relu(x)

        return x


class SRResNetPhase_16(nn.Module):
    def __init__(self, dropout=False, eval=False, dropout_p=0.5):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 1, 9, stride=1, padding=4, groups=1))

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        x0 = x[:, 1].unsqueeze(1)
        x = self.preBlock(x0)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.hardtanh(x)
        x = x + x0

        return x


class PhaseSRResNet56(nn.Module):
    def __init__(self, dropout=False, eval=False, dropout_p=0.5):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 1, 9, stride=1, padding=4, groups=1))

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        x0 = x[:, 1].unsqueeze(1)
        x = self.preBlock(x0)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.hardtanh(x)
        # x = x + x0

        return x


class SRResNetPhase_32(nn.Module):
    def __init__(self, dropout=True, eval=False, dropout_p=0.8):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
            SRBlock(64, 64, dropout=dropout, eval=eval, dropout_p=dropout_p),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 9, stride=1, padding=4, bias=False),
            nn.InstanceNorm2d(64, eps=0.1),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, stride=1, padding=1, groups=1))

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        x0 = x[:, 1].unsqueeze(1)
        x = self.preBlock(x0)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.hardtanh(x)
        # x = x + x0

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


class SRResNet_16_unc_no_grad(nn.Module):
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
        with torch.no_grad():
            x3 = self.elu(x3)

        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x4 = self.elu(x4)

        return torch.cat([x0, x3, x1, x4], dim=1)


class PhaseNet50(nn.Module):
    def __init__(self):
        super(PhaseNet50, self).__init__()

        self.pre_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False, groups=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

        self.stage1 = self._make_stage(
            in_channels=64,
            out_channels=256,
            num_blocks=3,
            stride=1,
        )
        self.stage2 = self._make_stage(
            in_channels=256,
            out_channels=512,
            num_blocks=4,
            stride=1,
        )
        self.stage3 = self._make_stage(
            in_channels=512,
            out_channels=1024,
            num_blocks=6,
            stride=1,
        )
        self.stage4 = self._make_stage(
            in_channels=1024,
            out_channels=2048,
            num_blocks=3,
            stride=1,
        )

        self.final_block = nn.Sequential(nn.Conv2d(2048, 1, kernel_size=1))

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        downsample = False
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(
            BottleneckResBlock(
                in_channels, out_channels, stride=stride, downsample=downsample
            )
        )

        for _ in range(num_blocks - 1):
            layers.append(BottleneckResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x[:, 1].unsqueeze(1)
        x = self.pre_block(x0)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_block(x)

        x = self.hardtanh(x)

        return x


class MiniPhaseNet21(nn.Module):
    def __init__(self):
        super(MiniPhaseNet21, self).__init__()

        self.pre_block = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4, groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.stage1 = self._make_stage(
            in_channels=64,
            out_channels=256,
            num_blocks=2,
            stride=1,
        )
        self.stage2 = self._make_stage(
            in_channels=256,
            out_channels=512,
            num_blocks=2,
            stride=1,
        )
        self.stage3 = self._make_stage(
            in_channels=512,
            out_channels=1024,
            num_blocks=2,
            stride=1,
        )
        self.stage4 = self._make_stage(
            in_channels=1024,
            out_channels=2048,
            num_blocks=1,
            stride=1,
        )

        self.final_block = nn.Sequential(nn.Conv2d(2048, 1, kernel_size=1))

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        downsample = False
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(
            BottleneckResBlock(
                in_channels, out_channels, stride=stride, downsample=downsample
            )
        )

        for _ in range(num_blocks - 1):
            layers.append(BottleneckResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x[:, 1].unsqueeze(1)
        x = self.pre_block(x0)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.final_block(x)

        x = self.hardtanh(x)

        return x
