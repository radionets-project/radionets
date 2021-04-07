import torch
from torch import nn
from radionets.dl_framework.model import (
    SRBlock,
    Lambda,
    symmetry,
    GeneralELU,
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
            SRBlock(64, 64), SRBlock(64, 64), SRBlock(64, 64), SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),)

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

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),)

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, s, s)
        x1 = self.symmetry_imag(x[:, 1]).reshape(-1, 1, s, s)

        return torch.cat([x0, x1], dim=1)


class SRResNet_unc(nn.Module):
    def __init__(self):
        super().__init__()

        n_channel = 64

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, n_channel, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 4
        self.blocks = nn.Sequential(
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_channel),
        )

        self.final = nn.Sequential(
            nn.Conv2d(n_channel, 4, 9, stride=1, padding=4, groups=2),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

        self.elu = GeneralELU(add=+(1 + 1e-5))

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, s, s)
        x0_unc = self.symmetry_amp(x[:, 1]).reshape(-1, 1, s, s)
        x0_unc = self.elu(x0_unc)
        x1 = self.symmetry_imag(x[:, 2]).reshape(-1, 1, s, s)
        x1_unc = self.symmetry_amp(x[:, 3]).reshape(-1, 1, s, s)
        x1_unc = self.elu(x1_unc)
        return torch.cat([x0, x0_unc, x1, x1_unc], dim=1)


class SRResNet_unc_amp(nn.Module):
    def __init__(self):
        super().__init__()

        n_channel = 56

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, n_channel, 9, stride=1, padding=4, groups=1), nn.PReLU()
        )

        # ResBlock 4
        self.blocks = nn.Sequential(
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_channel),
        )

        self.final = nn.Sequential(
            nn.Conv2d(n_channel, 2, 9, stride=1, padding=4, groups=1),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

        self.elu = GeneralELU(add=+(1 + 1e-5))

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x[:, 0].unsqueeze(1))

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, s, s)
        x0_unc = self.symmetry_amp(x[:, 1]).reshape(-1, 1, s, s)
        x0_unc = self.elu(x0_unc)
        return torch.cat([x0, x0_unc], dim=1)


class SRResNet_unc_phase(nn.Module):
    def __init__(self):
        super().__init__()

        n_channel = 56

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, n_channel, 9, stride=1, padding=4, groups=1), nn.PReLU()
        )

        # ResBlock 4
        self.blocks = nn.Sequential(
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
            SRBlock(n_channel, n_channel),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_channel),
        )

        self.final = nn.Sequential(
            nn.Conv2d(n_channel, 2, 9, stride=1, padding=4, groups=1),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

        self.elu = GeneralELU(add=+(1 + 1e-10))

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x[:, 1].unsqueeze(1))

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_imag(x[:, 0]).reshape(-1, 1, s, s)
        x0_unc = self.symmetry_amp(x[:, 1]).reshape(-1, 1, s, s)
        x0_unc = self.elu(x0_unc)
        return torch.cat([x0, x0_unc], dim=1)
