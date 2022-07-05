import torch
from torch import nn
from radionets.dl_framework.model import (
    SRBlock,
    Lambda,
    symmetry,
    GeneralELU,
)
from functools import partial
from math import pi


class SRResNet_shuffle(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 14
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
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
        )

        self.shuffle = nn.Sequential(
            nn.Conv2d(64, 252, 3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(3),
            nn.PReLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(28, 2, 9, stride=1, padding=4, groups=2),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))
        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.shuffle(x)

        x = self.final(x)

        s = x.shape[-1]

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, s, s)
        x1 = self.symmetry_imag(x[:, 1]).reshape(-1, 1, s, s)
        x1 = self.hardtanh(x1)

        return torch.cat([x0, x1], dim=1)


class SRResNet_bigger(nn.Module):
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

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))
        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = self.symmetry_amp(x[:, 0]).reshape(-1, 1, s, s)
        x1 = self.hardtanh(x[:, 1]).reshape(-1, 1, s, s)
        x1 = self.symmetry_imag(x1).reshape(-1, 1, s, s)

        return torch.cat([x0, x1], dim=1)


class SRResNet_bigger_no_symmetry(nn.Module):
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

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = nn.functional.relu(x[:, 0]).reshape(-1, 1, s, s)
        x1 = self.hardtanh(x[:, 1]).reshape(-1, 1, s, s)

        return torch.cat([x0, x1], dim=1)


class SRResNet_bigger_16(nn.Module):
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
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
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


class SRResNet_bigger_16_no_symmetry(nn.Module):
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
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),)

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s, s)
        x1 = self.hardtanh(x[:, 1]).reshape(-1, 1, s, s)

        return torch.cat([x0, x1], dim=1)


class SRResNet_amp(nn.Module):
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
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4, groups=1),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.symmetry_amp(x).reshape(-1, 1, s, s)

        return x


class SRResNet_phase(nn.Module):
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
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
            # SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4, groups=1),
        )

        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))
        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, x):
        x = x[:, 1].unsqueeze(1)
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.hardtanh(x).reshape(-1, 1, s, s)
        x = self.symmetry_imag(x).reshape(-1, 1, s, s)

        return x


class SRResNet_unc(nn.Module):
    def __init__(self):
        super().__init__()

        n_channel = 64

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, n_channel, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 8
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
            nn.Conv2d(n_channel, n_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_channel),
        )

        self.final = nn.Sequential(
            nn.Conv2d(n_channel, 4, 9, stride=1, padding=4, groups=2),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

        self.elu = GeneralELU(add=+(1 + 1e-10))

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

        # ResBlock 8
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
            nn.Conv2d(n_channel, n_channel, 3, stride=1, padding=1, bias=False),
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

        # ResBlock 8
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
            nn.Conv2d(n_channel, n_channel, 3, stride=1, padding=1, bias=False),
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
