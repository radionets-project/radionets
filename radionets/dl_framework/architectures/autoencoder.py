import torch
from torch import nn
from radionets.dl_framework.model import SRBlock

from math import pi


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.PReLU(),
            SRBlock(128, 1),
        )

        self.decoder = nn.Sequential(
            SRBlock(1, 128),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1, stride=2, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder_amp(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            #nn.InstanceNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            #nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            #nn.InstanceNorm2d(32),
            nn.PReLU(),
            SRBlock(128, 64),
        )

        self.decoder = nn.Sequential(
            SRBlock(64, 128),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1),
            #nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),
            #nn.InstanceNorm2d(8),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1, stride=2, output_padding=1),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder_phase(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2),
            #nn.InstanceNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            #nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            #nn.InstanceNorm2d(128),
            nn.PReLU(),
            SRBlock(128, 64),
        )

        self.decoder = nn.Sequential(
            SRBlock(64, 128),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1),
            #nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1),
            #nn.InstanceNorm2d(8),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1, stride=2, output_padding=1),
            #nn.Hardtanh(-pi, pi),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

