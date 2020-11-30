import torch
from torch import nn
from math import pi
from radionets.dl_framework.model import (
    GeneralELU,
    ResBlock_amp,
    ResBlock_phase,
    SRBlock,
    Lambda,
    symmetry,
)
from functools import partial


class superRes_simple(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.conv1_amp = nn.Sequential(
            nn.Conv2d(1, 4, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv1_phase = nn.Sequential(
            nn.Conv2d(1, 4, stride=2, kernel_size=3, padding=3 // 2), GeneralELU(1 - pi)
        )
        self.conv2_amp = nn.Sequential(
            nn.Conv2d(4, 8, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv2_phase = nn.Sequential(
            nn.Conv2d(4, 8, stride=2, kernel_size=3, padding=3 // 2), GeneralELU(1 - pi)
        )
        self.conv3_amp = nn.Sequential(
            nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv3_phase = nn.Sequential(
            nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=3 // 2),
            GeneralELU(1 - pi),
        )
        self.conv4_amp = nn.Sequential(
            nn.Conv2d(16, 32, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv4_phase = nn.Sequential(
            nn.Conv2d(16, 32, stride=2, kernel_size=3, padding=3 // 2),
            GeneralELU(1 - pi),
        )
        self.conv5_amp = nn.Sequential(
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv5_phase = nn.Sequential(
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=3 // 2),
            GeneralELU(1 - pi),
        )
        self.final_amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, img_size ** 2)
        )
        self.final_phase = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, img_size ** 2)
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.conv1_amp(amp)
        phase = self.conv1_phase(phase)

        amp = self.conv2_amp(amp)
        phase = self.conv2_phase(phase)

        amp = self.conv3_amp(amp)
        phase = self.conv3_phase(phase)

        amp = self.conv4_amp(amp)
        phase = self.conv4_phase(phase)

        amp = self.conv5_amp(amp)
        phase = self.conv5_phase(phase)

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class superRes_res18(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock_amp = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.preBlock_phase = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            GeneralELU(1 - pi),
        )

        self.maxpool_amp = nn.MaxPool2d(3, 2, 1)
        self.maxpool_phase = nn.MaxPool2d(3, 2, 1)

        # first block
        self.layer1_amp = nn.Sequential(ResBlock_amp(64, 64), ResBlock_amp(64, 64))
        self.layer1_phase = nn.Sequential(
            ResBlock_phase(64, 64), ResBlock_phase(64, 64)
        )

        self.layer2_amp = nn.Sequential(
            ResBlock_amp(64, 128, stride=2), ResBlock_amp(128, 128)
        )
        self.layer2_phase = nn.Sequential(
            ResBlock_phase(64, 128, stride=2), ResBlock_phase(128, 128)
        )

        self.layer3_amp = nn.Sequential(
            ResBlock_amp(128, 256, stride=2), ResBlock_amp(256, 256)
        )
        self.layer3_phase = nn.Sequential(
            ResBlock_phase(128, 256, stride=2), ResBlock_phase(256, 256)
        )

        self.layer4_amp = nn.Sequential(
            ResBlock_amp(256, 512, stride=2), ResBlock_amp(512, 512)
        )
        self.layer4_phase = nn.Sequential(
            ResBlock_phase(256, 512, stride=2), ResBlock_phase(512, 512)
        )

        self.final_amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )
        self.final_phase = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.preBlock_amp(amp)
        phase = self.preBlock_phase(phase)

        amp = self.maxpool_amp(amp)
        phase = self.maxpool_phase(phase)

        amp = self.layer1_amp(amp)
        phase = self.layer1_phase(phase)

        amp = self.layer2_amp(amp)
        phase = self.layer2_phase(phase)

        amp = self.layer3_amp(amp)
        phase = self.layer3_phase(phase)

        amp = self.layer4_amp(amp)
        phase = self.layer4_phase(phase)

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class superRes_res34(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

        self.preBlock_amp = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.preBlock_phase = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            GeneralELU(1 - pi),
        )

        self.maxpool_amp = nn.MaxPool2d(3, 2, 1)
        self.maxpool_phase = nn.MaxPool2d(3, 2, 1)

        # first block
        self.layer1_amp = nn.Sequential(
            ResBlock_amp(64, 64), ResBlock_amp(64, 64), ResBlock_amp(64, 64)
        )
        self.layer1_phase = nn.Sequential(
            ResBlock_phase(64, 64), ResBlock_phase(64, 64), ResBlock_phase(64, 64)
        )

        self.layer2_amp = nn.Sequential(
            ResBlock_amp(64, 128, stride=2),
            ResBlock_amp(128, 128),
            ResBlock_amp(128, 128),
            ResBlock_amp(128, 128),
        )
        self.layer2_phase = nn.Sequential(
            ResBlock_phase(64, 128, stride=2),
            ResBlock_phase(128, 128),
            ResBlock_phase(128, 128),
            ResBlock_phase(128, 128),
        )

        self.layer3_amp = nn.Sequential(
            ResBlock_amp(128, 256, stride=2),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
        )
        self.layer3_phase = nn.Sequential(
            ResBlock_phase(128, 256, stride=2),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
        )

        self.layer4_amp = nn.Sequential(
            ResBlock_amp(256, 512, stride=2),
            ResBlock_amp(512, 512),
            ResBlock_amp(512, 512),
        )
        self.layer4_phase = nn.Sequential(
            ResBlock_phase(256, 512, stride=2),
            ResBlock_phase(512, 512),
            ResBlock_phase(512, 512),
        )

        self.final_amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )
        self.final_phase = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.preBlock_amp(amp)
        phase = self.preBlock_phase(phase)

        amp = self.maxpool_amp(amp)
        phase = self.maxpool_phase(phase)

        amp = self.layer1_amp(amp)
        phase = self.layer1_phase(phase)

        amp = self.layer2_amp(amp)
        phase = self.layer2_phase(phase)

        amp = self.layer3_amp(amp)
        phase = self.layer3_phase(phase)

        amp = self.layer4_amp(amp)
        phase = self.layer4_phase(phase)

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class SRResNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        # torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock_amp = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4), nn.PReLU()
        )
        self.preBlock_phase = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4), nn.PReLU()
        )

        # ResBlock 16
        self.blocks_amp = nn.Sequential(
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
        self.blocks_phase = nn.Sequential(
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

        self.postBlock_amp = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )
        self.postBlock_phase = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final_amp = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4),
        )
        self.final_phase = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4),
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.preBlock_amp(amp)
        phase = self.preBlock_phase(phase)

        amp = amp + self.postBlock_amp(self.blocks_amp(amp))
        phase = phase + self.postBlock_phase(self.blocks_phase(phase))

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class SRResNet_corr(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.cuda.set_device(1)
        # self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 12
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


class SRResNet_amp(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.cuda.set_device(1)
        # self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 32, 9, stride=1, padding=4, groups=1), nn.PReLU()
        )

        # ResBlock 12
        self.blocks = nn.Sequential(
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32)
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 9, stride=1, padding=4, groups=1),
        )

        self.symmetry_amp = Lambda(partial(symmetry, mode="real"))

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.symmetry_amp(x).reshape(-1, 1, 63, 63)

        return x


class SRResNet_phase(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.cuda.set_device(1)
        # self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, 32, 9, stride=1, padding=4, groups=1), nn.PReLU()
        )

        # ResBlock 12
        self.blocks = nn.Sequential(
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
            SRBlock(32, 32),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32)
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 9, stride=1, padding=4, groups=1),
        )

        self.symmetry_phase = Lambda(partial(symmetry, mode="imag"))

    def forward(self, x):
        x = x[:, 1].unsqueeze(1)

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x = self.symmetry_phase(x).reshape(-1, 1, 63, 63)

        return x
