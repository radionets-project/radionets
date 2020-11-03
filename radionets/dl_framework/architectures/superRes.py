import torch
from torch import nn
from math import pi
import numpy as np
from radionets.dl_framework.model import (
    GeneralELU,
)




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

