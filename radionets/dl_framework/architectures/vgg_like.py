from radionets.dl_framework.model import SRBlock
from torch import nn
import numpy as np


class One_or_Two_sided(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.channels = 8
        self.repeats = np.log2(img_size)
        if self.repeats % 1 == 0:
            self.repeats = int(self.repeats)
        else:
            assert True, "Image size must be 2^n (32, 64, 128, ...)"

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, self.channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.SiLU(),
        )

        self.blocks = []
        for i in range(self.repeats):
            self.blocks.extend(
                SRBlock(self.channels * (i + 1), self.channels * (i + 2), stride=2)
            )
        self.blocks = nn.Sequential(*self.blocks)

        self.postBlock = nn.Sequential(
            nn.Conv2d(self.channels * (i + 2), self.channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.SiLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(self.channels, 1, 1, stride=1),
        )

    def forward(self, x):
        x = self.preBlock(x)
        x = self.blocks(x)
        x = self.postBlock(x)

        x = self.final(x)

        return x
