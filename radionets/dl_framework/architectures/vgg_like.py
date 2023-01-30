from torch import nn
import numpy as np


class One_or_Two_sided(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()

        channels = 8
        repeats = np.log2(img_size)
        if repeats % 1 == 0:
            repeats = int(repeats)
        else:
            assert True, "Image size must be 2^n (32, 64, 128, ...)"

        self.preBlock = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        self.blocks = []
        for _ in range(repeats):
            self.blocks.extend(
                (
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                )
            )
        self.blocks = nn.Sequential(*self.blocks)

        self.final = nn.Sequential(
            nn.Conv2d(channels, 1, 1, stride=1),
        )

    def forward(self, x):
        x = self.preBlock(x)
        x = self.blocks(x)
        x = self.final(x)

        return x.squeeze()
