from re import X
from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    double_conv,
    shape,
    linear,
)


class yolo_vgg(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 16
        self.input_size = 256

        self.dconv_down1 = nn.Sequential(*double_conv(1, self.channels))
        self.dconv_down2 = nn.Sequential(*double_conv(self.channels, 2 * self.channels))
        self.dconv_down3 = nn.Sequential(
            *double_conv(2 * self.channels, 4 * self.channels)
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(4 * self.channels, 8 * self.channels)
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(8 * self.channels, 16 * self.channels)
        )
        self.dconv_down6 = nn.Sequential(
            *double_conv(16 * self.channels, 32 * self.channels)
        )
        self.linear1 = nn.Sequential(
            *linear(32 * self.channels * (self.input_size / 2 ** 5) ** 2, 2048)
        )
        self.linear2 = nn.Sequential(
            *linear(2048, 11 * 7)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.output_activation = nn.Sequential(nn.Sigmoid(dim=1))

    def forward(self, x):
        x = self.dconv_down1(x)
        x = self.maxpool(x)
        x = self.dconv_down2(x)
        x = self.maxpool(x)
        x = self.dconv_down3(x)
        x = self.maxpool(x)
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        x = self.dconv_down5(x)
        x = self.maxpool(X)
        x = self.dconv_down6(x)

        x = self.linear1(x)
        x = self.linear2(x)
        out = self.output_activation(x)

        return out