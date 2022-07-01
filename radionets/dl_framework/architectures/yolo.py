from re import X
from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    double_conv,
    shape,
    linear,
    flatten,
)


class yolo_vgg(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 16
        self.input_size = 256

        self.dconv_down1 = nn.Sequential(*double_conv(1, self.channels))
        self.dconv_down2 = nn.Sequential(
            *double_conv(self.channels, 2 * self.channels)
        )
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
            *linear(int(32 * self.channels * (self.input_size / 2 ** 5) ** 2), 2048)
        )
        self.linear2 = nn.Sequential(
            *linear(2048, 11 * 7)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.output_activation = nn.Sequential(nn.Sigmoid())

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
        x = self.maxpool(x)
        x = self.dconv_down6(x)

        x = flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        out = self.output_activation(x)

        return out



class yolo_unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 64
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

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up5 = nn.Sequential(
            *double_conv(48 * self.channels, 16 * self.channels)
        )
        self.dconv_up4 = nn.Sequential(
            *double_conv(24 * self.channels, 8 * self.channels)
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(12 * self.channels, 4 * self.channels)
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(6 * self.channels, 2 * self.channels)
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(3 * self.channels, self.channels)
        )

        self.conv_last = nn.Conv2d(self.channels, 1, 1)
        self.linear = nn.Sequential(
            *linear(self.input_size ** 2, 77)
        )
        self.output_activation = nn.Sequential(nn.Sigmoid())

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        x = self.dconv_down6(x)

        x = self.upsample(x)
        x = torch.cat([x, conv5], dim=1)
        x = self.dconv_up5(x)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = flatten(x)
        x = self.linear(x)
        out = self.output_activation(x)

        return out
