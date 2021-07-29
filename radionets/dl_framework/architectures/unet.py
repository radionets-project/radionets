from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    Lambda,
    flatten,
    fft,
    double_conv,
    cut_off,
    flatten_with_channel,
    shape,
)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(1, 4))
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8))
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16))
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32))
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64))

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32))
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16))
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8))
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4))

        self.conv_last = nn.Conv2d(4, 1, 1)
        self.flatten = Lambda(flatten)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

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

        out = self.flatten(x)

        return out


class UNet_fft(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.linear1 = nn.Linear(8192, 4096)
        self.fft = Lambda(fft)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)

        x = self.flatten(x)
        x = self.fft(x)
        x = self.flatten(x)
        out = self.linear1(x)

        return out


class UNet_denoise(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.linear = nn.Linear(7938, 3969)
        self.fft = Lambda(fft)
        self.cut = Lambda(cut_off)
        self.shape = Lambda(shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fft(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        x = self.flatten(x)
        out = self.linear(x)

        return out


class UNet_fourier(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.linear = nn.Linear(8192, 4096)
        self.flatten = Lambda(flatten)
        self.flatten_with_channel = Lambda(flatten_with_channel)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        # out = self.flatten_with_channel(x)

        return x


class UNet_denoise_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )
        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.linear = nn.Linear(8192, 4096)
        self.fft = Lambda(fft)
        self.cut = Lambda(cut_off)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fft(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)
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
        x = self.flatten(x)
        out = self.linear(x)
        return out


class UNet_segmentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 8

        self.dconv_down1 = nn.Sequential(*double_conv(1, self.channels))
        self.dconv_down2 = nn.Sequential(*double_conv(self.channels, 2*self.channels))
        self.dconv_down3 = nn.Sequential(*double_conv(2*self.channels, 4*self.channels))
        self.dconv_down4 = nn.Sequential(*double_conv(4*self.channels, 8*self.channels))
        self.dconv_down5 = nn.Sequential(*double_conv(8*self.channels, 16*self.channels))

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(24*self.channels, 8*self.channels))
        self.dconv_up3 = nn.Sequential(*double_conv(12*self.channels, 4*self.channels))
        self.dconv_up2 = nn.Sequential(*double_conv(6*self.channels, 2*self.channels))
        self.dconv_up1 = nn.Sequential(*double_conv(3*self.channels, self.channels))

        self.conv_last = nn.Conv2d(8, 4, 1)
        self.output_activation = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

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

        out = self.conv_last(x)
        out = self.output_activation(out)

        return out
