from torch import nn
import torch
from dl_framework.model import conv, Lambda, flatten, shape, fft, deconv, double_conv, cut_off, flatten_with_channel


def cnn():
    """
    conv-layer: number of entry channels, number of exit channels,
                kerner size, stride, padding
    """
    arch = nn.Sequential(
        *conv(2, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        Lambda(flatten),
        # Lambda(shape),
        nn.Linear(64, 32768),
        # Lambda(shape),
        Lambda(fft),
        # Lambda(shape),
        # Lambda(flatten),
        # Lambda(shape),
        *conv(2, 1, 1, 1, 0),
        Lambda(flatten),
    )
    return arch


def autoencoder():
    arch = nn.Sequential(
        *conv(2, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),

        *deconv(64, 32, (3, 3), 2, 1, 0),
        *deconv(32, 16, (3, 3), 2, 1, 0),
        *deconv(16, 16, (3, 3), 2, 1, 0),
        *deconv(16, 8, (3, 3), 2, 1, 0),
        *deconv(8, 4, (3, 3), 2, 1, 0),

        # nn.ConvTranspose2d(4, 2, (3, 3), 2, 1, 1),
        *deconv(4, 2, (3, 3), 2, 1, 0),
        Lambda(flatten),
        # nn.Linear(8192, 4096)
        nn.Linear(2, 4096)
    )
    return arch


class UNet_fft(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.conv_shape = nn.Conv2d(2, 1, 1)
        self.flatten = Lambda(flatten)
        self.linear1 = nn.Linear(16384, 16384)
        # self.linear2 = nn.Linear(32768, 32768)
        self.fft = Lambda(fft)
        self.cut = Lambda(cut_off)
        self.dropout = nn.Dropout2d(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

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

        x = self.flatten(x)
        x = self.fft(x)
        x = self.conv_shape(x)
        out = self.flatten(x)
        # out = self.linear1(x)
        # x = self.linear2(x)
        # x = self.dropout2(x)
        # out = self.linear1(x)

        return out


class UNet_denoise(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

        self.conv_last = nn.Conv2d(4, 1, 1)
        self.flatten = Lambda(flatten)
        self.flatten_with_channel = Lambda(flatten_with_channel)
        self.linear = nn.Linear(8192, 4096)
        self.fft = Lambda(fft)
        self.cut = Lambda(cut_off)
        self.shape = Lambda(shape)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fft(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        # x = self.dropout(conv2)
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
        # out = self.linear(x)

        return out


class UNet_fourier(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.flatten_with_channel = Lambda(flatten_with_channel)
        self.cut = Lambda(cut_off)
        self.shape = Lambda(shape)
        self.dropout = nn.Dropout2d(p=0.5)

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
        out = self.flatten_with_channel(x)

        return out
