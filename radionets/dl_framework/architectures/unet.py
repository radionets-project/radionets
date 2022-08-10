from matplotlib.pyplot import xcorr
from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    Lambda,
    conv_bn,
    flatten,
    fft,
    double_conv,
    cut_off,
    flatten_with_channel,
    shape,
    linear,
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

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

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

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

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

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

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
        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)
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


class UNet_jet(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 16

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
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(24 * self.channels, 8 * self.channels)
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(12 * self.channels, 4 * self.channels)
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(6 * self.channels, 2 * self.channels)
        )
        self.dconv_up1 = nn.Sequential(*double_conv(3 * self.channels, self.channels))

        self.conv_last = nn.Conv2d(self.channels, 12, 1)
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


class UNet_jet_advanced(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 32  # 32 is enough to predict all (12) components as an image
        self.components = 12
        self.grad_backbone = False
        self.grad_regressor = True 

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

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(24 * self.channels, 8 * self.channels)
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(12 * self.channels, 4 * self.channels)
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(6 * self.channels, 2 * self.channels)
        )
        self.dconv_up1 = nn.Sequential(*double_conv(3 * self.channels, self.channels))
        self.conv_last = conv_bn(self.channels, self.components, 1)
        self.activation_unet = nn.Softmax(dim=1)

        self.conv1 = nn.Sequential(*conv(
            ni=self.components,
            nc=self.components,
            ks=3,
            stride=2,
            padding=1,
            groups=self.components,
            ))
        self.conv2 = nn.Sequential(*conv(
            ni=self.components,
            nc=self.components,
            ks=3,
            stride=2,
            padding=1,
            groups=self.components,
            ))

        # 6 for each component: confidence, amplitude, x, y, width, height
        self.linear_comp = nn.ModuleList(
            [nn.Linear(64 ** 2, 6) for _ in range(self.components - 1)]
        )
        # self.linear_comp = nn.Linear(64 ** 2, 6)

        # av: angle & velocity, 12: 11x velocity and 1 angle
        self.linear_av1 = linear(64 ** 2 * self.components, 64)
        self.linear_av2 = nn.Linear(64, 12)
        self.sigmoid = nn.Sigmoid()
    
    def apply_grad(self, layer, x, gradient):
        if gradient:
            y = layer(x)
        else:
            with torch.no_grad():
                y = layer(x)
        return y

    def forward(self, x):
        conv1 = self.apply_grad(self.dconv_down1, x, self.grad_backbone)
        x = self.maxpool(conv1)
        conv2 = self.apply_grad(self.dconv_down2, x, self.grad_backbone)
        x = self.maxpool(conv2)
        conv3 = self.apply_grad(self.dconv_down3, x, self.grad_backbone)
        x = self.maxpool(conv3)
        conv4 = self.apply_grad(self.dconv_down4, x, self.grad_backbone)
        x = self.maxpool(conv4)
        x = self.apply_grad(self.dconv_down5, x, self.grad_backbone)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.apply_grad(self.dconv_up4, x, self.grad_backbone)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.apply_grad(self.dconv_up3, x, self.grad_backbone)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.apply_grad(self.dconv_up2, x, self.grad_backbone)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.apply_grad(self.dconv_up1, x, self.grad_backbone)
        x = self.apply_grad(self.conv_last, x, self.grad_backbone)
        out_unet = self.activation_unet(x)

        # x = self.maxpool(out_unet)
        x = self.apply_grad(self.conv1, out_unet, self.grad_regressor)
        x = self.apply_grad(self.conv2, x, self.grad_regressor)

        x = torch.flatten(x, start_dim=2)

        d = str(x.get_device()) 
        if d == "-1":
            device = "cpu"
        else:
            device = "cuda:" + d

        out_comp = torch.zeros(
            (x.shape[0], x.shape[1] - 1, 6), device=torch.device(device)
        )
        for i, linear in enumerate(self.linear_comp):
            out_comp[:, i] = self.sigmoid(self.apply_grad(linear, x[:, i], self.grad_regressor))
        # for i in range(self.components - 1):
        #     out_comp[:, i] = self.sigmoid(self.apply_grad(self.linear_comp, x[:, i], self.grad_regressor))
        # print(out_comp.shape)
        # print(x.shape)

        # last component is background -> same information as clean image
        x = torch.flatten(x, start_dim=1)
        out_av = self.apply_grad(self.linear_av1, x, self.grad_regressor)
        out_av = self.sigmoid(self.apply_grad(self.linear_av2, out_av, self.grad_regressor))
        # print(out_av.shape)

        return out_unet, out_comp, out_av


class UNet_clean(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 16

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
        self.dconv_up1 = nn.Sequential(*double_conv(3 * self.channels, self.channels))

        self.conv_last = nn.Conv2d(self.channels, 2, 1)
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

        out = self.conv_last(x)
        out = self.output_activation(out)

        return out


class UNet_survey(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 8

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
        # self.dconv_down6 = nn.Sequential(*double_conv(16*self.channels, 32*self.channels))

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # self.dconv_up5 = nn.Sequential(*double_conv(48*self.channels, 16*self.channels))
        self.dconv_up4 = nn.Sequential(
            *double_conv(24 * self.channels, 8 * self.channels)
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(12 * self.channels, 4 * self.channels)
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(6 * self.channels, 2 * self.channels)
        )
        self.dconv_up1 = nn.Sequential(*double_conv(3 * self.channels, self.channels))

        self.conv_last = nn.Conv2d(self.channels, 4, 1)
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
        #        x = self.maxpool(conv5)
        #        x = self.dconv_down6(x)

        #        x = self.upsample(x)
        #        x = torch.cat([x, conv5], dim=1)
        #        x = self.dconv_up5(x)
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
