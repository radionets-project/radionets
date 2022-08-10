from re import X
from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    deconv,
    double_conv,
    shape,
    linear,
    flatten,
    RepBlock,
    RepVGGBlock,
    SimSPPF,
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


class YOLOv6(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 16
        # self.input_size = 256
        self.num_repeats = [1, 6, 12, 18, 6, 12, 12, 12, 12]
        self.strides_head = torch.tensor([8, 16, 32])
        
        ''' backbone '''
        self.bb_1 = RepVGGBlock(1, self.channels, kernel_size=3, stride=2)
        self.bb_2 = nn.Sequential(
            RepVGGBlock(self.channels, self.channels * 2, kernel_size=3, stride=2),
            RepBlock(self.channels * 2, self.channels * 2, n=self.num_repeats[1])
        )
        self.bb_3 = nn.Sequential(
            RepVGGBlock(self.channels * 2, self.channels * 4, kernel_size=3, stride=2),
            RepBlock(self.channels * 4, self.channels * 4, n=self.num_repeats[2])
        )
        self.bb_4 = nn.Sequential(
            RepVGGBlock(self.channels * 4, self.channels * 8, kernel_size=3, stride=2),
            RepBlock(self.channels * 8, self.channels * 8, n=self.num_repeats[3])
        )
        self.bb_5 = nn.Sequential(
            RepVGGBlock(self.channels * 8, self.channels * 16, kernel_size=3, stride=2),
            RepBlock(self.channels * 16, self.channels * 16, n=self.num_repeats[4]),
            SimSPPF(self.channels * 16, self.channels * 16)
        )

        ''' neck '''
        self.neck_Rep0 = RepBlock(self.channels * 12, self.channels * 4, n=self.num_repeats[5])
        self.neck_Rep1 = RepBlock(self.channels * 6, self.channels * 2, n=self.num_repeats[6])
        self.neck_Rep2 = RepBlock(self.channels * 4, self.channels * 4, n=self.num_repeats[7])
        self.neck_Rep3 = RepBlock(self.channels * 8, self.channels * 8, n=self.num_repeats[8])

        self.neck_reduce_layer0 = conv(self.channels * 16, self.channels * 4, 1)
        self.neck_upsample0 = nn.Sequential(
            *deconv(self.channels * 4, self.channels * 4, 2, 2, 0, 0)
        )
        self.neck_reduce_layer1 = conv(self.channels * 4, self.channels * 2, 1)
        self.neck_upsample1 = nn.Sequential(
            *deconv(self.channels * 2, self.channels * 2, 2, 2, 0, 0)
        )
        self.neck_downsample2 = conv(self.channels * 2, self.channels * 2, 3, 2)
        self.neck_downsample1 = conv(self.channels * 4, self.channels * 4, 3, 2)
        
        ''' head '''
        self.head_stems = nn.Sequential(
            # stem0
            conv(self.channels * 2, self.channels * 2, 1, 1),
            # stem1
            conv(self.channels * 4, self.channels * 4, 1, 1),
            # stem2
            conv(self.channels * 8, self.channels * 8, 1, 1),
        )
        self.head_reg_convs = nn.Sequential(
            # reg_conv0
            conv(self.channels * 2, self.channels * 2, 3, 1, 1),
            # reg_conv1
            conv(self.channels * 4, self.channels * 4, 3, 1, 1),
            # reg_conv2
            conv(self.channels * 8, self.channels * 8, 3, 1, 1),
        )
        self.head_reg_preds = nn.Sequential(
            # reg_pred0
            nn.Conv2d(self.channels * 2, 4, 1),
            # reg_pred1
            nn.Conv2d(self.channels * 4, 4, 1),
            # reg_pred2
            nn.Conv2d(self.channels * 8, 4, 1),
        )
        self.head_obj_preds = nn.Sequential(
            # obj_pred0
            nn.Conv2d(self.channels * 2, 1, 1),
            # obj_pred1
            nn.Conv2d(self.channels * 4, 1, 1),
            # obj_pred2
            nn.Conv2d(self.channels * 8, 1, 1),
        )



    def forward(self, x):
        ''' backbone '''
        x = self.self.bb_1(x)
        x = self.self.bb_2(x)
        x2 = self.self.bb_3(x)
        x1 = self.self.bb_4(x2)
        x0 = self.self.bb_5(x1)

        ''' neck '''
        # original names
        fpn_out0 = self.neck_reduce_layer0(x0)
        upsample_feat0 = self.neck_upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.neck_Rep0(f_concat_layer0)

        fpn_out1 = self.neck_reduce_layer1(f_out0)
        upsample_feat1 = self.neck_upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.neck_Rep1(f_concat_layer1)

        down_feat1 = self.neck_downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.neck_Rep2(p_concat_layer1)

        down_feat0 = self.neck_downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.neck_Rep3(p_concat_layer2)
        
        x = [pan_out2, pan_out1, pan_out0]

        # easier names
        # x_red0 = self.neck_reduce_layer0(x0)
        # x = self.neck_upsample0(x_red0)
        # x = torch.cat([x, x1], 1)
        # x = self.neck_Rep0(x)

        # x_red1 = self.neck_reduce_layer1(x)
        # x = self.neck_upsample1(x_red1)
        # x = torch.cat([x, x2], 1)
        # out2 = self.neck_Rep1(x)

        # x = self.neck_downsample2(out2)
        # x = torch.cat([x, x_red1], 1)
        # out1 = self.neck_Rep2(x)

        # x = self.neck_downsample1(out1)
        # x = torch.cat([x, x_red0], 1)
        # out0 = self.neck_Rep3(x)

        # x = [out2, out1, out0]

        ''' head '''
        z = []
        for i in range(3):  # 3 output layers
            x[i] = self.head_stems[i](x[i])
            reg_feat = self.head_reg_convs[i](x[i])
            reg_output = self.head_reg_preds[i](reg_feat)
            obj_output = self.head_obj_preds[i](reg_feat)

            y = torch.cat([reg_output, obj_output.sigmoid()], 1)
            bs, _, ny, nx = x[i].shape
            y = y.view(bs, 1, 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            d = self.strides_head.device
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
            grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
            y[..., 0:2] = (y[..., 0:2] + grid) * self.stride[i] # xy
            y[..., 0:22:4] = torch.exp(y[..., 2:4]) * self.stride[i] # xy

            z.append(y.view(bs, -1, 5))
        return torch.cat(z, 1)
