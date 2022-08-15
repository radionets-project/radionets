from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    deconv,
    RepBlock,
    RepVGGBlock,
    SimSPPF,
)


class YOLOv6(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 8
        # self.input_size = 256
        # self.num_repeats = [1, 6, 12, 18, 6, 12, 12, 12, 12]
        self.num_repeats = [1, 4, 8, 12, 4, 8, 8, 8, 8]
        self.strides_head = torch.tensor([8, 16, 32])

        """ backbone """
        self.bb_1 = RepVGGBlock(1, self.channels, kernel_size=3, stride=2)
        self.bb_2 = nn.Sequential(
            RepVGGBlock(self.channels, self.channels * 2, kernel_size=3, stride=2),
            RepBlock(self.channels * 2, self.channels * 2, n=self.num_repeats[1]),
        )
        self.bb_3 = nn.Sequential(
            RepVGGBlock(self.channels * 2, self.channels * 4, kernel_size=3, stride=2),
            RepBlock(self.channels * 4, self.channels * 4, n=self.num_repeats[2]),
        )
        self.bb_4 = nn.Sequential(
            RepVGGBlock(self.channels * 4, self.channels * 8, kernel_size=3, stride=2),
            RepBlock(self.channels * 8, self.channels * 8, n=self.num_repeats[3]),
        )
        self.bb_5 = nn.Sequential(
            RepVGGBlock(self.channels * 8, self.channels * 16, kernel_size=3, stride=2),
            RepBlock(self.channels * 16, self.channels * 16, n=self.num_repeats[4]),
            SimSPPF(self.channels * 16, self.channels * 16),
        )

        """ neck """
        self.neck_Rep0 = RepBlock(
            self.channels * 12, self.channels * 4, n=self.num_repeats[5]
        )
        self.neck_Rep1 = RepBlock(
            self.channels * 6, self.channels * 2, n=self.num_repeats[6]
        )
        self.neck_Rep2 = RepBlock(
            self.channels * 4, self.channels * 4, n=self.num_repeats[7]
        )
        self.neck_Rep3 = RepBlock(
            self.channels * 8, self.channels * 8, n=self.num_repeats[8]
        )

        self.neck_reduce_layer0 = nn.Sequential(
            *conv(self.channels * 16, self.channels * 4, 1)
        )
        self.neck_upsample0 = nn.Sequential(
            *deconv(self.channels * 4, self.channels * 4, 2, 2, 0, 0)
        )
        self.neck_reduce_layer1 = nn.Sequential(
            *conv(self.channels * 4, self.channels * 2, 1)
        )
        self.neck_upsample1 = nn.Sequential(
            *deconv(self.channels * 2, self.channels * 2, 2, 2, 0, 0)
        )
        self.neck_downsample2 = nn.Sequential(
            *conv(self.channels * 2, self.channels * 2, 3, 2, 1)
        )
        self.neck_downsample1 = nn.Sequential(
            *conv(self.channels * 4, self.channels * 4, 3, 2, 1)
        )

        """ head """
        self.head_stems = nn.Sequential(
            # stem0
            nn.Sequential(*conv(self.channels * 2, self.channels * 2, 1, 1)),
            # stem1
            nn.Sequential(*conv(self.channels * 4, self.channels * 4, 1, 1)),
            # stem2
            nn.Sequential(*conv(self.channels * 8, self.channels * 8, 1, 1)),
        )
        self.head_reg_convs = nn.Sequential(
            # reg_conv0
            nn.Sequential(*conv(self.channels * 2, self.channels * 2, 3, 1, 1)),
            # reg_conv1
            nn.Sequential(*conv(self.channels * 4, self.channels * 4, 3, 1, 1)),
            # reg_conv2
            nn.Sequential(*conv(self.channels * 8, self.channels * 8, 3, 1, 1)),
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
        """ backbone """
        x = self.bb_1(x)
        x = self.bb_2(x)
        x2 = self.bb_3(x)
        x1 = self.bb_4(x2)
        x0 = self.bb_5(x1)

        """ neck """
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

        """ head """
        z = []
        for i in range(3):  # 3 output layers
            x[i] = self.head_stems[i](x[i])
            reg_feat = self.head_reg_convs[i](x[i])
            reg_output = self.head_reg_preds[i](reg_feat)
            obj_output = self.head_obj_preds[i](reg_feat)

            y = torch.cat([reg_output.sigmoid(), obj_output.sigmoid()], 1)
            bs, _, ny, nx = x[i].shape
            y = y.view(bs, 5, ny, nx).permute(0, 2, 3, 1).contiguous()
            d = y.device
            yv, xv = torch.meshgrid(
                [torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).view(1, ny, nx, 2).float()
            #print(nx)
            # y[..., 0:2] = (y[..., 0:2] + grid) * self.strides_head[i].to(d)  # xy, org. YOLOv6
            #print(y[0, 0, 0:5, 0:2])
            y[..., 0:2] = (y[..., 0:2] + grid) * self.strides_head[i].to(d)  # xy
            # y[..., 2:4] = torch.exp(y[..., 2:4]) * self.strides_head[i].to(d)  # wh, org. YOLOv6
            y[..., 2:4] = y[..., 2:4] * max(nx, ny) * self.strides_head[i].to(d)  # wh
            #print(y[0, 0, 0:5, 0:2])
            #print(y[0, 0, 0:5, 2:4])
            # print(f'Output shape before concat: {y.view(bs, -1, 5).shape}')
            z.append(y.view(bs, -1, 5))
        return torch.cat(z, 1)
