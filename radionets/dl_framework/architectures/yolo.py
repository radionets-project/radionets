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
        self.anchors = 1
        # self.num_repeats = [1, 6, 12, 18, 6, 12, 12, 12, 12]
        self.num_repeats = [1, 2, 4, 6, 2, 4, 4, 4, 4]
        # self.num_repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.strides_head = torch.tensor([8, 16, 32])

        """ backbone """
        self.bb_1 = RepVGGBlock(1, self.channels, ks=3, stride=2)
        self.bb_2 = nn.Sequential(
            RepVGGBlock(self.channels, self.channels * 2, ks=3, stride=2),
            RepBlock(self.channels * 2, self.channels * 2, n=self.num_repeats[1]),
        )
        self.bb_3 = nn.Sequential(
            RepVGGBlock(self.channels * 2, self.channels * 4, ks=3, stride=2),
            RepBlock(self.channels * 4, self.channels * 4, n=self.num_repeats[2]),
        )
        self.bb_4 = nn.Sequential(
            RepVGGBlock(self.channels * 4, self.channels * 8, ks=3, stride=2),
            RepBlock(self.channels * 8, self.channels * 8, n=self.num_repeats[3]),
        )
        self.bb_5 = nn.Sequential(
            RepVGGBlock(self.channels * 8, self.channels * 16, ks=3, stride=2),
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
            *conv(self.channels * 16, self.channels * 4, 1, activation=nn.ReLU())
        )
        self.neck_upsample0 = nn.Sequential(
            *deconv(self.channels * 4, self.channels * 4, 2, 2, 0, 0)
        )
        self.neck_reduce_layer1 = nn.Sequential(
            *conv(self.channels * 4, self.channels * 2, 1, activation=nn.ReLU())
        )
        self.neck_upsample1 = nn.Sequential(
            *deconv(self.channels * 2, self.channels * 2, 2, 2, 0, 0)
        )
        self.neck_downsample2 = nn.Sequential(
            *conv(self.channels * 2, self.channels * 2, 3, 2, 1, activation=nn.ReLU())
        )
        self.neck_downsample1 = nn.Sequential(
            *conv(self.channels * 4, self.channels * 4, 3, 2, 1, activation=nn.ReLU())
        )

        """ head """
        self.head_stems = nn.Sequential(
            # stem0
            nn.Sequential(
                *conv(self.channels * 2, self.channels * 2, 1, 1, activation=nn.SiLU())
            ),
            # stem1
            nn.Sequential(
                *conv(self.channels * 4, self.channels * 4, 1, 1, activation=nn.SiLU())
            ),
            # stem2
            nn.Sequential(
                *conv(self.channels * 8, self.channels * 8, 1, 1, activation=nn.SiLU())
            ),
        )
        self.head_reg_convs = nn.Sequential(
            # reg_conv0
            nn.Sequential(
                *conv(
                    self.channels * 2, self.channels * 2, 3, 1, 1, activation=nn.SiLU()
                )
            ),
            # reg_conv1
            nn.Sequential(
                *conv(
                    self.channels * 4, self.channels * 4, 3, 1, 1, activation=nn.SiLU()
                )
            ),
            # reg_conv2
            nn.Sequential(
                *conv(
                    self.channels * 8, self.channels * 8, 3, 1, 1, activation=nn.SiLU()
                )
            ),
        )
        self.head_reg_preds = nn.Sequential(
            # reg_pred0
            nn.Conv2d(self.channels * 2, 4 * self.anchors, 1),
            # reg_pred1
            nn.Conv2d(self.channels * 4, 4 * self.anchors, 1),
            # reg_pred2
            nn.Conv2d(self.channels * 8, 4 * self.anchors, 1),
        )
        self.head_obj_preds = nn.Sequential(
            # obj_pred0
            nn.Conv2d(self.channels * 2, 1 * self.anchors, 1),
            # obj_pred1
            nn.Conv2d(self.channels * 4, 1 * self.anchors, 1),
            # obj_pred2
            nn.Conv2d(self.channels * 8, 1 * self.anchors, 1),
        )
        self.head_rot_preds = nn.Sequential(
            # rot_pred0
            nn.Conv2d(self.channels * 2, 1 * self.anchors, 1),
            # rot_pred1
            nn.Conv2d(self.channels * 4, 1 * self.anchors, 1),
            # rot_pred2
            nn.Conv2d(self.channels * 8, 1 * self.anchors, 1),
        )

    def forward(self, x):
        """backbone"""
        x = self.bb_1(x)
        x = self.bb_2(x)
        x2 = self.bb_3(x)
        x1 = self.bb_4(x2)
        x0 = self.bb_5(x1)

        """ neck """
        fpn_out0 = self.neck_reduce_layer0(x0)
        upsample_feat0 = self.neck_upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.neck_Rep0(f_concat_layer0)  # 64

        fpn_out1 = self.neck_reduce_layer1(f_out0)
        upsample_feat1 = self.neck_upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.neck_Rep1(f_concat_layer1)  # 32

        down_feat1 = self.neck_downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.neck_Rep2(p_concat_layer1)  # 64

        down_feat0 = self.neck_downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.neck_Rep3(p_concat_layer2)  # 128

        x = [pan_out2, pan_out1, pan_out0]

        """ head """
        for i in range(3):  # 3 output layers
            x[i] = self.head_stems[i](x[i])
            reg_feat = self.head_reg_convs[i](x[i])
            reg_output = self.head_reg_preds[i](reg_feat)
            obj_output = self.head_obj_preds[i](reg_feat)
            rot_output = self.head_rot_preds[i](reg_feat)

            x[i] = torch.cat([reg_output, obj_output, rot_output], 1)
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.anchors, 6, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        return x


class YOLOv6flex(nn.Module):
    """ "YOLOv6 with flexible output shape.

    Parameters to be set:
        self.channels: int
            number of default channels for convolutions, 2*n -> 2, 4, 6, ...
        self.anchors: int
            number of boxes predicted per pixel
            !!! Not implemented yet !!!
        self.strides_head: tensor
            reduction of input to output, 2^n -> 2, 4, 8, ... (1 does not work)
        self.bb_repeats: list
            number of repetitions of the RepBlock in the bacbbone. Each list
            entry leads to a new RepBlock which halves the output size.
        self.neck_repeats: int
            number of repetitions of the RepBlock in the neck.
    """

    def __init__(self):
        super().__init__()

        self.channels = 8
        self.anchors = 3
        self.strides_head = torch.tensor([8, 16, 32])
        self.bb_repeats = [1, 2, 4, 6, 4]
        # self.bb_repeats = [1, 6, 12, 18, 6]
        self.neck_repeats = 4
        # self.neck_repeats = 12

        if torch.log2(torch.max(self.strides_head)) > len(self.bb_repeats):
            print("Warning. Backbone size is larger than output size")

        self.strides_head_log = torch.log2(self.strides_head)
        self.channels_list = [
            self.channels * 2**i for i in range(len(self.bb_repeats))
        ]

        """ backbone """
        self.bb = []
        self.bb_out_idx = []
        for i, rep in enumerate(self.bb_repeats):
            channels_in = self.channels_list[i - 1]
            channels_out = self.channels_list[i]

            if i == 0:  # first layer block
                self.bb.extend(
                    (
                        RepVGGBlock(1, channels_out, ks=3, stride=2),
                        RepBlock(
                            channels_out, channels_out, n=rep
                        ),  # not in original YOLOv6, but maybe useful for large feature maps
                    )
                )
            elif i == len(self.bb_repeats) - 1:  # last layer block
                self.bb.extend(
                    (
                        RepVGGBlock(channels_in, channels_out, ks=3, stride=2),
                        RepBlock(channels_out, channels_out, n=rep),
                        SimSPPF(channels_out, channels_out),
                    )
                )
            else:
                self.bb.extend(
                    (
                        RepVGGBlock(channels_in, channels_out, ks=3, stride=2),
                        RepBlock(channels_out, channels_out, n=rep),
                    )
                )

            self.bb_out_idx.append(len(self.bb) - 1)

        self.bb = nn.Sequential(*self.bb)

        """ neck """
        # Upsample to increase image size as often as needed
        self.n_upsampling = int(len(self.bb_repeats) - torch.min(self.strides_head_log))
        self.n_downsampling = int(
            torch.max(self.strides_head_log) - torch.min(self.strides_head_log)
        )

        self.neck_Rep = []  # RepBlocks of the neck. Used after concatenations
        self.neck_reduce_layer = (
            []
        )  # Reduce number of layers to create a bottle neck effect
        self.neck_upsample = []
        self.neck_downsample = []
        self.neck_out_idx = []

        for i in range(self.n_upsampling):
            channels_reduce = self.channels_list[-i - 1]
            channels_upsampling = self.channels_list[-i - 2]
            # print('channels reduce:', channels_reduce)
            # print('channels upsampling:', channels_upsampling)

            self.neck_reduce_layer.append(
                nn.Sequential(
                    *conv(channels_reduce, channels_upsampling, 1, activation=nn.ReLU())
                )
            )
            self.neck_upsample.append(
                nn.Sequential(
                    *deconv(channels_upsampling, channels_upsampling, 2, 2, 0, 0)
                )
            )
            self.neck_Rep.append(
                RepBlock(channels_reduce, channels_upsampling, n=self.neck_repeats)
            )

        for i in range(self.n_downsampling):
            channels_downsampling = self.channels_list[
                len(self.bb_repeats) - self.n_upsampling + i - 1
            ]
            # print('channels downsampling:', channels_downsampling)

            self.neck_downsample.append(
                nn.Sequential(
                    *conv(
                        channels_downsampling,
                        channels_downsampling,
                        3,
                        2,
                        1,
                        activation=nn.ReLU(),
                    )
                )
            )
            self.neck_Rep.append(
                RepBlock(
                    channels_downsampling * 2,
                    channels_downsampling * 2,
                    n=self.neck_repeats,
                )
            )
            if torch.min(self.strides_head_log) + i + 1 in self.strides_head_log:
                self.neck_out_idx.append(i)

        # print('neck out idx:', self.neck_out_idx)
        # print('strides head log:', self.strides_head_log)

        self.neck_reduce_layer = nn.Sequential(*self.neck_reduce_layer)
        self.neck_upsample = nn.Sequential(*self.neck_upsample)
        self.neck_downsample = nn.Sequential(*self.neck_downsample)
        self.neck_Rep = nn.Sequential(*self.neck_Rep)

        """ head """
        self.n_head = len(self.strides_head)
        self.head_stems = []
        self.head_reg_convs = []
        self.head_reg_preds = []
        self.head_obj_preds = []
        self.head_rot_preds = []

        for i in range(self.n_head):
            channels_head = self.channels_list[int(self.strides_head_log[i]) - 1]

            self.head_stems.append(
                nn.Sequential(
                    *conv(channels_head, channels_head, 1, 1, activation=nn.SiLU())
                )
            )
            self.head_reg_convs.append(
                nn.Sequential(
                    *conv(channels_head, channels_head, 3, 1, 1, activation=nn.SiLU())
                )
            )
            self.head_reg_preds.append(
                nn.Conv2d(channels_head, 4 * self.anchors, 1),
            )
            self.head_obj_preds.append(
                nn.Conv2d(channels_head, 1 * self.anchors, 1),
            )
            self.head_rot_preds.append(
                nn.Conv2d(channels_head, 1 * self.anchors, 1),
            )

        self.head_stems = nn.Sequential(*self.head_stems)
        self.head_reg_convs = nn.Sequential(*self.head_reg_convs)
        self.head_reg_preds = nn.Sequential(*self.head_reg_preds)
        self.head_obj_preds = nn.Sequential(*self.head_obj_preds)
        self.head_rot_preds = nn.Sequential(*self.head_rot_preds)

    def forward(self, x):
        """backbone"""
        x_bb = []
        for i in range(len(self.bb)):
            if i == 0:
                x_bb_calc = self.bb[i](x)
            else:
                x_bb_calc = self.bb[i](x_bb_calc)

            if i in self.bb_out_idx:
                x_bb.append(x_bb_calc)
            # print(i, x_bb[-1].shape)

        """ neck """
        reduce_out = []
        for i in range(self.n_upsampling):
            if i == 0:
                reduce_out.append(self.neck_reduce_layer[i](x_bb[-1]))
                upsample_feat = self.neck_upsample[i](reduce_out[-1])
                concat_layer = torch.cat([upsample_feat, x_bb[-i - 2]], 1)
                u_out = self.neck_Rep[i](concat_layer)
            else:
                reduce_out.append(self.neck_reduce_layer[i](u_out))
                upsample_feat = self.neck_upsample[i](reduce_out[-1])
                concat_layer = torch.cat([upsample_feat, x_bb[-i - 2]], 1)
                u_out = self.neck_Rep[i](concat_layer)

        d_out = []
        for i in range(self.n_downsampling):
            if i == 0:
                down_feat = self.neck_downsample[i](u_out)
                concat_layer = torch.cat([down_feat, reduce_out[-i - 1]], 1)
                d_out_calc = self.neck_Rep[i + self.n_upsampling](concat_layer)
            else:
                down_feat = self.neck_downsample[i](d_out_calc)
                concat_layer = torch.cat([down_feat, reduce_out[-i - 1]], 1)
                d_out_calc = self.neck_Rep[i + self.n_upsampling](concat_layer)

            if i in self.neck_out_idx:
                d_out.append(d_out_calc)
            # print(i, d_out[-1].shape)

        d_out.insert(0, u_out)

        # for out in d_out:
        #     print('output shape:', out.shape)
        # quit()

        """ head """
        x = d_out
        for i in range(len(self.strides_head)):
            x[i] = self.head_stems[i](x[i])
            reg_feat = self.head_reg_convs[i](x[i])
            reg_output = self.head_reg_preds[i](reg_feat)
            obj_output = self.head_obj_preds[i](reg_feat)
            rot_output = self.head_rot_preds[i](reg_feat)

            x[i] = torch.cat([reg_output, obj_output, rot_output], 1)
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.anchors, 6, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        return x
