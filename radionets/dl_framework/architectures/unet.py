from functools import partial

import fastcore.all as fc

# import torch
import torch.nn.functional as F
from torch import nn


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x -= self.sub
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def conv(ni, nf, ks=3, stride=1, act=nn.ReLU, norm=None, bias=True, groups=1):
    layers = []
    if norm:
        layers.append(norm(ni))
    if act:
        layers.append(act())
    layers.append(
        nn.Conv2d(
            ni,
            nf,
            stride=stride,
            kernel_size=ks,
            padding=ks // 2,
            bias=bias,
            groups=groups,
        )
    )
    return nn.Sequential(*layers)


act_gr = partial(GeneralRelu, leak=0.1, sub=0.4)


def _conv_block(ni, nf, stride, act=act_gr, norm=None, ks=3, groups=1):
    return nn.Sequential(
        conv(ni, nf, stride=1, act=act, norm=norm, ks=ks, groups=groups),
        conv(nf, nf, stride=stride, act=act, norm=norm, ks=ks, groups=groups),
    )


class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=3, act=act_gr, norm=None, param_groups=1):
        super().__init__()
        self.convs = _conv_block(
            ni, nf, stride, act=act, ks=ks, norm=norm, groups=param_groups
        )
        self.idconv = (
            fc.noop if ni == nf else conv(ni, nf, ks=1, stride=1, act=None, norm=norm)
        )
        self.pool = fc.noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))


def up_block(ni, nf, ks=3, act=act_gr, norm=None):
    return nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2),
        ResBlock(ni, nf, ks=ks, act=act, norm=norm),
    )


def up_block2(ni, nf, ks=3, act=act_gr, norm=None):
    return nn.Sequential(
        # ResBlock(ni, int(2*ni), ks=ks, act=act, norm=norm),
        # nn.PixelShuffle(2),
        nn.ConvTranspose2d(ni, nf, ks, padding=1, stride=2, output_padding=1),
        ResBlock(nf, nf, ks=ks, act=act, norm=norm),
    )


class UNet(nn.Module):
    def __init__(self, act=act_gr, nfs=(32, 64, 128, 256, 256), norm=nn.InstanceNorm2d):
        super().__init__()
        self.start = ResBlock(12, nfs[0], stride=1, act=act, norm=norm, param_groups=2)
        self.dn = nn.ModuleList(
            [
                ResBlock(nfs[i], nfs[i + 1], act=act, norm=norm, stride=2)
                for i in range(len(nfs) - 1)
            ]
        )
        self.up = nn.ModuleList(
            [
                up_block(nfs[i], nfs[i - 1], act=act, norm=norm)
                for i in range(len(nfs) - 1, 0, -1)
            ]
        )
        self.up += [ResBlock(nfs[0], 12, act=act, norm=norm)]
        self.end = ResBlock(12, 12, act=nn.Identity, norm=norm, param_groups=2)

    def forward(self, x):
        # means = (
        #    x.mean(axis=-1)
        #    .mean(axis=-1)
        #    .reshape(x.shape[0], x.shape[1], 1, 1)
        # )
        # stds = (
        #    x.std(axis=-1)
        #    .std(axis=-1)
        #    .reshape(x.shape[0], x.shape[1], 1, 1)
        # )
        layers = []
        layers.append(x)
        x = self.start(x)
        for lay in self.dn:
            layers.append(x)
            x = lay(x)
        n = len(layers)
        for i, lay in enumerate(self.up):
            if i != 0:
                x += layers[n - i]
            x = lay(x)
        x = self.end(x + layers[0])
        return x  # (x, means, stds)
