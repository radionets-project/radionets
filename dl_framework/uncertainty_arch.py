from torch import nn
from dl_framework.model import GeneralELU, LocallyConnected2d
from math import pi


def block_1():
    layers = [
        nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(23, 23),
            stride=1,
            padding=11,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(4),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(21, 21),
            stride=1,
            padding=10,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(8),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=(17, 17),
            stride=1,
            padding=8,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(12),
        GeneralELU(add=-(pi-1), maxv=pi),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi-1), maxv=pi),
    ]
    return nn.Sequential(*layers)


def block_2():
    layers = [
        nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(4),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(5, 5),
            stride=1,
            padding=3,
            dilation=2,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(8),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(12),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=12,
            out_channels=20,
            kernel_size=(3, 3),
            stride=1,
            padding=3,
            dilation=2,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(20),
        GeneralELU(add=-(pi-1), maxv=pi),
        LocallyConnected2d(20, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi-1), maxv=pi),
    ]
    return nn.Sequential(*layers)


def block_3():
    layers = [
        nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(4),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(8),
        GeneralELU(add=-(pi-1), maxv=pi),
        nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3),
            stride=1,
            padding=2,
            dilation=2,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(12),
        GeneralELU(add=-(pi-1), maxv=pi),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi-1), maxv=pi),
    ]
    return nn.Sequential(*layers)


def block_1_unc():
    layers = [
        nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(23, 23),
            stride=1,
            padding=11,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(4),
        GeneralELU(add=+1+1e-5),
        nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(21, 21),
            stride=1,
            padding=10,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(8),
        GeneralELU(add=+1+1e-5),
        nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=(17, 17),
            stride=1,
            padding=8,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(12),
        GeneralELU(add=+1+1e-5),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1+1e-5),
    ]
    return nn.Sequential(*layers)


def block_2_unc():
    layers = [
        nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(4),
        GeneralELU(add=+1+1e-5),
        nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(5, 5),
            stride=1,
            padding=3,
            dilation=2,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(8),
        GeneralELU(add=-(pi-1)),
        nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(12),
        GeneralELU(add=+1+1e-5),
        nn.Conv2d(
            in_channels=12,
            out_channels=20,
            kernel_size=(3, 3),
            stride=1,
            padding=3,
            dilation=2,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(20),
        GeneralELU(add=+1+1e-5),
        LocallyConnected2d(20, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1+1e-5),
    ]
    return nn.Sequential(*layers)


def block_3_unc():
    layers = [
        nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(4),
        GeneralELU(add=+1+1e-5),
        nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(8),
        GeneralELU(add=+1+1e-5),
        nn.Conv2d(
            in_channels=8,
            out_channels=12,
            kernel_size=(3, 3),
            stride=1,
            padding=2,
            dilation=2,
            padding_mode="zeros",
            bias=False,
        ),
        nn.BatchNorm2d(12),
        GeneralELU(add=+1+1e-5),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1+1e-5),
    ]
    return nn.Sequential(*layers)
