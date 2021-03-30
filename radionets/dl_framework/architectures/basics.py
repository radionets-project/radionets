from torch import nn
from radionets.dl_framework.model import (
    conv,
    Lambda,
    flatten,
    fft,
    deconv,
    GeneralRelu,
)


def cnn():
    """
    conv-layer: number of entry channels, number of exit channels,
                kerner size, stride, padding
    """
    arch = nn.Sequential(
        *conv(1, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        Lambda(flatten),
        nn.Linear(64, 8192),
        Lambda(fft),
        Lambda(flatten),
        # *conv(2, 1, 1, 1, 0),
        nn.Linear(8192, 4096),
        # Lambda(flatten),
    )
    return arch


def autoencoder():
    arch = nn.Sequential(
        *conv(1, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        nn.ConvTranspose2d(64, 32, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(32),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(32, 16, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(16),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(16, 16, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(16),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(16, 8, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(8),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(8, 4, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(4),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(4, 1, (3, 3), 2, 1, 1),
        Lambda(flatten),
    )
    return arch


def autoencoder_two_channel():
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
        nn.Linear(2, 4096),
    )
    return arch
