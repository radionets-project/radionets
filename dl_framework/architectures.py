from torch import nn
from dl_framework.model import conv, Lambda, flatten, shape, fft


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
        nn.Linear(64, 4096),
        Lambda(fft),
        Lambda(shape),
    )
    return arch
