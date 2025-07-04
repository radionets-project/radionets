import torch
from torch import nn
from torch.nn.modules.utils import _pair

from radionets.dl_framework.architecture import GeneralELU, SRResNet_34


class Uncertainty(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, 9, stride=1, padding=4, groups=2),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 9, stride=1, padding=4, groups=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            LocallyConnected2d(
                64,
                2,
                [img_size // 2 + 1, img_size],
                1,
                stride=1,
                bias=False,
            )
        )

        self.elu = GeneralELU(add=+(1 + 1e-7))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.final(x)

        return self.elu(x)


class UncertaintyWrapper(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.pred = SRResNet_34()

        self.uncertainty = Uncertainty(img_size)

    def forward(self, x):
        inp = x.clone()

        pred = self.pred(x)

        # x = torch.abs(pred - inp)
        x = torch.cat([pred, inp], dim=1)

        unc = self.uncertainty(x)

        val = torch.cat(
            [
                pred[:, 0].unsqueeze(1),
                unc[:, 0].unsqueeze(1),
                pred[:, 1].unsqueeze(1),
                unc[:, 1].unsqueeze(1),
            ],
            dim=1,
        )

        return val


class LocallyConnected2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride, bias=False
    ):
        super(LocallyConnected2d, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size**2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
