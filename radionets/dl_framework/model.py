import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from pathlib import Path
from math import pi


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


class GeneralELU(nn.Module):
    def __init__(self, add=None, maxv=None):
        super().__init__()
        self.add = add
        self.maxv = maxv

    def forward(self, x):
        x = F.elu(x)
        if self.add is not None:
            x = x + self.add
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, "bias", None) is not None:
            m.bias.data.zero_()
    for c in m.children():
        init_cnn_(c, f)


def init_cnn(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f)


def conv(ni, nc, ks, stride, padding):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def conv_amp(ni, nc, ks, stride, padding, dilation):
    """Create a convolutional layer for the amplitude reconstruction.
    The activation function ist ReLU with a 2d Batch normalization.

    Parameters
    ----------
    ni : int
        Number of input channels
    nc : int
        Number of output channels
    ks : tuple
        Size of the kernel
    stride : int
        Stepsize between use of kernel
    padding : int
        Number of pixels added to edges of picture
    dilation : int
        Factor for spreading the receptive field

    Returns
    -------
    list
        list of convolutional layer, 2d Batch Normalisation and Activation function.
    """
    conv = (
        nn.Conv2d(
            ni, nc, ks, stride, padding, dilation, bias=False, padding_mode="replicate"
        ),
    )
    bn = (nn.BatchNorm2d(nc),)
    act = nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def conv_phase(ni, nc, ks, stride, padding, dilation, add):
    """Create a convolutional layer for the amplitude reconstruction.
    The activation function ist GeneralELU with a 2d Batch normalization.

    Parameters
    ----------
    ni : int
        Number of input channels
    nc : int
        Number of output channels
    ks : tuple
        Size of the kernel
    stride : int
        Stepsize between use of kernel
    padding : int
        Number of pixels added to edges of picture
    dilation : int
        Factor for spreading the receptive field
    add : int
        Number which is added to GeneralELU

    Returns
    -------
    list
        list of convolutional layer, 2d Batch Normalisation and Activation function.
    """
    conv = (
        nn.Conv2d(
            ni, nc, ks, stride, padding, dilation, bias=False, padding_mode="replicate"
        ),
    )
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralELU(add)
    layers = [*conv, *bn, act]
    return layers


def load_pre_model(learn, pre_path, visualize=False, plot_loss=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = Path(pre_path).stem
    print(f"\nLoad pretrained model: {name_pretrained}\n")
    if torch.cuda.is_available() and not plot_loss:
        checkpoint = torch.load(pre_path)
    else:
        checkpoint = torch.load(pre_path, map_location=torch.device("cpu"))

    if visualize:
        learn.load_state_dict(checkpoint["model"])
    elif plot_loss:
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
    else:
        learn.model.load_state_dict(checkpoint["model"])
        learn.opt.load_state_dict(checkpoint["opt"])
        learn.epoch = checkpoint["epoch"]
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
        learn.recorder.iters = checkpoint["iters"]
        learn.recorder.values = checkpoint["vals"]


def save_model(learn, model_path):
    torch.save(
        {
            "model": learn.model.state_dict(),
            "opt": learn.opt.state_dict(),
            "epoch": learn.epoch,
            "iters": learn.recorder.iters,
            "vals": learn.recorder.values,
            "train_loss": learn.avg_loss.loss_train,
            "valid_loss": learn.avg_loss.loss_valid,
            "lrs": learn.avg_loss.lrs,
        },
        model_path,
    )


class LocallyConnected2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride, bias=False
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
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


class ResBlock_amp(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = self._conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else nn.Conv2d(ni, nf, 1)
        self.pool = (
            nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        )  # nn.AvgPool2d(8, 2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
        )


class ResBlock_phase(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = self._conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else nn.Conv2d(ni, nf, 1)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        self.relu = GeneralELU(1 - pi)

    def forward(self, x):
        return self.relu(self.convs(x) + self.idconv(self.pool(x)))

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1),
            nn.BatchNorm2d(nf),
            GeneralELU(1 - pi),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
        )


class SRBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = self._conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else nn.Conv2d(ni, nf, 1)
        self.pool = (
            nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        )  # nn.AvgPool2d(8, 2, ceil_mode=True)

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nf),
        )


def symmetry(x):
    if x.shape[-1] % 2 != 0:
        raise ValueError("The symmetry function only works for even image sizes.")
    upper_half = x[:, :, 0 : x.shape[2] // 2, :].clone()
    upper_left = upper_half[:, :, :, 0 : upper_half.shape[3] // 2].clone()
    upper_right = upper_half[:, :, :, upper_half.shape[3] // 2 :].clone()
    a = torch.flip(upper_left, dims=[-2, -1])
    b = torch.flip(upper_right, dims=[-2, -1])

    upper_half[:, :, :, 0 : upper_half.shape[3] // 2] = b
    upper_half[:, :, :, upper_half.shape[3] // 2 :] = a

    x[:, 0, x.shape[2] // 2 :, :] = upper_half[:, 0]
    x[:, 1, x.shape[2] // 2 :, :] = -upper_half[:, 1]
    return x
