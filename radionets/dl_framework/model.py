import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from pathlib import Path
from math import sqrt, pi


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def reshape(x):
    return x.reshape(-1, 2, 63, 63)


def fft(x):
    """
    Layer that performs a fast Fourier-Transformation.
    """
    img_size = x.size(1) // 2
    # sort the incoming tensor in real and imaginary part
    arr_real = x[:, 0:img_size].reshape(-1, int(sqrt(img_size)), int(sqrt(img_size)))
    arr_imag = x[:, img_size:].reshape(-1, int(sqrt(img_size)), int(sqrt(img_size)))
    arr = torch.stack((arr_real, arr_imag), dim=-1)
    # perform fourier transformation and switch imaginary and real part
    arr_fft = torch.ifft(arr, 2).permute(0, 3, 2, 1).transpose(2, 3)
    return arr_fft


def shape(x):
    print(x.shape)
    return x


def euler(x):
    img_size = x.size(1) // 2
    arr_amp = x[:, 0:img_size]
    arr_phase = x[:, img_size:]

    arr_real = (10 ** (10 * (arr_amp - 1)) - 1e-10) * torch.cos(arr_phase)
    arr_imag = (10 ** (10 * (arr_amp - 1)) - 1e-10) * torch.sin(arr_phase)

    arr = torch.stack((arr_real, arr_imag), dim=-1).permute(0, 2, 1)
    return arr


def flatten(x):
    return x.reshape(x.shape[0], -1)


def flatten_with_channel(x):
    return x.reshape(x.shape[0], x.shape[1], -1)


def cut_off(x):
    a = x.clone()
    a[a <= 1e-10] = 1e-10
    return a


def symmetry(x, mode="real"):
    center = (x.shape[1]) // 2
    u = torch.arange(center)
    v = torch.arange(center)

    diag1 = torch.arange(center, x.shape[1])
    diag2 = torch.arange(center, x.shape[1])
    diag_indices = torch.stack((diag1, diag2))
    grid = torch.tril_indices(x.shape[1], x.shape[1], -1)

    x_sym = torch.cat(
        (grid[0].reshape(-1, 1), diag_indices[0].reshape(-1, 1)),
    )
    y_sym = torch.cat(
        (grid[1].reshape(-1, 1), diag_indices[1].reshape(-1, 1)),
    )
    x = torch.rot90(x, 1, dims=(1, 2))
    i = center + (center - x_sym)
    j = center + (center - y_sym)
    u = center - (center - x_sym)
    v = center - (center - y_sym)
    if mode == "real":
        x[:, i, j] = x[:, u, v]
    if mode == "imag":
        x[:, i, j] = -x[:, u, v]
    return torch.rot90(x, 3, dims=(1, 2))


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
    for l in m.children():
        init_cnn_(l, f)


def init_cnn(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f)


def linear(in_features, out_features):
    lin = (nn.Linear(in_features, out_features),)
    bn = (nn.BatchNorm1d(out_features),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*lin, *bn, act]
    return layers


def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=not bn),
        GeneralRelu(**kwargs),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def conv(ni, nc, ks, stride=1, padding=0, dilation=1, groups=1):
    conv = (nn.Conv2d(ni, nc, ks, stride=stride, padding=padding, dilation=dilation, groups=groups),)
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


def double_conv(ni, nc, ks=3, stride=1, padding=1):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = (nn.ReLU(inplace=True),)
    conv2 = (nn.Conv2d(nc, nc, ks, stride, padding),)
    bn2 = (nn.BatchNorm2d(nc),)
    act2 = nn.ReLU(inplace=True)
    layers = [*conv, *bn, *act, *conv2, *bn2, act2]
    return layers


def deconv(ni, nc, ks, stride, padding, out_padding):
    conv = (nn.ConvTranspose2d(ni, nc, ks, stride, padding, out_padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
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
                kernel_size ** 2,
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


class MultiBlock(nn.Module):
    def __init__(self, ni, nc, n=1):
        super().__init__()
        self.conv = MultiBlockConv(ni, nc)
        self.block = nn.Sequential(*(MultiBlockConv(nc, nc) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv(x)
        if self.block is not None:
            x = self.block(x)
        return x


class MultiBlockConv(nn.Module):
    def __init__(self, ni, nc):
        super(MultiBlockConv, self).__init__()
        self.ni = ni
        self.nc = nc

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = nn.BatchNorm2d(ni) if ni == nc else None
        self.rbr_dense = conv_bn(ni, nc, kernel_size=3, padding=1)
        self.rbr_1x1 = conv_bn(ni, nc, kernel_size=1)

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
