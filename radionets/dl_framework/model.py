from numpy.lib.function_base import diff
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from pathlib import Path
from math import sqrt, pi
from fastcore.foundation import L
from torch.nn.common_types import  _size_4_t
import numpy as np
import radionets.simulations.utils as utils


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

def better_symmetry(x):
    # rotation
    x = torch.flip(x, [3])

    # indices of upper and lower triangle
    triu = torch.triu_indices(x.shape[2], x.shape[3], 1)
    tril = torch.tril_indices(x.shape[2], x.shape[3], -1)
    triu = torch.flip(triu, [1])

    # sym amp and phase
    x[:,0,tril[0], tril[1]] = x[:,0, triu[0], triu[1]]
    x[:,1,tril[0], tril[1]] = -x[:,1, triu[0], triu[1]]

    # rotation
    x = torch.flip(x, [3])

    return x

def tf_shift(x):
    triu = torch.triu_indices(x.shape[2], x.shape[2], 0)
    tf = torch.flip(x, [3])[:,:,triu[0], triu[1]].reshape(x.shape[0],x.shape[1],x.shape[2],int(x.shape[3]/2)+1)

    return tf

def btf_shift(x):
    btf = torch.zeros((x.shape[0],x.shape[1],x.shape[2], x.shape[3]*2-1)).cuda()
    triu = torch.triu_indices(x.shape[2], x.shape[2], 0)

    btf[:,:,triu[0], triu[1]] = x[:,:].reshape(x.shape[0], x.shape[1], -1)
    btf = torch.flip(btf, [3])

    btf = better_symmetry(btf)
    return btf

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


def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=not bn),
        GeneralRelu(**kwargs),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


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


def depth_conv(ni, nc, ks, stride, padding, dilation):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding, dilation=dilation, groups=ni),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
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


def load_pre_model(learn, pre_path, visualize=False, gan=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = Path(pre_path).stem
    print(f"\nLoad pretrained model: {name_pretrained}\n")
    if torch.cuda.is_available():
        checkpoint = torch.load(pre_path)
    else:
        checkpoint = torch.load(pre_path, map_location=torch.device("cpu"))

    if visualize:
        learn.load_state_dict(checkpoint["model"])
    else:
        learn.model.load_state_dict(checkpoint["model"])
        learn.opt.load_state_dict(checkpoint["opt"])
        learn.epoch = checkpoint["epoch"]
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
        learn.recorder.iters = checkpoint["iters"]
        learn.recorder.values = checkpoint["vals"]


def save_model(learn, model_path, gan=False):
    # print(learn.model.generator)
    if not gan:
        torch.save(
            {   
                "model": learn.model.state_dict(),
                "opt": learn.opt.state_dict(),
                "epoch": learn.epoch,
                "loss": learn.loss,
                "iters": learn.recorder.iters,
                "vals": learn.recorder.values,
                "train_loss": learn.avg_loss.loss_train,
                "valid_loss": learn.avg_loss.loss_valid,
                "lrs": learn.avg_loss.lrs,
                "recorder_train_loss": L(learn.recorder.values[0:]).itemgot(0),
                "recorder_valid_loss": L(learn.recorder.values[0:]).itemgot(1),
                "recorder_losses": learn.recorder.losses,
                "recorder_lrs": learn.recorder.lrs,
            },
            model_path,
        )
    else:
        torch.save(
            {   
                "model": learn.model.generator.state_dict(),
                "opt": learn.opt.state_dict(),
                "epoch": learn.epoch,
                "loss": learn.loss,
                "iters": learn.recorder.iters,
                "vals": learn.recorder.values,
                "train_loss": learn.avg_loss.loss_train,
                "valid_loss": learn.avg_loss.loss_valid,
                "lrs": learn.avg_loss.lrs,
                "recorder_train_loss": L(learn.recorder.values[0:]).itemgot(0),
                "recorder_valid_loss": L(learn.recorder.values[0:]).itemgot(1),
                "recorder_losses": learn.recorder.losses,
                "recorder_lrs": learn.recorder.lrs,
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

class SRBlock_noBias(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = self._conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else nn.Conv2d(ni, nf, 1,bias=False)
        self.pool = (
            nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        )  # nn.AvgPool2d(8, 2, ceil_mode=True)

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1,bias=False),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(nf),
        )

class SRBlockPad(nn.Module):
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
            BetterShiftPad((1,1,1,1)),
            nn.Conv2d(ni, nf, 3, stride=stride, padding=0),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            BetterShiftPad((1,1,1,1)),
            nn.Conv2d(nf, nf, 3, stride=1, padding=0),
            nn.BatchNorm2d(nf),
        )

class EDSRBaseBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = self._conv_block(ni,nf,stride)
        self.idconv = nn.Identity() if ni == nf else nn.Conv2d(ni, nf, 1)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)#nn.AvgPool2d(8, 2, ceil_mode=True)

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1),
            nn.PReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1)
        )

class RDB(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.conv1 = self._conv_block(ni,nf,stride)
        self.conv2 = self._conv_block(ni+nf,nf,stride)
        self.conv3 = self._conv_block(ni+2*nf,nf,stride)
        self.conv4 = self._conv_block(ni+3*nf,nf,stride)
        self.conv5 = self._conv_block(ni+4*nf,nf,stride)
        self.conv6 = self._conv_block(ni+5*nf,nf,stride)

        self.convFusion = nn.Conv2d(ni+6*nf, ni, 1, stride=1, padding=0, groups=2, bias=False)

    def forward(self, x):
        x1_c = self.conv1(x)
        cat = self._cat_split(x, x1_c)
        x2_c = self.conv2(cat)
        cat = self._cat_split(cat, x2_c)
        x3_c = self.conv3(cat)
        cat = self._cat_split(cat, x3_c)
        x4_c = self.conv4(cat)
        cat = self._cat_split(cat, x4_c)
        x5_c = self.conv5(cat)
        cat = self._cat_split(cat, x5_c)
        x6_c = self.conv6(cat)
        cat = self._cat_split(cat, x6_c)


        return self.convFusion(cat) + x

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1, bias=False),
            nn.PReLU()
        )
    
    def _cat_split(self, x, y):
        x1, x2 = torch.chunk(x,2, dim=1)
        y1, y2 = torch.chunk(y,2, dim=1)
        return torch.cat((x1,y1,x2,y2), dim=1)


class FBB(nn.Module):
    def __init__(self, ni, nf, stride=1, first=False):
        super().__init__()
        self.first = first
        if first:
            self.convCat = nn.Conv2d(ni*2, ni, 1, stride=1, padding=0, groups=2, bias=False)
        else:
            self.convCat = nn.Identity()
        self.conv1 = self._conv_block(ni,nf,stride)
        self.conv2 = self._conv_block(ni+nf,nf,stride)
        self.conv3 = self._conv_block(ni+2*nf,nf,stride)
        self.conv4 = self._conv_block(ni+3*nf,nf,stride)
        self.conv5 = self._conv_block(ni+4*nf,nf,stride)
        self.conv6 = self._conv_block(ni+5*nf,nf,stride)

        self.convFusion = nn.Conv2d(ni+6*nf, ni, 1, stride=1, padding=0, groups=2, bias=False)

    def forward(self, x):
        # if self.first:
        #     comb = torch.chunk(x,2, dim=1)
        #     skip = comb[0]
        #     x = self._cat_split(comb[0], comb[1])
        #     # x = self._cat_split(x, comb[2])
        #     # x = self._cat_split(x, comb[3])
        # else:
        #     skip = x

        x_cc = self.convCat(x)
        x1_c = self.conv1(x_cc)
        cat = self._cat_split(x_cc, x1_c)
        x2_c = self.conv2(cat)
        cat = self._cat_split(cat, x2_c)
        x3_c = self.conv3(cat)
        cat = self._cat_split(cat, x3_c)
        x4_c = self.conv4(cat)
        cat = self._cat_split(cat, x4_c)
        x5_c = self.conv5(cat)
        cat = self._cat_split(cat, x5_c)
        x6_c = self.conv6(cat)
        cat = self._cat_split(cat, x6_c)


        return self.convFusion(cat) + x_cc

    def _conv_block(self, ni, nf, stride):
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1, bias=False),
            nn.PReLU()
        )
    
    def _cat_split(self, x, y):
        x1, x2 = torch.chunk(x,2, dim=1)
        y1, y2 = torch.chunk(y,2, dim=1)
        return torch.cat((x1,y1,x2,y2), dim=1)


class _CirculationPadNd(nn.Module):
    __constants__ = ['padding']

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.pad(input, self.padding, 'circular')

    def extra_repr(self) -> str:
        return '{}'.format(self.padding)

class CirculationPad2d(_CirculationPadNd):
    padding: _size_4_t

    def __init__(self, padding: _size_4_t) -> None:
        super(CirculationPad2d, self).__init__()
        self.padding = _pair(padding)

class CirculationShiftPad(nn.Module):
    padding: _size_4_t

    def __init__(self, padding: _size_4_t) -> None:
        super(CirculationShiftPad, self).__init__()
        self.padding = _pair(padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.pad(input, self.padding, 'circular')
        x[...,:self.padding[2],:] = 0
        x[...,-self.padding[3]:,:] = 0
        x[...,:,:self.padding[0]] = torch.roll(x[...,:,:self.padding[0]],1,2)
        x[...,:,-self.padding[1]:] = torch.roll(x[...,:,-self.padding[1]:],-1,2)
        x[...,:self.padding[2],:] = 0
        x[...,-self.padding[3]:,:] = 0
        return x

def better_padding(input, padding):
    in_shape = input.shape
    paddable_shape = in_shape[2:]
    
    out_shape = in_shape[:2]
    for idx, size in enumerate(paddable_shape):
        out_shape += (size + padding[-(idx * 2 + 1)] + padding[-(idx * 2 + 2)],)
    
    # fill empty tensor of new shape with input
    out = torch.zeros(out_shape, dtype=input.dtype, layout=input.layout,
                      device=input.device)
    
    out[..., padding[-2]:(out_shape[2]-padding[-1]), padding[-4]:(out_shape[3]-padding[-3])] = input
    
    # pad left
    i0 = out_shape[3] - padding[-4] - padding[-3]
    i1 = out_shape[3] - padding[-3]
    o0 = 0
    o1 = padding[-4]
    out[:, :, padding[-2]:out_shape[2]-padding[-1], o0:o1] = out[:, :, padding[-2]-1:out_shape[2]-padding[-1]-1, i0:i1]
    
    # pad right
    i0 = padding[-4]
    i1 = padding[-4] + padding[-3]
    o0 = out_shape[3] - padding[-3]
    o1 = out_shape[3]
    out[:, :, padding[-2]:out_shape[2]-padding[-1], o0:o1] = out[:, :, padding[-2]+1:out_shape[2]-padding[-1]+1, i0:i1]
    
    return out

class BetterShiftPad(nn.Module):
    padding: _size_4_t

    def __init__(self, padding: _size_4_t) -> None:
        super(BetterShiftPad, self).__init__()
        self.padding = _pair(padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = better_padding(input, self.padding)
        return x

class HardDC(nn.Module):
    def __init__(self, base_nums, n_tel):
        super().__init__()
        self.base_nums = torch.zeros(base_nums)
        self.n_tel = n_tel
        self.weights = nn.Parameter(torch.tensor(1).float())
        
    def forward(self, x, input, A, base_mask):
        c = 0
        for i in range(self.n_tel):
            for j in range(self.n_tel):
                if j<=i:
                    continue
                self.base_nums[c] = 256 * (i + 1) + j + 1
                c += 1


        pred = torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]), dtype=torch.complex64).to('cuda')
        c = 0
        for idx, bn in enumerate(self.base_nums):
            s_uv = torch.sum((base_mask == bn),3)
            if not (base_mask == bn).any():
                continue

            xA = torch.einsum('bclm,blm->bclm',x,A[...,idx])
            x_prime = xA[:,0] + 1j*xA[:,1] #from 2 channels to complex for fft
            k_prime = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(x_prime)))
            y_prime = torch.einsum('blm,bclm->bclm',(1-s_uv),k_prime.unsqueeze(1))

            Y = torch.einsum('blm,bclm->bclm', s_uv, input)

            full_k_space = Y + self.weights*y_prime

            pred += full_k_space # maybe a conj(A) missing, see paper 1910.07048
            c += 1

        points = base_mask.clone()
        points[points != 0] = 1
        points = torch.sum(points,3)
        points[points == 0] = 1

        
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(pred/c))) #divide by c because we summed c fully sampled maps in pred??? 

class SoftDC(nn.Module):
    def __init__(self, base_nums, n_tel):
        super().__init__()
        self.base_nums = torch.zeros(base_nums)
        self.n_tel = n_tel
        self.weights = nn.Parameter(torch.tensor(1).float())
        
    def forward(self, x, measured, A, base_mask):
        c = 0
        for i in range(self.n_tel):
            for j in range(self.n_tel):
                if j<=i:
                    continue
                self.base_nums[c] = 256 * (i + 1) + j + 1
                c += 1


        sum = torch.zeros((x.shape[0],1,x.shape[2],x.shape[3]), dtype=torch.complex64).to('cuda')
        c = 0
        for idx, bn in enumerate(self.base_nums):
            s_uv = torch.sum((base_mask == bn),3)
            if not (base_mask == bn).any():
                continue

            xA = torch.einsum('bclm,blm->bclm',x,A[...,idx])
            x_prime = xA[:,0] + 1j*xA[:,1] #from 2 channels to complex for fft
            k_prime = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(x_prime)))
            y_prime = torch.einsum('blm,bclm->bclm',s_uv,k_prime.unsqueeze(1))

            # Y = torch.einsum('blm,bclm->bclm', s_uv, input)

            diff = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(y_prime))) 

            sum += diff # maybe a conj(A) missing, see paper 1910.07048
            c += 1

        sum = sum/c

        pred = torch.zeros(x.shape).to('cuda')

        pred[:,0] = sum.real.squeeze(1)
        pred[:,1] = sum.imag.squeeze(1)




        
        return x + self.weights*(pred-measured) #divide by c because we summed c fully sampled maps in pred??? 


def calc_DirtyBeam(base_mask):
    s_uv = torch.sum(base_mask,3)
    s_uv[s_uv != 0] = 1

    b = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(s_uv)))
    beam = torch.zeros((b.shape[0],2, b.shape[1], b.shape[2])).to('cuda')
    beam[:,0] = b.real.squeeze(1)
    beam[:,1] = b.imag.squeeze(1)
    return beam


def gauss(kernel_size, sigma):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).to('cuda')
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel

    
class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        self.Wih = nn.Conv2d(input_size, 3*hidden_size, kernel_size, dilation=dilation, bias=bias, padding=dilation*(kernel_size-1)//2)
        self.Whh = nn.Conv2d(input_size, 3*hidden_size, kernel_size, dilation=dilation, bias=bias, padding=dilation*(kernel_size-1)//2)

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros((x.size(0), self.hidden_size) + x.size()[2:], requires_grad=False).to('cuda')
        
        ih = self.Wih(x).chunk(3, dim=1)
        hh = self.Whh(hx).chunk(3, dim=1)

        z = torch.sigmoid(ih[0] + hh[0])
        r = torch.sigmoid(ih[1] + hh[1])
        n = torch.tanh(ih[2]+ r*hh[2])

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.abs(hx[0,0]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        hx = (1-z)*hx + z*n

        return hx

class ConvGRUCellBN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        self.Wih = nn.Conv2d(input_size, 3*hidden_size, kernel_size, dilation=dilation, bias=bias, padding=dilation*(kernel_size-1)//2)
        self.Whh = nn.Conv2d(input_size, 3*hidden_size, kernel_size, dilation=dilation, bias=bias, padding=dilation*(kernel_size-1)//2)

        self.bn1 = nn.BatchNorm2d(3*hidden_size)
        self.bn2 = nn.BatchNorm2d(3*hidden_size)

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros((x.size(0), self.hidden_size) + x.size()[2:], requires_grad=False).to('cuda')
        
        ih = self.bn1(self.Wih(x)).chunk(3, dim=1)
        hh = self.bn2(self.Whh(hx)).chunk(3, dim=1)

        z = torch.sigmoid(ih[0] + hh[0])
        r = torch.sigmoid(ih[1] + hh[1])
        n = torch.tanh(ih[2]+ r*hh[2])

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.abs(hx[0,0]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        hx = (1-z)*hx + z*n

        return hx

def gradFunc(x, y, A, base_mask, n_tel, base_nums): 
    does_require_grad = x.requires_grad
    with torch.enable_grad():
        x.requires_grad_(True)

        mask = torch.sum(base_mask, 3)
        mask[mask != 0] = 1

        fx = torch.fft.fft2(torch.fft.fftshift(x[:,0]+1j*x[:,1])) # shift x low freq to corner & fft
        pfx = torch.einsum('blm,blm->blm', mask, torch.fft.ifftshift(fx)) # shift low freq to center
        py = torch.einsum('blm,blm->blm', mask, y)

        # import matplotlib.pyplot as plt
        # plt.imshow(torch.absolute(py[0]-pfx[0]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        diff = (py-pfx)**2
        # import matplotlib.pyplot as plt
        # plt.imshow(torch.absolute(py[0]-pfx[0]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        # diff_shift = torch.fft.fftshift(diff) # shift low freq to corner

        # error = torch.sum(torch.fft.ifftshift(torch.fft.ifft2(diff_shift))) # ifft & shift low freq to center

        # grad = torch.zeros((ift.size(0), 2) + ift.size()[1:]).to('cuda')
        # grad[:,0] = ift.real.squeeze(1)
        # grad[:,1] = ift.imag.squeeze(1)

        grad_x = torch.autograd.grad(torch.sum(diff), inputs=x, retain_graph=does_require_grad,
                                     create_graph=does_require_grad)[0]

        
        # import matplotlib.pyplot as plt
        # plt.imshow(np.absolute((grad_x[:,0]+1j*grad_x[:,1])[0].cpu().detach().numpy()))
        # plt.colorbar()
        # plt.show()
        # import matplotlib.pyplot as plt
        # plt.imshow(np.absolute(torch.fft.ifftshift(torch.fft.ifft(torch.fft.fftshift(grad_x)))[0].cpu().detach().numpy()))
        # plt.colorbar()
        # plt.show()



    # import matplotlib.pyplot as plt
    # # print(grad_x.shape)
    # plt.imshow(torch.absolute(diff[0]).cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()
    x.requires_grad_(does_require_grad)

    return grad_x


def gradFunc2(x, y, A, base_mask, n_tel, base_nums): 
    

    mask = torch.sum(base_mask, 3)
    mask[mask != 0] = 1

    fx = torch.fft.fft2(torch.fft.fftshift(x[:,0]+1j*x[:,1])) # shift x low freq to corner & fft
    pfx = torch.einsum('blm,blm->blm', mask, torch.fft.ifftshift(fx)) # shift low freq to center
    py = torch.einsum('blm,blm->blm', mask, y) # mask y otherwise diff is not zero if x=y since we do a lot of ffts

    diff = pfx-py
    diff_shift = torch.fft.fftshift(diff) # shift low freq to corner
    error = torch.fft.ifftshift(torch.fft.ifft2(diff_shift))


    grad = torch.zeros((error.size(0), 2) + error.size()[1:]).to('cuda')
    grad[:,0] = error.real.squeeze(1)
    grad[:,1] = error.imag.squeeze(1)

    # import matplotlib.pyplot as plt
    # # # print(grad_x.shape)
    # plt.imshow(torch.absolute(error[0]).cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()

    return grad

def gradFunc_putzky(x, y): 
    base_mask = y[1]
    data = y[0]
    does_require_grad = x.requires_grad
    with torch.enable_grad():
        x.requires_grad_(True)

        mask = torch.sum(base_mask, 3)
        mask[mask != 0] = 1
        
        fx = torch.fft.fft2(torch.fft.fftshift(x), norm="forward")
       
        # import matplotlib.pyplot as plt
        # plt.imshow((torch.abs(torch.fft.ifftshift(torch.fft.ifft2(fx))))[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        pfx = torch.einsum('blm,bclm->bclm', torch.flip(mask, [1]), torch.fft.ifftshift(fx)) # shift low freq to center
        # py = torch.einsum('blmforward,bclm->bclm', mask, data)
        # plt.imshow(torch.abs(pfx-data)[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        difference = pfx-data
        # import matplotlib.pyplot as plt
        # plt.imshow(abs(difference[0,0].cpu().detach().numpy()))
        # plt.colorbar()
        # plt.show()

        # import matplotlib.pyplot as plt
        # plt.imshow((torch.abs(torch.fft.ifftshift(torch.fft.ifft2(data))-torch.fft.ifftshift(torch.fft.ifft2(pfx))))[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        #import matplotlib.pyplot as plt
        #plt.imshow((torch.abs(torch.fft.ifftshift(torch.fft.ifft2(data))))[0,0].cpu().detach().numpy())
        #plt.colorbar()
        #plt.show()

        #import matplotlib.pyplot as plt
        #plt.imshow((torch.abs(torch.fft.ifftshift(torch.fft.ifft2(fx))))[0,0].cpu().detach().numpy())
        #plt.colorbar()
        #plt.show()


        chi2 = torch.sum(torch.square(torch.abs(difference)))


        grad_x = torch.autograd.grad(chi2, inputs=x, retain_graph=does_require_grad,
                                     create_graph=does_require_grad)[0]

        # import matplotlib.pyplot as plt
        # # # print(grad_x.shape)
        # # test[test==0] = 1
        # # plt.figure(figsize=(12,8))
        # plt.imshow((torch.abs(grad_x))[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

    x.requires_grad_(does_require_grad)

    return grad_x

def manual_grad(x, y): 
    base_mask = y[1]
    data = y[0]

    mask = torch.sum(base_mask, 3)
    mask[mask != 0] = 1
    
    fx = torch.fft.fft2(torch.fft.fftshift(x), norm="forward")
    
    # import matplotlib.pyplot as plt
    # plt.imshow((torch.abs(torch.fft.ifftshift(torch.fft.ifft2(fx))))[0,0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()
    pfx = torch.einsum('blm,bclm->bclm', torch.flip(mask, [1]), torch.fft.ifftshift(fx)) # shift low freq to center
    # py = torch.einsum('blmforward,bclm->bclm', mask, data)
    # plt.imshow(torch.abs(pfx-data)[0,0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()
    difference = pfx-data

    # import matplotlib.pyplot as plt
    # plt.imshow((torch.abs(torch.fft.ifftshift(torch.fft.ifft2(data))-torch.fft.ifftshift(torch.fft.ifft2(pfx))))[0,0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()

    grad_x = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(difference), norm='forward'))

    # import matplotlib.pyplot as plt
    # # # print(grad_x.shape)
    # # test[test==0] = 1
    # # plt.figure(figsize=(12,8))
    # plt.imshow((torch.abs(grad_x))[0,0].cpu().detach().numpy())
    # plt.colorbar()
    # plt.show()

    return grad_x

def fft_conv(a,b):
    multiply = (torch.fft.fft2(torch.fft.fftshift(a))*torch.fft.fft2(torch.fft.fftshift(b), norm="ortho"))
    ifft =torch.fft.ifftshift(torch.fft.ifft2(multiply))
    import matplotlib.pyplot as plt
    # plt.imshow(abs(ifft[0,0].cpu().detach().numpy()))
    # plt.colorbar()
    # plt.show()
    return ifft
