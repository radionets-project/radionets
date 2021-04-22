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


def compress_image(img):
    part1, part2, part3, part4 = split_parts(img, pad=True)

    part1_1, _, part1_3, _ = split_parts(part1, pad=False)
    part2_1, part2_2, part2_3, part2_4 = split_parts(part2, pad=False)
    part3_1, part3_2, _, _ = split_parts(part3, pad=False)
    part4_1, part4_2, _, _ = split_parts(part4, pad=False)

    return (
        part1_1,
        part1_3,
        part2_1,
        part2_2,
        part2_3,
        part2_4,
        part3_1,
        part3_2,
        part4_1,
        part4_2,
    )


def expand_image(params):
    (
        part1_1,
        part1_3,
        part2_1,
        part2_2,
        part2_3,
        part2_4,
        part3_1,
        part3_2,
        part4_1,
        part4_2,
    ) = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
    )
    bs = part1_1.shape[0]
    part1 = combine_parts(
        [
            part1_1,
            -torch.rot90(part1_1, 2, dims=(2, 3)),
            part1_3,
            -torch.rot90(part1_3, 2, dims=(2, 3)),
            (bs, 1, 32, 32),
            False,
        ]
    )

    part2 = combine_parts([part2_1, part2_2, part2_3, part2_4, (bs, 1, 32, 32), False])

    part3 = combine_parts(
        [
            part3_1,
            part3_2,
            F.pad(
                input=-torch.rot90(part3_2[:, :, :, :-1], 2, dims=(2, 3)),
                pad=(0, 1, 0, 0),
                mode="constant",
                value=0,
            ),
            -torch.rot90(part3_1, 2, dims=(2, 3)),
            (bs, 1, 32, 32),
            False,
        ]
    )

    part4 = combine_parts(
        [
            part4_1,
            part4_2,
            -torch.rot90(part4_1, 2, dims=(2, 3)),
            F.pad(
                input=-torch.rot90(part4_2[:, :, :-1, :], 2, dims=(2, 3)),
                pad=(0, 0, 0, 1),
                mode="constant",
                value=0,
            ),
            (bs, 1, 32, 32),
            False,
        ]
    )

    img = combine_parts([part1, part2, part3, part4, (bs, 1, 63, 63), True])
    return img


def split_parts(img, pad=True):
    t_img = img.clone()
    part1 = t_img[:, 0, 0::2, 0::2]
    part2 = t_img[:, 0, 1::2, 1::2]
    part3 = t_img[:, 0, 0::2, 1::2]
    part4 = t_img[:, 0, 1::2, 0::2]
    if pad:
        # print("Padding done.")
        part2 = F.pad(input=part2, pad=(0, 1, 0, 1), mode="constant", value=0)
        part3 = F.pad(input=part3, pad=(0, 1, 0, 0), mode="constant", value=0)
        part4 = F.pad(input=part4, pad=(0, 0, 0, 1), mode="constant", value=0)
    return (
        part1.unsqueeze(1),
        part2.unsqueeze(1),
        part3.unsqueeze(1),
        part4.unsqueeze(1),
    )


def combine_parts(params):
    part1, part2, part3, part4, img_size, final = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )
    comb = torch.zeros(img_size).cuda()
    if final is True:
        comb[:, :, 0::2, 0::2] = part1
        comb[:, :, 1::2, 1::2] = part2[:, :, :-1, :-1]
        comb[:, :, 0::2, 1::2] = part3[:, :, :, :-1]
        comb[:, :, 1::2, 0::2] = part4[:, :, :-1, :]
    else:
        comb[:, :, 0::2, 0::2] = part1
        comb[:, :, 1::2, 1::2] = part2
        comb[:, :, 0::2, 1::2] = part3
        comb[:, :, 1::2, 0::2] = part4
    return comb


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
    # shift the lower frequencies in the middle
    # axes = tuple(range(arr_fft.ndim))
    # shift = [-(dim // 2) for dim in arr_fft.shape]
    # arr_shift = torch.roll(arr_fft, shift, axes)
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

def phase_range(phase):
    # if isinstance(phase, float):
    #     phase = torch.tensor([phase])
    mult = phase / pi
    mult[mult <= 1] = 0
    mult[mult % 2 <= 1] -= 1
    mult = torch.round(mult / 2)
    mult[(phase / pi > 1) & (mult == 0)] = 1
    phase = phase - mult * 2 * pi
    return phase


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


class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer("sums", torch.zeros(1, nf, 1, 1))
        self.register_buffer("sqrs", torch.zeros(1, nf, 1, 1))
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("factor", torch.tensor(0.0))
        self.register_buffer("offset", torch.tensor(0.0))
        self.batch = 0

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x * x).sum(dims, keepdim=True)
        c = s.new_tensor(x.numel() / nc)
        mom1 = s.new_tensor(1 - (1 - self.mom) / sqrt(bs - 1))
        self.sums.lerp_(s, mom1)
        self.sqrs.lerp_(ss, mom1)
        self.count.lerp_(c, mom1)
        self.batch += bs
        means = self.sums / self.count
        varns = (self.sqrs / self.count).sub_(means * means)
        if bool(self.batch < 20):
            varns.clamp_min_(0.01)
        self.factor = self.mults / (varns + self.eps).sqrt()
        self.offset = self.adds - means * self.factor

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        return x * self.factor + self.offset


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


def load_pre_model(learn, pre_path, visualize=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = Path(pre_path).stem
    print("\nLoad pretrained model: {}\n".format(name_pretrained))
    checkpoint = torch.load(pre_path)

    if visualize:
        learn.load_state_dict(checkpoint["model"])

    else:
        learn.model.load_state_dict(checkpoint["model"])
        learn.opt.load_state_dict(checkpoint["opt"])
        learn.epoch = checkpoint["epoch"]
        learn.loss = checkpoint["loss"]
        learn.recorder.iters = checkpoint["iters"]
        learn.recorder.values = checkpoint["vals"]
        learn.recorder.train_losses = checkpoint["recorder_train_loss"]
        learn.recorder.valid_losses = checkpoint["recorder_valid_loss"]
        learn.recorder.losses = checkpoint["recorder_losses"]
        learn.recorder.lrs = checkpoint["recorder_lrs"]


def save_model(learn, model_path):
    # print(learn.model.generator)
    torch.save(
        {   
            "model": learn.model.state_dict(),
            "opt": learn.opt.state_dict(),
            "epoch": learn.epoch,
            "loss": learn.loss,
            "iters": learn.recorder.iters,
            "vals": learn.recorder.values,
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


def create_rot_mat(alpha):
    rot_mat = torch.tensor(
        [[torch.cos(alpha), -torch.sin(alpha)], [torch.sin(alpha), torch.cos(alpha)]]
    )
    rot_mat = rot_mat.cuda()
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot, center=None):
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(torch.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2
    gauss = flux * torch.exp(
        -((x_0 - x) ** 2 / (2 * (x_fwhm) ** 2) + (y_0 - y) ** 2 / (2 * (y_fwhm) ** 2))
    )
    gauss = gauss.cuda()
    return gauss


def create_grid(pixel):
    x = torch.linspace(0, pixel - 1, steps=pixel)
    y = torch.linspace(0, pixel - 1, steps=pixel)
    X, Y = torch.meshgrid(x, y)
    X = X.cuda()
    Y = Y.cuda()
    X.unsqueeze_(0)
    Y.unsqueeze_(0)
    mesh = torch.cat((X, Y))
    x = torch.zeros(X.shape) + 1e-10
    # grid = torch.tensor((x))
    grid = x.clone().detach()
    grid = grid.cuda()
    grid = torch.cat((grid, mesh))
    return grid


def gauss_valid(params):  # setzt aus den einzelen parametern (54) ein bild zusammen
    gauss_param = torch.split(params, 9)
    grid = create_grid(63)
    source = torch.tensor((grid[0]))
    source = grid.clone().detach()
    for i in range(len(gauss_param)):
        cent = torch.tensor(
            [
                len(grid[0]) // 2 + gauss_param[1][i],
                len(grid[0]) // 2 + gauss_param[2][i],
            ]
        )
        cent = cent.cuda()
        s = gaussian_component(
            grid[1],
            grid[2],
            gauss_param[0][i],
            gauss_param[3][i],
            gauss_param[4][i],
            rot=gauss_param[5][i],
            center=cent,
        )
        source = torch.add(source, s)
    return source


def vaild_gauss_bs(in_put):
    for i in range(in_put.shape[0]):
        if i == 0:
            source = gauss_valid(in_put[i])  # gauss parameter des ersten gausses
            source.unsqueeze_(0)
        else:
            h = gauss_valid(in_put[i])
            h.unsqueeze_(0)
            source = torch.cat((source, h))
    return source


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
            nn.Conv2d(ni, nf, 3, stride=stride, padding=1),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
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