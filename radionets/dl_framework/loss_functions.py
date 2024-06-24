import numpy as np
import torch
from torch import nn
from math import pi


def l1(x, y):
    l1 = nn.L1Loss()
    loss = l1(x, y)
    return loss


def create_circular_mask(h, w, center=None, radius=None, bs=64):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return np.repeat([mask], bs, axis=0)

import matplotlib.pyplot as plt
from radionets.evaluation.utils import apply_symmetry
def splitted_L1_fft(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)

    ri_pred = torch.cat([inp_amp[:, None], inp_phase[:, None]], dim=1)
    ri_pred = apply_symmetry({"lol": ri_pred})["lol"]
    ifft_pred = torch.abs(torch.fft.ifft2(ri_pred[:, 0] + 1j * ri_pred[:, 1]))
    ifft_pred[ifft_pred <= 1e-8] = 1e-8
    ifft_pred = torch.log10(ifft_pred)

    ri_true = torch.cat([tar_amp[:, None], tar_phase[:, None]], dim=1)
    ri_true = apply_symmetry({"lol": ri_true})["lol"]
    ifft_true = torch.abs(torch.fft.ifft2(ri_true[:, 0] + 1j * ri_true[:, 1]))
    ifft_true[ifft_true <= 1e-8] = 1e-8
    ifft_true = torch.log10(ifft_true)

    mse = nn.MSELoss()
    loss_ifft = l1(ifft_pred, ifft_true)
    loss = loss_amp * 1e4 + loss_phase * 1e4 + loss_ifft
    #plt.imshow(torch.abs(ifft_pred[0].detach().cpu()))
    #plt.colorbar()
    #plt.savefig("test_pred.png")
    #plt.clf()
    #plt.imshow(torch.abs(ifft_true[0]).detach().cpu())
    #plt.colorbar()
    #plt.savefig("test_true.png")
    #plt.clf()
    return loss


from astropy.convolution import Gaussian2DKernel
def splitted_L1(x, y):
    kernel = Gaussian2DKernel(256, 256, x_size = 512, y_size = 512).array[:257]
    kernel /= kernel.max()
    kernel *= 10
    scale = torch.from_numpy(kernel).to("cuda:0")
    inp_amp = x[:, 0] * scale
    inp_phase = x[:, 1] * scale

    tar_amp = y[:, 0] * scale
    tar_phase = y[:, 1] * scale

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase * 100
    return loss


def MSE(x, y):
    nf = y.amax(dim=(-1, -2), keepdim=True) / 10
    mse = nn.MSELoss()
    return mse(x / nf, y / nf)


def L1(x, y):
    l1 = nn.L1Loss()
    return l1(x, y)

def scaled_MSE(x, y):
    xx = torch.linspace(1, 257, steps=257, device=x.device)
    yy = torch.linspace(1, 512, steps=512, device=x.device)
    scale = torch.meshgrid([xx, yy], indexing="ij")[0]

    nf = y.amax(dim=(-1, -2), keepdim=True) / 10

    mse = nn.MSELoss()
    return mse(x / nf * scale, y / nf * scale)


def scaled_L1(x, y):
    xx = torch.linspace(1, 256, steps=256, device=x.device)
    xx = torch.cat([xx, torch.flip(xx, dims=(0,))])
    yy = torch.linspace(1, 512, steps=512, device=x.device)
    scale = torch.meshgrid([xx, yy], indexing="ij")[0]

    nf = y.amax(dim=(-1, -2), keepdim=True) / 10

    mse = nn.L1Loss()
    return mse(x / nf * scale, y / nf * scale)


def splitted_MSE(x, y):
    inp_amp = x[:, 0]
    inp_phase = x[:, 1]

    tar_amp = y[:, 0]
    tar_phase = y[:, 1]
    #print("x_r", inp_amp.max())
    #print("y_r", tar_amp.max())
    #print("y_p", tar_phase.max())
    #print("y_p", tar_phase.min())

    mse = nn.MSELoss()
    loss_amp = mse(inp_amp, tar_amp)
    loss_phase = mse(inp_phase, tar_phase)
    loss = loss_amp + loss_phase
    return loss


def beta_nll_loss(x, y, beta=0.5):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
    weighting between data points, where "0" corresponds to
    high weight on low error points and "1" to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]
    mean = torch.stack([pred_amp, pred_phase], axis=1)

    unc_amp = x[:, 1, :]
    unc_phase = x[:, 3, :]
    variance = torch.stack([unc_amp, unc_phase], axis=1)

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]
    target = torch.stack([tar_amp, tar_phase], axis=1)

    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * variance.detach() ** beta

    return loss.mean()


def mse(x, y):
    mse = nn.MSELoss()
    loss = mse(x, y)
    return loss


def jet_seg(x, y):
    # weight components farer outside more
    loss_l1_weighted = 0
    for i in range(x.shape[1]):
        loss_l1_weighted += l1(x[:, i], y[:, i]) * (i + 1)

    return loss_l1_weighted
