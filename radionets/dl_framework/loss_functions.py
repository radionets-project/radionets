import numpy as np
import torch
from torch import nn


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


def splitted_L1_masked(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(create_circular_mask(256, 256, radius=50, bs=y.shape[0]))

    inp_amp[~mask] *= 0.3
    inp_phase[~mask] *= 0.3
    tar_amp[~mask] *= 0.3
    tar_phase[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase
    return loss


def splitted_L1(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    mask = torch.zeros(tar_amp.shape)
    mask[:, 206:306, 206:306] = 1

    mask2 = torch.zeros(tar_amp.shape)
    mask2[:, 236:376, 236:376] = 1

    mask3 = mask.bool() & mask2.bool()

    l1 = nn.L1Loss(reduction="sum")
    loss_amp = l1(inp_amp[~mask3.bool()], tar_amp[~mask3.bool()])
    loss_amp_mid = l1(inp_amp[mask.bool()], tar_amp[mask.bool()])
    loss_amp_inner = l1(inp_amp[mask2.bool()], tar_amp[mask2.bool()])
    loss_phase = l1(inp_phase[~mask3.bool()], tar_phase[~mask3.bool()])
    loss_phase_mid = l1(inp_phase[mask.bool()], tar_phase[mask.bool()])
    loss_phase_inner = l1(inp_phase[mask2.bool()], tar_phase[mask2.bool()])
    loss = (
        loss_amp
        + loss_phase
        + loss_amp_mid * 10
        + loss_phase_mid * 10
        + loss_amp_inner * 100
        + loss_phase_inner * 100
    )
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
