import numpy as np
import torch
from torch import nn

from radionets.dl_framework.architectures.neural_int import conjugate_vis, get_halfspace


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


# from radionets.evaluation.utils import apply_symmetry


def splitted_L1(x, y):
    # x, means, stds = x
    # x[:, 0] = x[:, 0] * stds[:, 0] + means[:, 0]
    # x[:, 1] = x[:, 1] * stds[:, 1] + means[:, 1]
    # x[:, 2] = x[:, 2] * stds[:, 2] + means[:, 2]
    # x[:, 3] = x[:, 3] * stds[:, 3] + means[:, 3]
    # x[:, 4] = x[:, 4] * stds[:, 4] + means[:, 4]
    # x[:, 5] = x[:, 5] * stds[:, 5] + means[:, 5]
    # x[:, 6] = x[:, 6] * stds[:, 6] + means[:, 6]
    # x[:, 7] = x[:, 7] * stds[:, 7] + means[:, 7]
    # x[:, 8] = x[:, 8] * stds[:, 8] + means[:, 8]
    # x[:, 9] = x[:, 9] * stds[:, 9] + means[:, 9]
    # x[:, 10] = x[:, 10] * stds[:, 10] + means[:, 10]
    # x[:, 11] = x[:, 11] * stds[:, 11] + means[:, 11]
    # y[:, 0] = y[:, 0] * stds[:, 0] + means[:, 0]
    # y[:, 1] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 2] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 3] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 4] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 5] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 6] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 7] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 8] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 9] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 10] = y[:, 1] * stds[:, 1] + means[:, 1]
    # y[:, 11] = y[:, 1] * stds[:, 1] + means[:, 1]
    inp_amp = x[:, :6]
    inp_phase = x[:, 6:]

    tar_amp = y[:, :6]
    tar_phase = y[:, 6:]

    # a_pred = x[:, 0] #* torch.cos(x[:, 1])
    # b_pred = x[:, 1] #* torch.sin(x[:, 1])
    # compl_pred = a_pred + b_pred * 1j
    # ifft_pred = torch.abs(
    # torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(compl_pred)))
    # )

    # a_true = y[:, 0] #* torch.cos(y[:, 1])
    # b_true = y[:, 1] #* torch.sin(y[:, 1])
    # compl_true = a_true + b_true * 1j
    # ifft_true = torch.abs(
    # torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(compl_true)))
    # )

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    # ifft_pred[ifft_pred <= 1e-8] = 1e-8
    # ifft_true[ifft_true <= 1e-8] = 1e-8
    # loss_fft = l1(torch.log10(ifft_pred), torch.log10(ifft_true))
    # print("fft", loss_fft)
    # print("real", loss_amp)
    # print("imag", loss_phase)
    loss = loss_amp + loss_phase  # + loss_fft
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


def vis_data_loss(x, y):
    _, uv_dense, _, visibilities, img = y
    halfspace = get_halfspace(uv_dense)
    vis_conj = conjugate_vis(visibilities, halfspace)
    vis_real = vis_conj.real.float()
    vis_imag = vis_conj.imag.float()
    x_reshaped = x.reshape(-1, 64, 64, 2)
    ifft_img = torch.abs(
        torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.fftshift(x_reshaped[..., 0] + 1j * x_reshaped[..., 1])
            )
        )
    )

    l1 = nn.L1Loss()
    real_loss = l1(vis_real, x[:, :, 0])
    imaginary_loss = l1(vis_imag, x[:, :, 1])
    img_loss = l1(ifft_img, img)
    return real_loss + imaginary_loss + img_loss
