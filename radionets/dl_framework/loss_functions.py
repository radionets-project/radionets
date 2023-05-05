import numpy as np
import torch
from pytorch_msssim import MS_SSIM
from torch import nn

from radionets.dl_framework.utils import (
    bbox_iou,
    build_target_yolo,
    decode_yolo_box,
    get_ifft_torch,
)


def l1(x, y):
    l1 = nn.L1Loss()
    loss = l1(x, y)
    return loss


def l1_amp(x, y):
    l1 = nn.L1Loss()
    loss = l1(x, y[:, 0].unsqueeze(1))
    return loss


def l1_phase(x, y):
    l1 = nn.L1Loss()
    loss = l1(x, y[:, 1].unsqueeze(1))
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


def fft_L1(x, y):
    ifft_pred = get_ifft_torch(x.clamp(0, 2), amp_phase=True, scale=True)
    ifft_truth = get_ifft_torch(y, amp_phase=True, scale=True)

    ifft_pred[torch.isnan(ifft_pred)] = 1
    ifft_pred[torch.isinf(ifft_pred)] = 1

    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss_fft = l1(ifft_pred[ifft_truth > 0], ifft_truth[ifft_truth > 0])
    loss = loss_amp + 10 * loss_phase + loss_fft
    return loss


def splitted_L1(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
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


def mse_amp(x, y):
    tar = y[:, 0, :].unsqueeze(1)
    mse = nn.MSELoss()
    loss = mse(x, tar)
    return loss


def mse_phase(x, y):
    tar = y[:, 1, :].unsqueeze(1)
    mse = nn.MSELoss()
    loss = mse(x, tar)
    return loss


def loss_new_msssim(x, y):
    msssim_loss = MS_SSIM(data_range=10, channel=2)
    loss = 1 - msssim_loss(x, y)

    return loss


def yolo(x, y):
    """Loss for YOLO training

    Parameters
    ----------
    x: list
        output layers, each of shape (bs, n_anchors, ny, nx, 6)
    y: ndarray
        truth from simulation
        (amplitude, x, y, width, height, comp-rotation, jet-rotation, velocity)

    Returns
    -------
    loss: float
        loss for the batch
    """
    w_box = 1
    w_obj = 5
    w_rot = 3

    # how much the image got reduced, must match self.strides_head of architecture
    strides_head = torch.tensor([2, 4, 8])
    # strides_head = torch.tensor([4, 8, 16])
    weighted_bce = True

    loss_box = 0
    loss_obj = 0
    loss_rot = 0

    for i_layer, output in enumerate(x):
        target = build_target_yolo(y, output.shape, strides_head[i_layer]).to(
            output.device
        )

        if w_box:
            output = decode_yolo_box(output, strides_head[i_layer])
            target_obj = target[..., 4].reshape(-1)

            output_box = output[..., :4].reshape(-1, 4)
            output_box_rot = output[..., 5].reshape(-1)
            sin_adj = torch.abs(torch.cos(output_box_rot * torch.pi))
            cos_adj = torch.abs(torch.cos(output_box_rot * torch.pi))
            output_box_packed = [
                output_box[:, 0],
                output_box[:, 1],
                output_box[:, 2] * cos_adj + output_box[:, 3] * sin_adj,
                output_box[:, 3] * cos_adj + output_box[:, 2] * sin_adj,
            ]

            target_box = target[..., :4].reshape(-1, 4)
            target_box_rot = target[..., 5].reshape(-1)
            sin_adj = torch.abs(torch.cos(target_box_rot))
            cos_adj = torch.abs(torch.cos(target_box_rot))
            target_box_packed = [
                target_box[:, 0],
                target_box[:, 1],
                (target_box[:, 2] * cos_adj + target_box[:, 3] * sin_adj)
                * 2,  # target width is std of gauss -> too small, increased by *2
                (target_box[:, 3] * cos_adj + target_box[:, 2] * sin_adj)
                * 2,  # target heigth is std of gauss -> too small, increased by *2
            ]

            ciou = bbox_iou(output_box_packed, target_box_packed, iou_type="ciou")

            target_obj = target[..., 4].reshape(-1)
            ciou = ciou[target_obj.bool()]  # no loss, if no object
            loss_box += (1.0 - ciou).mean() * w_box / len(x)
            # print(f'loss box: {loss_box}')

        if w_obj:
            output_obj = output[..., 4].reshape(-1)
            if not w_box:
                target_obj = target[..., 4].reshape(-1)

            if weighted_bce:
                bcewithlog_loss = nn.BCEWithLogitsLoss(
                    pos_weight=(target_obj == 0).sum() / (target_obj == 1).sum()
                )
            else:
                bcewithlog_loss = nn.BCEWithLogitsLoss()

            loss_obj_bce = bcewithlog_loss(output_obj, target_obj) * w_obj / len(x)
            loss_obj += loss_obj_bce
            # print(f'loss obj: {loss_obj}')

        if w_rot:
            output_rot = output[..., 5].reshape(-1)
            target_rot = target[..., 5].reshape(-1) / torch.pi

            loss_rot += (
                l1(output_rot[target_rot > 0], target_rot[target_rot > 0])
                * w_rot
                / len(x)
            )

    loss = loss_box + loss_obj + loss_rot

    if torch.isnan(loss):
        print(
            f"Loss got nan. Box loss: {loss_box}, Objectness loss: {loss_obj}, Rotation loss: {loss_rot}"
        )
        quit()
    return loss


def counterjet(x, y):
    n_components = y.shape[1]
    amps = y[:, -int((n_components - 1) / 2) :, 0]
    amps_summed = torch.sum(amps, axis=1)

    target = (amps_summed > 0).float()
    bce = nn.BCEWithLogitsLoss()
    loss = bce(x, target)
    return loss
