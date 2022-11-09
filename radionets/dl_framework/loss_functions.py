import numpy as np
import torch
from torch import nn
from radionets.dl_framework.utils import (
    bbox_iou,
    build_target_yolo,
    decode_yolo_box,
)
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


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


def mse_modified(x, y):
    # Modified, so the MSE is also larger than the L1 loss between 0 and 1
    loss = torch.mean(torch.pow(torch.abs(x - y) + 1, 2) - 1)
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


def jet_seg(x, y):
    # weight components farer outside more
    loss_main_comp = l1(x[:, 0], y[:, 0])
    loss_all_comps_summed = l1(x[:, -1], y[:, -1])
    loss_l1_weighted = 0
    for i in range(
        int((x.shape[1] - 2) / 2)
    ):  # -2: main component and summed component, /2: two sides
        loss_l1_weighted += l1(x[:, i + 1], y[:, i + 1]) * (i / 2 + 1.5)
        loss_l1_weighted += l1(x[:, i + 6], y[:, i + 6]) * (i / 2 + 1.5)

    loss = loss_main_comp + loss_all_comps_summed + loss_l1_weighted
    return loss


def jet_list(x, y):
    """
    Loss function for architecture UNet_jet_advanced

    Parameters
    ----------
    x: tuple
        Output of the network
        (component images, components parameters (confidence, amplitude, x, y, width, height), jet parameters (velocity, angle))
        shapes: (bs, 12, 256, 256), (bs, 11, 6), (bs, 12)
    y: tuple
        Simulated values
        (component images, all parameters (amplitude, x, y, width, height, comp-rotation, jet-rotation, velocity))
        shapes: (bs, 13, 256, 256)

    Returns
    -------
    loss: float
    """
    x_comp, x_list, x_angle = x

    y_comp, y_param = torch.split(y, [y.shape[1] - 1, 1], dim=1)

    w_image = 0.0  # weight image loss
    w_regressor = 1
    w_box = min(1, w_regressor)  # weight image loss
    w_conf = min(0.5, w_regressor)  # weight image losss
    w_amp = min(0.3, w_regressor)  # weight image loss
    w_angle = min(0.2, w_regressor)  # weight image loss
    w_beta = min(0.2, w_regressor)  # weight image loss

    loss = 0

    if w_image:
        loss += jet_seg(x_comp, y_comp) * w_image

    y_param = y_param.squeeze()
    for i_row in range(len(y_param[0])):
        if y_param[0, i_row, 0].isnan():
            break
    for i_col in range(len(y_param[0])):
        if y_param[0, 0, i_col].isnan():
            break
    y_param = y_param[:, 0:i_row, 0:i_col]

    assert y_param.shape[2] == 8, (
        "Number of parameters for the components has changed. Check simulations and/or "
        "        indexing in loss function!"
    )

    y_box_list = y_param[..., 1:5] / 256

    # IoU-loss for the box
    if w_box:
        box_size_scale = 1
        x_box = x_list[..., 2:6].reshape(-1, 4)
        x_box_packed = [
            x_box[:, 0],
            x_box[:, 1],
            x_box[:, 2] * box_size_scale,
            x_box[:, 3] * box_size_scale,
        ]
        y_box = y_box_list.reshape(-1, 4)
        y_box_packed = [
            y_box[:, 0],
            y_box[:, 1],
            y_box[:, 2] * box_size_scale,
            y_box[:, 3] * box_size_scale,
        ]

        ciou = bbox_iou(x_box_packed, y_box_packed, iou_type="ciou")
        ciou = ciou[y_param[..., 0].reshape(-1).bool()]  # no loss, if amplitude is 0
        loss += (1.0 - ciou).mean() * w_box
        # print('loss box:', (1.0 - ciou).mean() * w_box)

    if w_conf:
        confidence = bbox_iou(x_box_packed, y_box_packed)
        loss += l1(x_list[..., 0].reshape(-1), confidence) * w_box
        # print('loss confidence:', l1(x_list[..., 0].reshape(-1), confidence) * w_box)

    if w_amp:
        loss += l1(x_list[..., 1], y_param[..., 0]) * w_amp
        # print('loss amplitude:', l1(x_list[..., 1], y_param[..., 0]) * w_amp)

    y_angle = y_param[:, 0, 6] / (torch.pi / 2)
    if w_angle:
        loss += l1(x_angle[..., -1], y_angle) * w_angle
        # print('loss angle:', l1(x_angle[..., -1], y_angle) * w_angle)

    if w_beta:
        loss += l1(x_angle[..., 0:-1], y_param[..., 7]) * w_beta
        # print('loss velocity:', l1(x_angle[..., 0:-1], y_param[..., 7]) * w_beta)

    # print()
    # print(loss)
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
    w_rot = 0.5

    # how much the image got reduced, must match self.strides_head of architecture
    strides_head = torch.tensor([4, 8, 16])
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

            output_box = output[..., :4].reshape(-1, 4)
            output_box_packed = [
                output_box[:, 0],
                output_box[:, 1],
                output_box[:, 2],
                output_box[:, 3],
            ]

            target_box = target[..., :4].reshape(-1, 4)
            target_box_packed = [
                target_box[:, 0],
                target_box[:, 1],
                target_box[:, 2] * 2,  # target width is std of gauss -> too small
                target_box[:, 3] * 2,  # target heigth is std of gauss -> too small
            ]

            ciou = bbox_iou(output_box_packed, target_box_packed, iou_type="ciou")

            target_obj = target[..., 4].reshape(-1)
            ciou = ciou[target_obj.bool()]  # no loss, if no object
            # ciou = ciou[torch.sigmoid(output[..., 4].reshape(-1)) > 0.5]
            loss_box += (1.0 - ciou).mean() * w_box / len(x)
            # print(f'loss box: {loss_box}')

        if w_obj:
            output_obj = output[..., 4].reshape(-1)
            if not w_box:
                target_obj = target[..., 4].reshape(-1)

            if weighted_bce:
                bcewithlog_loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.sqrt(
                        (target_obj == 0).sum() / (target_obj == 1).sum()
                    )
                )
            else:
                bcewithlog_loss = nn.BCEWithLogitsLoss()

            loss_obj_bce = bcewithlog_loss(output_obj, target_obj) * w_obj / len(x)
            loss_obj += loss_obj_bce
            # print(f'loss obj: {loss_obj}')

        if w_rot:
            output_rot = output[..., 5].reshape(-1)
            target_rot = target[..., 5].reshape(-1) / np.pi

            loss_rot += l1(output_rot[target_rot > 0], target_rot[target_rot > 0])

    # print(f'loss box: {loss_box:.4f}, obj: {loss_obj:.4f}')

    loss = loss_box + loss_obj + loss_rot
    if torch.isnan(loss):
        print(
            f"Loss got nan. Box loss: {loss_box}, Objectness loss: {loss_obj}, \
                Rotation loss: {loss_rot}"
        )
        quit()

    return loss
