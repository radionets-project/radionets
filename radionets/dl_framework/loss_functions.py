# -*- coding: utf-8 -*-
import torch
from torch import nn
from radionets.dl_framework.hook_fastai import hook_outputs
from torchvision.models import vgg16_bn
from radionets.dl_framework.utils import children
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, MS_SSIM
from radionets.dl_framework.regularization import (
    inv_fft,
    calc_jet_angle,
    rot,
    calc_spec,
)


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, base_loss, layer_ids, layer_wgts):
        """ "
        m_feat: enthält das vortrainierte Netz
        loss_features: dort werden alle features gespeichert, deren Loss
        man berechnen will
        """
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = base_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        # self.metric_names = (
        #     ["pixel", ]
        #     + [f"feat_{i}" for i in range(len(layer_ids))]
        #     + [f"gram_{i}" for i in range(len(layer_ids))]
        # )

    def make_features(self, x, clone=False):
        """ "
        Hier wird das Objekt x durch das vortrainierte Netz geschickt und somit die
        Aktivierungsfunktionen berechnet. Dies geschieht sowohl einmal für
        die Wahrheit "target" und einmal für die Prediction "input"
        aus dem Generator. Dann werden die berechneten Aktivierungsfunktionen als Liste
        gespeichert und zurückgegeben.
        """
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        # resizing the input, before it gets into the net
        # shape changes from 4096 to 64x64
        target = target.view(-1, 2, 63, 63)
        input = input.view(-1, 2, 63, 63)

        # create dummy tensor of zeros to add another dimension
        padding_target = torch.zeros(
            target.size(0), 1, target.size(2), target.size(3)
        ).cuda()
        padding_input = torch.zeros(
            input.size(0), 1, input.size(2), input.size(3)
        ).cuda()

        # 'add' the extra channel
        target = torch.cat((target, padding_target), 1)
        input = torch.cat((input, padding_input), 1)

        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        # Hier wird jetzt der L1-Loss zwischen Input und Target berechnet
        self.feat_losses = [self.base_loss(input, target)]

        # hier wird das gleiche nochmal für alle Features gemacht
        self.feat_losses += [
            self.base_loss(f_in, f_out)
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        # erstmal den Teil mit der gram_matrix auskommentiert, bis er
        # verstanden ist
        self.feat_losses += [
            self.base_loss(gram_matrix(f_in), gram_matrix(f_out))
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        # Wird als Liste gespeichert, um es in metrics abspeichern
        # zu können und printen zu können

        # erstmal unnötig
        # self.metrics = dict(zip(self.metric_names, self.feat_losses))

        # zum Schluss wird hier aufsummiert
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


def init_feature_loss(
    pre_net=vgg16_bn,
    pixel_loss=F.l1_loss,
    begin_block=2,
    end_block=5,
    layer_weights=[5, 15, 2],
):
    """
    method to initialise the pretrained net, which will be used for the feature loss.
    """
    vgg_m = pre_net(True).features.cuda().eval()
    for param in vgg_m.parameters():
        param.requires_grad = False
    blocks = [
        i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)
    ]
    feat_loss = FeatureLoss(
        vgg_m, pixel_loss, blocks[begin_block:end_block], layer_weights
    )
    return feat_loss


def mse(x, y):
    mse = nn.MSELoss()
    loss = mse(x, y)
    return loss


def l1(x, y):
    l1 = nn.L1Loss()
    loss = l1(x, y)
    return loss


def splitted_mse(x, y):
    inp_real = x[:, 0, :]
    inp_imag = x[:, 1, :]

    tar_real = y[:, 0, :]
    tar_imag = y[:, 1, :]

    loss_real = (
        torch.sum(1 / inp_real.shape[1] * torch.sum((inp_real - tar_real) ** 2, 1))
        * 1
        / inp_real.shape[0]
    )
    loss_imag = (
        torch.sum(1 / inp_imag.shape[1] * torch.sum((inp_imag - tar_imag) ** 2, 1))
        * 1
        / inp_real.shape[0]
    )

    return loss_real + loss_imag


def regularization(pred_phase, img_true):
    real_amp_re = img_true[:, 0].reshape(-1, 63 ** 2)
    real_phase_re = img_true[:, 1].reshape(-1, 63 ** 2)
    pred_phase_re = pred_phase[:, 0].reshape(-1, 63 ** 2)

    img_pred = inv_fft(real_amp_re, pred_phase_re)
    img_true = inv_fft(real_amp_re, real_phase_re)

    m_true, n_true, alpha_true = calc_jet_angle(img_true)

    img_rot_pred = rot(img_pred, alpha_true)
    s_pred = calc_spec(img_rot_pred)
    img_rot_true = rot(img_true, alpha_true)
    s_true = calc_spec(img_rot_true)

    img_rot_pred_cj = rot(img_pred, alpha_true - 90)
    s_pred_cj = calc_spec(img_rot_pred_cj)
    img_rot_true_cj = rot(img_true, alpha_true - 90)
    s_true_cj = calc_spec(img_rot_true_cj)

    loss_1 = (((s_pred - s_true) ** 2).sum(axis=0)).mean()
    # print(loss_1)
    # loss_2 = (((s_pred_cj - s_true_cj) ** 2).sum(axis=0)).mean()
    # print(loss_2)
    loss = loss_1  # + loss_2
    print(loss)
    return loss


def my_loss(x, y):
    img_true = y.clone()
    y = y[:, 1].unsqueeze(1)
    assert y.shape == x.shape
    loss = (((x - y)).pow(2)).mean()
    print(loss)
    # final_loss = loss * 10 + regularization(x, img_true) / 100
    # print(final_loss)
    print("")
    return loss


def likelihood(x, y):
    y = y[:, 0]
    inp = x[:, 2]
    unc = x[:, 1][inp == 0]
    y_pred = x[:, 0][inp == 0]
    y = y[inp == 0]
    loss = (2 * torch.log(unc) + ((y - y_pred).pow(2) / unc.pow(2))).mean()
    assert unc.shape == y_pred.shape == y.shape
    return loss


def likelihood_phase(x, y):
    y = y[:, 1]
    inp = x[:, 2]
    unc = x[:, 1][inp == 0]
    assert len(unc[unc <= 0]) == 0
    y_pred = x[:, 0][inp == 0]
    y = y[inp == 0]
    loss = (2 * torch.log(unc) + ((y - y_pred).pow(2) / unc.pow(2))).mean()
    assert unc.shape == y_pred.shape == y.shape
    return loss


def loss_amp(x, y):
    tar = y[:, 0, :].unsqueeze(1)
    assert tar.shape == x.shape

    mse = nn.MSELoss()
    loss = mse(x, tar)

    return loss


def loss_phase(x, y):
    tar = y[:, 1, :].unsqueeze(1)
    assert tar.shape == x.shape

    mse = nn.MSELoss()
    loss = mse(x, tar)

    return loss


def loss_l1_amp(x, y):
    tar = y[:, 0, :].unsqueeze(1)
    assert tar.shape == x.shape

    l1 = nn.L1Loss()
    loss = l1(x, tar)

    return loss


def loss_l1_phase(x, y):
    tar = y[:, 1, :].unsqueeze(1)
    assert tar.shape == x.shape

    l1 = nn.L1Loss()
    loss = l1(x, tar)

    return loss


def loss_new_msssim(x, y):
    msssim_loss = MS_SSIM(data_range=10, channel=2)
    loss = 1 - msssim_loss(x, y)

    return loss


def loss_msssim(x, y):
    """Loss function with 1 - the value of the MS-SSIM

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        output of net

    Returns
    -------
    float
        value of 1 - MS-SSIM
    """
    inp_real = x[:, 0, :].unsqueeze(1)
    inp_imag = x[:, 1, :].unsqueeze(1)

    tar_real = y[:, 0, :].unsqueeze(1)
    tar_imag = y[:, 1, :].unsqueeze(1)

    loss = (
        1.0
        - pytorch_msssim.msssim(inp_real, tar_real, normalize="relu")
        + 1.0
        - pytorch_msssim.msssim(inp_imag, tar_imag, normalize="relu")
    )

    return loss


def loss_msssim_diff(x, y):
    """Loss function with negative value of MS-SSIM

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        target image

    Returns
    -------
    float
        value of negative MS-SSIM
    """
    inp_real = x[:, 0, :].unsqueeze(1)
    inp_imag = x[:, 1, :].unsqueeze(1)

    tar_real = y[:, 0, :].unsqueeze(1)
    tar_imag = y[:, 1, :].unsqueeze(1)

    loss = -(
        pytorch_msssim.msssim(inp_real, tar_real, normalize="relu")
        + pytorch_msssim.msssim(inp_imag, tar_imag, normalize="relu")
    )

    return loss


def loss_msssim_amp(x, y):
    """Loss function with 1 - the value of the MS-SSIM for amplitude

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        output of net

    Returns
    -------
    float
        value of 1 - MS-SSIM
    """
    inp_real = x
    tar_real = y[:, 0, :].unsqueeze(1)

    loss = 1.0 - pytorch_msssim.msssim(inp_real, tar_real, normalize="relu")

    return loss


def loss_mse_msssim_phase(x, y):
    """Combine MSE and MS-SSIM loss for phase

    Parameters
    ----------
    x : tensor
        ouptut of net
    y : tensor
        target image

    Returns
    -------
    float
        value of addition of MSE and MS-SSIM
    """

    tar_phase = y[:, 1, :].unsqueeze(1)
    inp_phase = x

    loss_mse_phase = nn.MSELoss()
    loss_mse_phase = loss_mse_phase(inp_phase, tar_phase)

    loss_phase = (
        loss_mse_phase
        + 1.0
        - pytorch_msssim.msssim(inp_phase, tar_phase, normalize="relu")
    )

    return loss_phase


def loss_mse_msssim_amp(x, y):
    """Combine MSE and MS-SSIM loss for amp

    Parameters
    ----------
    x : tensor
        ouptut of net
    y : tensor
        target image

    Returns
    -------
    float
        value of addition of MSE and MS-SSIM
    """

    tar_amp = y[:, 0, :].unsqueeze(1)
    inp_amp = x

    loss_mse_amp = nn.MSELoss()
    loss_mse_amp = loss_mse_amp(inp_amp, tar_amp)

    loss_amp = (
        loss_mse_amp + 1.0 - pytorch_msssim.msssim(inp_amp, tar_amp, normalize="relu")
    )

    return loss_amp


def loss_mse_msssim(x, y):
    """Combine MSE and MS-SSIM loss

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        target image

    Returns
    -------
    float
        value of addition of MSE and MS-SSIM
    """
    # Split in amplitude and phase
    inp_amp = x[:, 0, :].unsqueeze(1)
    inp_phase = x[:, 1, :].unsqueeze(1)

    tar_amp = y[:, 0, :].unsqueeze(1)
    tar_phase = y[:, 1, :].unsqueeze(1)

    # calculate mse loss
    loss_mse_amp = nn.MSELoss()
    loss_mse_amp = loss_mse_amp(inp_amp, tar_amp)

    loss_mse_phase = nn.MSELoss()
    loss_mse_phase = loss_mse_phase(inp_phase, tar_phase)

    # add mse loss to MS-SSIM
    loss_amp = (
        loss_mse_amp + 1.0 - pytorch_msssim.msssim(inp_amp, tar_amp, normalize="relu")
    )
    loss_phase = (
        loss_mse_phase
        + 1.0
        - pytorch_msssim.msssim(inp_phase, tar_phase, normalize="relu")
    )

    return loss_amp + loss_phase


def spe(x, y):
    y = y.squeeze()
    x = x.squeeze()
    # y = y[:, 0:2]
    loss = []
    value = 0
    for i in range(len(x)):
        value += torch.abs(x[i] - y[i])
        loss.append(value)
        value = 0
    loss = sum(loss) / len(x)
    return loss


def spe_square(x, y):
    y = y.squeeze()
    loss = []
    value = 0
    for i in range(len(x)):
        for k in range(1, len(x[0])):
            if k == 1:
                value += (x[i][k - 1] - y[i][k - 1] + x[i][k] - y[i][k]) ** 2
            else:
                value += torch.abs(x[i][k] - y[i][k])
        loss.append(value / len(x[0]))
        value = 0
    k = sum(loss)
    loss = k / len(x)
    return loss


def spe_(x, y):
    loss = []
    value = 0
    for i in range(len(x)):
        for k in range(len(x[0])):
            value += torch.abs(x[i][k] - y[i][k])
        loss.append(value / len(x[0]))
        value = 0
    k = sum(loss)
    loss = k / len(x)
    return loss


def splitted_L1(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()

    loss_amp = l1(inp_amp, tar_amp)

    loss_phase = l1(inp_phase, tar_phase)

    return loss_amp * 10 + loss_phase


def classifier_loss(x, y):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(x,y.squeeze(1).long())
    return loss
