import torch
from torch import nn
from dl_framework.hook_fastai import hook_outputs
from torchvision.models import vgg16_bn
from dl_framework.utils import children
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dl_framework.model import fft, flatten, euler
from math import pi


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, base_loss, layer_ids, layer_wgts):
        """"
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
        """"
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
        target = target.view(-1, 2, 64, 64)
        input = input.view(-1, 2, 64, 64)

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
    method to initialise  the pretrained net, which will be used for the feature loss.
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


def torch_abs(pred):
    return torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2)


def bmul(vec, mat, axis=0):
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


def PCA(image):
    """
    Compute the major components of an image. The Image is treated as a
    distribution.

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be used as distribution

    Returns
    -------
    cog_x: Skalar
            X-position of the distributions center of gravity
    cog_y: Skalar
            Y-position of the distributions center of gravity
    psi: Skalar
            Angle between first mjor component and x-axis

    """
    torch.set_printoptions(precision=16)

    pix_x, pix_y, image = im_to_array_value(image)

    cog_x = (torch.sum(pix_x * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)
    cog_y = (torch.sum(pix_y * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)

    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    inp = torch.cat([delta_x.unsqueeze(1), delta_y.unsqueeze(1)], dim=1)

    cov_w = bmul(
        (cog_x - 1 * torch.sum(image * image, axis=1).unsqueeze(-1) / cog_x).squeeze(1),
        (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
    )

    eig_vals_torch, eig_vecs_torch = torch.symeig(cov_w, eigenvectors=True)
    sqrt = torch.sqrt(eig_vals_torch)
    width_torch, length_torch = sqrt[:, 0], sqrt[:, 1]

    psi_torch = torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1])

    return cog_x, cog_y, psi_torch


def im_to_array_value(image):
    """
    Transforms the image to an array of pixel coordinates and the containt
    intensity

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be transformed

    Returns
    -------
    x_coords: Numpy 1Darray (N*M, 1)
            Contains the x-pixel-position of every pixel in the image
    y_coords: Numpy 1Darray (N*M, 1)
            Contains the y-pixel-position of every pixel in the image
    value: Numpy 1Darray (N*M, 1)
            Contains the image-value corresponding to every x-y-pair

    """
    num = image.shape[0]
    pix = image.shape[-1]

    a = torch.arange(0, pix, 1).cuda()
    grid_x, grid_y = torch.meshgrid(a, a)
    x_coords = torch.cat(num * [grid_x.flatten().unsqueeze(0)])
    y_coords = torch.cat(num * [grid_y.flatten().unsqueeze(0)])
    value = image.reshape(-1, pix ** 2)
    return x_coords, y_coords, value


def cross_section(img):
    img = torch.abs(img).clone().detach()

    # delete outer parts
    img[:, 0:10] = 0
    img[:, 53:63] = 0
    img[:, :, 0:10] = 0
    img[:, :, 53:63] = 0

    for i in img:
        # only use brightest pixel
        i[i < i.max() * 0.4] = 0

    # pca
    y, x, alpha = PCA(img)

    # Get line of major component
    m = torch.tan(pi / 2 - alpha)
    n = y - m.unsqueeze(-1) * x
    return m, n, alpha


def calc_spec(img, m, n, alpha):
    s = [
        np.abs(
            np.fft.fft(
                ndimage.rotate(np.abs(i), 180 * (2 * pi - a) / pi, reshape=True).sum(
                    axis=0
                )
            )
        )
        for i, a in zip(img, alpha)
    ]
    return s


def regularization(pred_phase, img_true):
    real_amp_re = img_true[:, 0].reshape(-1, 63 ** 2)
    real_phase_re = img_true[:, 1].reshape(-1, 63 ** 2)
    pred_phase_re = pred_phase[:, 0].reshape(-1, 63 ** 2)

    x = torch.cat([real_amp_re, pred_phase_re], dim=1)
    comp_pred = flatten(euler(x))
    fft_pred = fft(comp_pred)
    img_pred = torch_abs(fft_pred)

    y = torch.cat([real_amp_re, real_phase_re], dim=1)
    comp_true = flatten(euler(y))
    fft_true = fft(comp_true)
    img_true = torch_abs(fft_true)

    m_pred, n_pred, alpha_pred = cross_section(img_pred)
    s_pred = calc_spec(
        img_pred.detach().cpu(),
        m_pred.detach().cpu(),
        n_pred.detach().cpu(),
        alpha_pred.detach().cpu(),
    )

    m_true, n_true, alpha_true = cross_section(img_true)
    s_true = calc_spec(
        img_true.detach().cpu(),
        m_true.detach().cpu(),
        n_true.detach().cpu(),
        alpha_true.detach().cpu(),
    )

    loss = torch.tensor(
        [
            (
                (s_p[: len(s_p) // 2] - s_t[: len(s_p) // 2]) ** 2
            ).sum()  # / np.abs(s_t).max()
            for s_p, s_t in zip(s_pred, s_true)
        ]
    ).mean()
    print(loss)
    return loss


def my_loss(x, y):
    img_true = y.clone()
    y = y[:, 1].unsqueeze(1)
    assert y.shape == x.shape
    loss = (((x - y)).pow(2)).mean()
    print(loss)
    return loss + regularization(x, img_true)


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

    loss = ((x - tar).pow(2)).mean()

    return loss


def loss_phase(x, y):
    tar = y[:, 1, :].unsqueeze(1)
    assert tar.shape == x.shape

    loss = ((x - tar).pow(2)).mean()

    return loss
