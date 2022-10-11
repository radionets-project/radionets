import torch
from torch import nn
from pytorch_msssim import MS_SSIM
from scipy.optimize import linear_sum_assignment


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


def l1_phase_unc(x, y):
    phase_pred = x[:, 0]
    y_phase = y[:, 1]

    l1 = nn.SmoothL1Loss()
    loss = l1(phase_pred, y_phase)
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


def splitted_L1_unc(x, y):
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_amp = l1(pred_amp, tar_amp)
    loss_phase = l1(pred_phase, tar_phase)
    return loss_amp + loss_phase


def L1_icecube(x, y):
    # get components for computation
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]

    unc_amp = x[:, 1, :]
    unc_phase = x[:, 3, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    # compute L1 Loss on amplitude and phase
    l1 = nn.L1Loss()
    loss_amp = l1(pred_amp, tar_amp)
    loss_phase = l1(pred_phase, tar_phase)

    # compute additional uncertainty loss from IceCube
    unc = (unc_amp - (tar_amp - pred_amp).detach()) ** 2 + (
        unc_phase - (tar_phase - pred_phase).detach()
    ) ** 2

    # add up all loss parts
    loss = loss_amp + loss_phase + unc.mean()

    return loss


def splitted_SmoothL1_unc(x, y):
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.SmoothL1Loss()
    loss_amp = l1(pred_amp, tar_amp)
    loss_phase = l1(pred_phase, tar_phase)
    return loss_amp + loss_phase * 10


def MSE_icecube(x, y):
    # get components for computation
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]

    unc_amp = x[:, 1, :]
    unc_phase = x[:, 3, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    # compute MSE Loss on amplitude and phase
    MSE = nn.MSELoss()
    loss_amp = MSE(pred_amp, tar_amp)
    loss_phase = MSE(pred_phase, tar_phase)

    # compute additional uncertainty loss from IceCube
    unc = (unc_amp - (tar_amp - pred_amp).detach()) ** 2 + (
        unc_phase - (tar_phase - pred_phase).detach()
    ) ** 2

    # add up all loss parts
    loss = loss_amp + loss_phase + unc.mean()

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


def amp_likelihood(x, y):
    amp_pred = x[:, 0]
    amp_unc = x[:, 1]
    y_amp = y[:, 0]
    loss_amp = (
        2 * torch.log(amp_unc) + ((y_amp - amp_pred).pow(2) / amp_unc.pow(2))
    ).mean()
    return loss_amp


def phase_likelihood(x, y):
    phase_pred = x[:, 0]
    phase_unc = x[:, 1]
    y_phase = y[:, 1]
    loss_phase = (
        2 * torch.log(phase_unc) + ((y_phase - phase_pred).pow(2) / phase_unc.pow(2))
    ).mean()
    return loss_phase


def mse_phase(x, y):
    tar = y[:, 1, :].unsqueeze(1)
    mse = nn.MSELoss()
    loss = mse(x, tar)
    return loss


def comb_likelihood(x, y):
    amp_pred = x[:, 0]
    amp_unc = x[:, 1]
    phase_pred = x[:, 2]
    phase_unc = x[:, 3]
    y_amp = y[:, 0]
    y_phase = y[:, 1]

    loss_amp = (
        2 * torch.log(amp_unc) + ((y_amp - amp_pred).pow(2) / amp_unc.pow(2))
    ).mean()
    loss_phase = (
        2 * torch.log(phase_unc) + ((y_phase - phase_pred).pow(2) / phase_unc.pow(2))
    ).mean()

    loss = loss_amp + loss_phase
    return loss


def f(x, mu):
    return x - mu


def g(x, mu, sig):
    return (sig ** 2 - (x - mu) ** 2) ** 2


def new_like(x, y):
    amp_pred = x[:, 0]
    amp_unc = x[:, 1]
    phase_pred = x[:, 2]
    phase_unc = x[:, 3]
    y_amp = y[:, 0]
    y_phase = y[:, 1]

    loss_amp = (f(y_amp, amp_pred) + g(y_amp, amp_pred, amp_unc)).mean()
    loss_phase = (f(y_phase, phase_pred) + g(y_phase, phase_pred, phase_unc)).mean()

    loss = loss_amp + loss_phase
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


def spe(x, y):
    y = y.squeeze()
    y = y / 62

    for i in range(len(x)):
        x[i] = sort_param(x[i], y[i])

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


# sort after fitting x pos (dependent on y or not)
def sort_param(a, b):
    x_pred = [a[2 * i] for i in range(len(a.split(2)))]
    y_pred = [a[2 * i + 1] for i in range(len(a.split(2)))]
    x_truth = [b[2 * i] for i in range(len(b.split(2)))]
    y_truth = [b[2 * i + 1] for i in range(len(b.split(2)))]
    h = []
    matcher = build_matcher()
    x = matcher(
        torch.tensor((x_pred)).unsqueeze(1), torch.tensor((x_truth)).unsqueeze(1)
    )[0][0]
    y = matcher(
        torch.tensor((y_pred)).unsqueeze(1), torch.tensor((y_truth)).unsqueeze(1)
    )[0][0]
    for k in range(len(x)):
        h.append(x_pred[x[k]])
        # h.append(y_pred[x[k]])  # if just x is considered (x and y are dependent)
        h.append(y_pred[y[k]])  # if x and y are independent
    return torch.tensor((h))


# Sort after distance between vektor
def sort_vektor(a, b, param):
    xy = a.split(param)
    xy_pred = [vektor_abs(xy[i]) for i in range(len(xy))]
    xy = b.split(param)
    xy_truth = [vektor_abs(xy[i]) for i in range(len(xy))]
    h = []
    matcher = build_matcher()
    indice = matcher(
        torch.tensor((xy_pred)).unsqueeze(1), torch.tensor((xy_truth)).unsqueeze(1)
    )[0][0]
    for k in range(len(indice)):
        h.append(b[indice[k] * 2])
        h.append(b[indice[k] * 2 + 1])
    return torch.tensor((h))


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
        loss.append(value)
        value = 0
    loss = sum(loss) / len(x)
    return loss


def list_loss(x, y):
    y = y.squeeze(1)
    x_pred = x[:]
    x_true = y[:, 0:2] / 63
    m = nn.MSELoss()
    loss = m(x_pred, x_true)
    # print(x_pred[0], x_true[0])
    return loss


def phase_likelihood_l1(x, y):
    phase_pred = x[:, 0]
    phase_unc = x[:, 1]

    y_phase = y[:, 1]

    loss_phase = (
        2 * torch.log(phase_unc) + ((y_phase - phase_pred).pow(2) / phase_unc.pow(2))
    ).mean()

    l1 = nn.L1Loss()
    loss_l1 = l1(phase_pred, y_phase)

    loss = loss_phase + loss_l1
    return loss


def vektor_abs(a):
    return (a[0] ** 2 + a[1] ** 2) ** (1 / 2)


class HungarianMatcher(nn.Module):
    """
    Solve assignment Problem.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):

        assert outputs.shape[-1] is targets.shape[-1]

        C = torch.cdist(targets.to(torch.float64), outputs.to(torch.float64), p=1)
        C = C.cpu()

        if len(outputs.shape) == 3:
            bs = outputs.shape[0]
        else:
            bs = 1
            C = C.unsqueeze(0)

        indices = [linear_sum_assignment(C[b]) for b in range(bs)]
        return [(torch.as_tensor(j), torch.as_tensor(i)) for i, j in indices]


def build_matcher():
    return HungarianMatcher()


def pos_loss(x, y):
    """
    Permutation Loss for Source-positions list. With hungarian method
    to solve assignment problem.
    """
    out = x.reshape(-1, 3, 2)
    tar = y[:, 0, :, :2] / 63

    matcher = build_matcher()
    matches = matcher(out[:, :, 0].unsqueeze(-1), tar[:, :, 0].unsqueeze(-1))

    out_ord, _ = zip(*matches)

    ordered = [sort(out[v], out_ord[v]) for v in range(len(out))]
    out = torch.stack(ordered)

    loss = nn.MSELoss()
    loss = loss(out, tar)

    return loss


def sort(x, permutation):
    return x[permutation, :]


def jet_seg(x, y):
    # weight components farer outside more
    loss_l1_weighted = 0
    for i in range(x.shape[1]):
        loss_l1_weighted += l1(x[:, i], y[:, i]) * (i + 1)

    return loss_l1_weighted
