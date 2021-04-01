import torch
from torch import nn
from torchvision.models import vgg16_bn
from radionets.dl_framework.utils import children
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from scipy.optimize import linear_sum_assignment


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
    return loss_amp * 10 + loss_phase


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
    print("amp: ", loss_amp)
    print("phase: ", loss_phase)
    # print(loss)
    # assert unc.shape == y_pred.shape == y.shape
    return loss


def loss_new_msssim(x, y):
    msssim_loss = MS_SSIM(data_range=10, channel=2)
    loss = 1 - msssim_loss(x, y)

    return loss


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


def list_loss(x, y):
    y = y.squeeze(1)
    x_pred = x[:]
    x_true = y[:, 0:2] / 63
    m = nn.MSELoss()
    loss = m(x_pred, x_true)
    # print(x_pred[0], x_true[0])
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
