import torch
from torch import nn
from torchvision.models import vgg16_bn
from radionets.dl_framework.utils import children
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from scipy.optimize import linear_sum_assignment
import math


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
    return loss_amp + loss_phase * 10


def splitted_SmoothL1_unc(x, y):
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.SmoothL1Loss()
    loss_amp = l1(pred_amp, tar_amp)
    loss_phase = l1(pred_phase, tar_phase)
    return loss_amp + loss_phase * 10


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
    # print("amp: ", loss_amp)
    # print("phase: ", loss_phase)
    # print(loss)
    # assert unc.shape == y_pred.shape == y.shape
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


def yolo(x, y):
    """
    Loss for YOLO training
    """
    print(x.shape)
    x = x.view(y.shape)
    print(x.shape)
    loss = 0
    for img_x, img_y in zip(x, y):
        for comp_x, comp_y in zip(img_x, img_y):
            box_x = comp_x[1:5]
            box_y = comp_y[1:5]
            loss += bbox_iou(box_x, box_y, CIoU=True)
    
    return loss


# def CIoU(pred, true, eps=1e-7):
#     """
#     Complete Intersection over Union

#     pred: Predictions (bs, components * params)
#     true: Truth (bs, components, params)
    
#     params: (x, y, width, height)
#     """

#     # Get the coordinates of bounding boxes
#     pred_x = pred[..., 0]
#     pred_y = pred[..., 1]
#     pred_h = pred[..., 2]
#     pred_w = pred[..., 3]
#     pred_x1 = pred_x - pred_h / 2
#     pred_x2 = pred_x + pred_h / 2
#     pred_y1 = pred_y - pred_w / 2
#     pred_y2 = pred_y + pred_w / 2
#     true_x = true[..., 0]
#     true_y = true[..., 1]
#     true_h = true[..., 2]
#     true_w = true[..., 3]
#     true_x1 = true_x - true_h / 2
#     true_x2 = true_x + true_h / 2
#     true_y1 = true_y - true_w / 2
#     true_y2 = true_y + true_w / 2

#     # Intersection area
#     inter = (torch.min(pred_x2, true_x2) - torch.max(pred_x1, true_x1)).clamp(0) * \
#             (torch.min(pred_y2, true_y2) - torch.max(pred_y1, true_y1)).clamp(0)

#     # Union Area
#     union = pred_h * pred_w + true_h * true_w - inter + eps

#     # IoU
#     iou = inter / union

#     # CIoU
#     # convex (smallest enclosing box) width
#     cw = torch.max(pred_x2, true_x2) - torch.min(pred_x1, true_x1)
#     # convex height
#     ch = torch.max(pred_y2, true_y2) - torch.min(pred_y1, true_y1)

#     # convex diagonal squared
#     c2 = cw ** 2 + ch ** 2 + eps
#     # center dist ** 2
#     rho2 = ((true_x1 + true_x2 - pred_x1 - pred_x2) ** 2 + \
#             (true_y1 + true_y2 - pred_y1 - pred_y2) ** 2) / 4 

#     v = (4 / math.pi ** 2) * \
#         torch.pow(torch.atan(true_w / true_h) - torch.atan(pred_w / pred_h), 2)

#     with torch.no_grad():
#         alpha = v / (v - iou + (1 + eps))
#     return iou - (rho2 / c2 + v * alpha)  # CIoU


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU