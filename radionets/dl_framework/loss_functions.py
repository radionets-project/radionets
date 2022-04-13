import torch
from torch import nn
from torchvision.models import vgg16_bn
from radionets.dl_framework.utils import children
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from scipy.optimize import linear_sum_assignment
from radionets.dl_framework.architectures import superRes
from fastai.vision.gan import _tk_mean
import matplotlib.pyplot as plt


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

def l1_phyinfo(x, y):
    l1 = nn.L1Loss()
    return l1(x[1],y[0])

def l1_GANCS(x,y):
    amp = y[:,0].clone().detach()
    phase = y[:,1].clone().detach()
    amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
    compl = amp_rescaled * torch.exp(1j * phase)
    ifft = torch.fft.ifft2(compl)
    spatial = torch.fft.ifftshift(ifft).unsqueeze(1)
    # change to two channels real/imag
    # input = torch.zeros(y.shape)
    # input[:,0] = spatial.real
    # input[:,1] = spatial.imag

    l1 = nn.L1Loss()

    return l1(x,spatial)

def l1_CLEAN(x,y):
    amp = y[:,0].clone().detach()
    phase = y[:,1].clone().detach()
    amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
    compl = amp_rescaled * torch.exp(1j * phase)
    ifft = torch.fft.ifft2(compl)
    spatial = torch.fft.ifftshift(ifft).unsqueeze(1)


    l1 = nn.L1Loss()

    return l1((x[1][:,0]+1j*x[1][:,1]).unsqueeze(1),spatial)

def l1_RIM(x,y):
    amp = y[:,0].clone().detach()
    phase = y[:,1].clone().detach()
    amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
    compl = amp_rescaled * torch.exp(1j * phase)
    ifft = torch.fft.ifft2(compl)
    spatial = torch.fft.ifftshift(ifft).unsqueeze(1)


    l1 = nn.L1Loss()
    loss = 0
    for eta in x:
        loss += l1((eta[:,0]+1j*eta[:,1]).unsqueeze(1),spatial)

    loss = loss/len(x)
    return loss

def mse_RIM(x,y):
    amp = y[:,0].clone().detach()
    phase = y[:,1].clone().detach()
    amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
    compl = amp_rescaled * torch.exp(1j * phase)
    compl_shift = torch.fft.fftshift(compl)
    ifft = torch.fft.ifft2(compl_shift, norm="forward")
    true = torch.fft.ifftshift(ifft).unsqueeze(1)

    
    complex2channels_y = torch.cat((true.real,true.imag), dim=1)
    
    mse = nn.MSELoss()
    loss = 0
    for eta in x:
        complex2channels_x = torch.cat((eta.real,eta.imag), dim=1)
        loss += mse(complex2channels_x*eta.shape[2]**2,complex2channels_y)

    loss = loss/len(x)
    return loss

def l1_wgan_GANCS(fake_pred,x,y):
    amp = y[:,0].clone().detach()
    phase = y[:,1].clone().detach()
    amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
    compl = amp_rescaled * torch.exp(1j * phase)
    ifft = torch.fft.ifft2(compl)
    spatial = torch.fft.ifftshift(ifft).unsqueeze(1)

    l1 = nn.L1Loss()
    lamb = 1e-5

    return l1(x,spatial)+lamb*_tk_mean(fake_pred, x, spatial)

def dirty_model(x, y):
    amp = x[1][:,0]
    phase = x[1][:,1]
    amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
    compl = amp_rescaled * torch.exp(1j * phase)
    ifft = torch.fft.ifft2(compl)
    pred = torch.fft.ifftshift(torch.absolute(ifft)).unsqueeze(1)
    
    # amp_t = y[0][:,0]
    # phase_t = y[0][:,1]
    # amp_rescaled_t = (10 ** (10 * amp_t) - 1) / 10 ** 10
    # compl_t = amp_rescaled_t * torch.exp(1j * phase_t)
    # ifft_t = torch.fft.ifft2(compl_t)
    # true = torch.fft.ifftshift(torch.absolute(ifft_t)).unsqueeze(1)

    
    base_nums = torch.zeros(45) #hard code
    n_tel = 10 #hardcode
    c = 0
    for i in range(n_tel):
        for j in range(n_tel):
            if j<=i:
                continue
            base_nums[c] = 256 * (i + 1) + j + 1
            c += 1

    base_mask = y[1]
    A = y[2]
    MD = torch.zeros(pred.shape, dtype=torch.complex64).to('cuda')

      
    for idx, bn in enumerate(base_nums):
        s_uv = torch.sum((base_mask == bn),3)
        if not (base_mask == bn).any():
            continue
        AI = torch.einsum('blm,bclm->bclm',A[...,idx],pred)
        MD += torch.einsum('blm,bclm->bclm',s_uv,torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(AI)))) #spatial
    
    points = base_mask.clone()
    points[points != 0] = 1
    points = torch.sum(points,3)
    points[points == 0] = 1
    
    MD = torch.fft.ifftshift(torch.absolute(torch.fft.ifft2(MD/points.unsqueeze(1))))
    
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    loss = l1(pred, x[0])
    # loss = vgg19_feature_loss(MD,x[0])
    return loss

# vgg19 = superRes.vgg19_feature_maps(5,4).eval().to('cuda:1')
def vgg19_feature_loss(x, y):
    print()
    if 'vgg19_feature_model_12' not in globals():
        global vgg19_feature_model_12
        # global vgg19_feature_model_22
        # global vgg19_feature_model_34
        # global vgg19_feature_model_44
        # global vgg19_feature_model_54
        vgg19_feature_model_12 = superRes.vgg19_feature_maps(1,2).eval().to('cuda:1')
        # vgg19_feature_model_22 = superRes.vgg19_feature_maps(2,2).eval().to('cuda:1')
        # vgg19_feature_model_34 = superRes.vgg19_feature_maps(3,4).eval().to('cuda:1')
        # vgg19_feature_model_44 = superRes.vgg19_feature_maps(4,4).eval().to('cuda:1')
        # vgg19_feature_model_54 = superRes.vgg19_feature_maps(5,4).eval().to('cuda:1')

    
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # up1 = nn.Upsample(size=7, mode='nearest').to('cuda:1')
    # up2 = nn.Upsample(size=15, mode='nearest').to('cuda:1')
    # up3 = nn.Upsample(size=31, mode='nearest').to('cuda:1')
    # up4 = nn.Upsample(size=63, mode='nearest').to('cuda:1')
    # c34 = nn.Conv2d(256, 512, 1).to('cuda:1')
    # c22 = nn.Conv2d(128, 512, 1).to('cuda:1')
    # c12 = nn.Conv2d(64, 512, 1).to('cuda:1')

    # upx1 = up1(vgg19_feature_model_54(x))
    # upy1 = up1(vgg19_feature_model_54(y))

    # upx2 = up2(vgg19_feature_model_44(x) + upx1)
    # upy2 = up2(vgg19_feature_model_44(y) + upy1)

    # upx3 = up3(c34(vgg19_feature_model_34(x)) + upx2)
    # upy3 = up3(c34(vgg19_feature_model_34(y)) + upy2)

    # upx4 = up4(c22(vgg19_feature_model_22(x)) + upx3)
    # upy4 = up4(c22(vgg19_feature_model_22(y)) + upy3)

    # upx5 = (c12(vgg19_feature_model_12(x)) + upx4)
    # upy5 = (c12(vgg19_feature_model_12(y)) + upy4)


    # mix_x = (0.5*x[:,0]+0.5*x[:,1]).unsqueeze(1)
    # mix_y = (0.5*y[:,0]+0.5*y[:,1]).unsqueeze(1)
    # ones = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to('cuda:1')
    # x_3c = torch.cat((x, ones), dim=1)
    # y_3c = torch.cat((y, ones), dim=1)

    # loss = l1(vgg19_feature_model_22(x), vgg19_feature_model_22(y))# + l1(vgg19_feature_model_12(x), vgg19_feature_model_12(y)) + l1(vgg19_feature_model_34(x), vgg19_feature_model_34(y)) + l1(vgg19_feature_model_44(x), vgg19_feature_model_44(y)) + l1(vgg19_feature_model_54(x), vgg19_feature_model_54(y))
    loss = l1(vgg19_feature_model_22(x), vgg19_feature_model_22(y))
    return loss

def gen_loss_func(fake_pred, x, y):
    l1 = nn.L1Loss()
    bce = nn.BCELoss()

    # mask = torch.zeros(x.shape).to('cuda:1')
    # mask[:,:,31-10:31+10,31-10:31+10]=1
    # xm = torch.einsum('bcij,bcjk->bcik',x,mask)
    # ym = torch.einsum('bcij,bcjk->bcik',y,mask)

    content_loss = l1(x,y)
    # content_loss = automap_l2(x,y)
    # content_loss = vgg19_feature_loss(x,y)
    adv_loss = bce(fake_pred, torch.ones_like(fake_pred))

    return content_loss + 1e-3*adv_loss

def gen_loss_wgan_l1(fake_pred, x, y):
    l1 = nn.L1Loss()
    content_loss = l1(x[1], y[0])


    adv_loss = _tk_mean(fake_pred, x, y)
    lamb = 1.5 # first:0.9

    return lamb*content_loss + (1-lamb)*adv_loss


def gen_loss_func_physics_informed(fake_pred, x, y):



    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    ######### physics informed stuff
    # base_nums = torch.zeros(45) #hard code
    # n_tel = 10 #hardcode
    # c = 0
    # for i in range(n_tel):
    #     for j in range(n_tel):
    #         if j<=i:
    #             continue
    #         base_nums[c] = 256 * (i + 1) + j + 1
    #         c += 1

    # base_mask = y[1]
    # A = y[2]
    # MD = torch.zeros(x[1].shape, dtype=torch.complex64).to('cuda')

  
   
        
    # for idx, bn in enumerate(base_nums):
    #     s_uv = torch.sum((base_mask == bn),3)
    #     if not (base_mask == bn).any():
    #         continue
    #     AI = torch.einsum('blm,bclm->bclm',A[...,idx],x[1])
    #     MD += torch.einsum('blm,bclm->bclm',s_uv,torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(AI)))) #spatial
    
    # points = base_mask.clone()
    # points[points != 0] = 1
    # points = torch.sum(points,3)
    # points[points == 0] = 1
    
    # MD = torch.fft.ifftshift(torch.absolute(torch.fft.ifft2(MD/points.unsqueeze(1))))
    

    content_loss = l1(x[1], y[0])
    # print(fake_pred.requires_grad)
    adv_loss = bce(fake_pred, torch.ones_like(fake_pred))

    return 1e-3*adv_loss +content_loss



def crit_loss_func(real_pred, fake_pred):
    bce = nn.BCELoss()
    loss = bce(real_pred, torch.ones_like(real_pred)) + bce(fake_pred, torch.zeros_like(fake_pred))
    # print(fake_pred.requires_grad)
    return loss

def cross_entropy(x,y):
    loss = nn.CrossEntropyLoss()
    return loss(x, y.squeeze().long())


def l1_rnn(x, y):
    l1 = nn.L1Loss()
    x = torch.chunk(x, 4, dim=0)

    l = 0
    for i in range(4):
        l += l1(x[i], y)
    return l/4

def splitted_l1(x, y):
    l1 = nn.L1Loss()
    l = (10*l1(x[:,0], y[:,0]) + l1(x[:,1], y[:,1]))/2
    return l

def l1_ssim(x,y):
    fft_x, fft_y = inspec.fft_pred_torch(x,y)
    l1 = nn.L1Loss()
    print(inspec.ssim_torch(fft_x, fft_y).shape)
    l = (l1(fft_x, fft_y) + (1-inspec.ssim_torch(fft_x, fft_y)))/2
    return l



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
