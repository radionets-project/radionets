import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import fastcore.all as fc
from math import pi


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x -= self.sub
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x


class Mod(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x + pi) % (2*pi) - pi

act_gr = partial(GeneralRelu, leak=0.1, sub=0.4)
act_gr2 = partial(GeneralRelu, leak=0.1, sub=pi)

def conv(ni, nf, ks=3, stride=1, act=nn.ReLU, norm=None, bias=True, groups=1):
    layers = []
    if norm: layers.append(norm(ni))
    if act : layers.append(act())
    layers.append(nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2, bias=bias, groups=groups))
    return nn.Sequential(*layers)

def _conv_block(ni, nf, stride, act=act_gr, norm=None, ks=3, groups=1):
    return nn.Sequential(conv(ni, nf, stride=1     , act=act, norm=norm, ks=ks, groups=groups),
                         conv(nf, nf, stride=stride, act=act, norm=norm, ks=ks, groups=groups))

class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=3, act=act_gr, norm=None, param_groups=1):
        super().__init__()
        self.convs = _conv_block(ni, nf, stride, act=act, ks=ks, norm=norm, groups=param_groups)
        self.idconv = fc.noop if ni==nf else conv(ni, nf, ks=1, stride=1, act=None, norm=norm)
        self.pool = fc.noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return self.convs(x) + self.idconv(self.pool(x))


def up_block(ni, nf, ks=3, act=act_gr, norm=None):
    return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                         ResBlock(ni, nf, ks=ks, act=act, norm=norm))


def up_block2(ni, nf, ks=3, act=act_gr, norm=None):
    return nn.Sequential( 
        #ResBlock(ni, int(2*ni), ks=ks, act=act, norm=norm),
        #nn.PixelShuffle(2),
        nn.ConvTranspose2d(ni, nf, ks, padding=1, stride=2, output_padding=1),
        ResBlock(nf, nf, ks=ks, act=act, norm=norm))

class UNet(nn.Module):
    def __init__(self, act=act_gr, nfs=(16, 32, 64, 128, 128), norm=nn.InstanceNorm2d):
        super().__init__()
        self.start = ResBlock(2, nfs[0], stride=1, act=act, norm=norm, param_groups=2)
        self.dn = nn.ModuleList([ResBlock(nfs[i], nfs[i+1], act=act, norm=norm, stride=2)
                                 for i in range(len(nfs)-1)])
        self.up = nn.ModuleList([up_block(nfs[i], nfs[i-1], act=act, norm=norm)
                                 for i in range(len(nfs)-1,0,-1)])
        self.up += [ResBlock(nfs[0], 2, act=act, norm=norm)]
        self.end = ResBlock(2, 2, act=nn.Identity, norm=norm, param_groups=2)

    def forward(self, x):
        # x = x[:, 64:]
        #rand = torch.normal(0, 1, size=x.shape, device=x.device)
        #mask = x<1e-6
        #x[mask] += rand[mask]
        layers = []
        layers.append(x)
        x = self.start(x)
        for l in self.dn:
            layers.append(x)
            x = l(x)
        n = len(layers)
        for i,l in enumerate(self.up):
            if i!=0: x += layers[n-i]
            x = l(x)
        x = self.end(x + layers[0])
        return x


class UNet_phase(nn.Module):
    def __init__(self, act=nn.PReLU, nfs=(32, 64, 128, 256, 512), norm=nn.BatchNorm2d):
        super().__init__()
        self.start = ResBlock(1, nfs[0], stride=1, act=act, norm=norm)
        self.dn = nn.ModuleList([ResBlock(nfs[i], nfs[i+1], act=act, norm=norm, stride=2)
                                 for i in range(len(nfs)-1)])
        self.up = nn.ModuleList([up_block2(nfs[i], nfs[i-1], act=act, norm=norm)
                                 for i in range(len(nfs)-1,0,-1)])
        self.up += [ResBlock(nfs[0], 1, act=act, norm=norm)]
        self.end = ResBlock(2, 1, act=nn.Identity, norm=norm)
        self.mod = Mod()

    def forward(self, x):
        # x = x[:, 64:]
        #rand = torch.normal(0, 1, size=x.shape, device=x.device)
        #mask = (x<1e-6) & (x>-1e-6)
        #x[mask] += rand[mask]
        layers = []
        layers.append(x)
        x = self.start(x)
        for l in self.dn:
            layers.append(x)
            x = l(x)
        n = len(layers)
        for i,l in enumerate(self.up):
            if i!=0: x += layers[n-i]
            x = l(x)
        x = self.end(torch.cat([x, layers[0]], dim=1))
        x = self.mod(x)
        return x

