import torch.nn as nn
import torch.nn.functional as F

def conv(ni, nc, ks, stride):
    conv = nn.Conv2d(ni, nc, ks, stride),
    bn = nn.BatchNorm2d(nc),
    act = GeneralRelu(leak=0.4) #nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def init_cnn(model):
    for l in model:
        if isinstance(l, nn.Conv2d):
            print(l)
            nn.init.kaiming_uniform_(l.weight)
            l.bias.data.zero_()


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.data = data


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x): return self.func(x)

def flatten(x): return x.view(x.shape[0], -1)


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x