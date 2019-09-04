import torch.nn as nn


def conv(ni, nc, ks, stride):
    conv = nn.Conv2d(ni, nc, ks, stride),
    bn = nn.BatchNorm2d(nc),
    act = nn.ReLU()
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