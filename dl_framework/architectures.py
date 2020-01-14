from torch import nn
from dl_framework.model import conv, Lambda, flatten, GeneralRelu
import torch
from collections import OrderedDict

def cnn():
    """
    conv-layer: number of entry channels, number of exit channels,
                kerner size, stride, padding
    """
    arch = nn.Sequential(
        *conv(1, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        Lambda(flatten),
        nn.Linear(64, 4096)
    )
    return arch

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 4)
        self.dconv_down2 = double_conv(4, 8)
        self.dconv_down3 = double_conv(8, 16)
        self.dconv_down4 = double_conv(16, 32)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(16 + 32, 16)
        self.dconv_up2 = double_conv(8 + 16, 8)
        self.dconv_up1 = double_conv(4 + 8, 4)
        
        self.conv_last = nn.Conv2d(4, 1, 1)

        self.flatten = Flatten()
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)

        out = self.flatten(x)
        
        return out

def autoencoder():
    arch = nn.Sequential(
        *conv(1, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        nn.ConvTranspose2d(64,32,(3,3),2,1,1),
        nn.BatchNorm2d(32),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(32,16,(3,3),2,1,1),
        nn.BatchNorm2d(16),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(16,16,(3,3),2,1,1),
        nn.BatchNorm2d(16),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(16,8,(3,3),2,1,1),
        nn.BatchNorm2d(8),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(8,4,(3,3),2,1,1),
        nn.BatchNorm2d(4),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(4,1,(3,3),2,1,1),
        Flatten()
    )
    return arch

