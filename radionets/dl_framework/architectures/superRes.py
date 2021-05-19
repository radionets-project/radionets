import torch
from torch import nn
from math import pi
from radionets.dl_framework.model import (
    GeneralELU,
    ResBlock_amp,
    ResBlock_phase,
    SRBlock,
    EDSRBaseBlock,
    RDB,
    FBB,
    Lambda,
    better_symmetry,
    tf_shift,
    btf_shift,
    CirculationShiftPad,
    SRBlockPad,
    BetterShiftPad,
    Lambda,
    symmetry,
)
from functools import partial
import torchvision
import radionets.evaluation.utils as ut
import numpy as np


class superRes_simple(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.conv1_amp = nn.Sequential(
            nn.Conv2d(1, 4, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv1_phase = nn.Sequential(
            nn.Conv2d(1, 4, stride=2, kernel_size=3, padding=3 // 2), GeneralELU(1 - pi)
        )
        self.conv2_amp = nn.Sequential(
            nn.Conv2d(4, 8, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv2_phase = nn.Sequential(
            nn.Conv2d(4, 8, stride=2, kernel_size=3, padding=3 // 2), GeneralELU(1 - pi)
        )
        self.conv3_amp = nn.Sequential(
            nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv3_phase = nn.Sequential(
            nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=3 // 2),
            GeneralELU(1 - pi),
        )
        self.conv4_amp = nn.Sequential(
            nn.Conv2d(16, 32, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv4_phase = nn.Sequential(
            nn.Conv2d(16, 32, stride=2, kernel_size=3, padding=3 // 2),
            GeneralELU(1 - pi),
        )
        self.conv5_amp = nn.Sequential(
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=3 // 2), nn.ReLU()
        )
        self.conv5_phase = nn.Sequential(
            nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=3 // 2),
            GeneralELU(1 - pi),
        )
        self.final_amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, img_size ** 2)
        )
        self.final_phase = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, img_size ** 2)
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.conv1_amp(amp)
        phase = self.conv1_phase(phase)

        amp = self.conv2_amp(amp)
        phase = self.conv2_phase(phase)

        amp = self.conv3_amp(amp)
        phase = self.conv3_phase(phase)

        amp = self.conv4_amp(amp)
        phase = self.conv4_phase(phase)

        amp = self.conv5_amp(amp)
        phase = self.conv5_phase(phase)

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class superRes_res18(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock_amp = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.preBlock_phase = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            GeneralELU(1 - pi),
        )

        self.maxpool_amp = nn.MaxPool2d(3, 2, 1)
        self.maxpool_phase = nn.MaxPool2d(3, 2, 1)

        # first block
        self.layer1_amp = nn.Sequential(ResBlock_amp(64, 64), ResBlock_amp(64, 64))
        self.layer1_phase = nn.Sequential(
            ResBlock_phase(64, 64), ResBlock_phase(64, 64)
        )

        self.layer2_amp = nn.Sequential(
            ResBlock_amp(64, 128, stride=2), ResBlock_amp(128, 128)
        )
        self.layer2_phase = nn.Sequential(
            ResBlock_phase(64, 128, stride=2), ResBlock_phase(128, 128)
        )

        self.layer3_amp = nn.Sequential(
            ResBlock_amp(128, 256, stride=2), ResBlock_amp(256, 256)
        )
        self.layer3_phase = nn.Sequential(
            ResBlock_phase(128, 256, stride=2), ResBlock_phase(256, 256)
        )

        self.layer4_amp = nn.Sequential(
            ResBlock_amp(256, 512, stride=2), ResBlock_amp(512, 512)
        )
        self.layer4_phase = nn.Sequential(
            ResBlock_phase(256, 512, stride=2), ResBlock_phase(512, 512)
        )

        self.final_amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )
        self.final_phase = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.preBlock_amp(amp)
        phase = self.preBlock_phase(phase)

        amp = self.maxpool_amp(amp)
        phase = self.maxpool_phase(phase)

        amp = self.layer1_amp(amp)
        phase = self.layer1_phase(phase)

        amp = self.layer2_amp(amp)
        phase = self.layer2_phase(phase)

        amp = self.layer3_amp(amp)
        phase = self.layer3_phase(phase)

        amp = self.layer4_amp(amp)
        phase = self.layer4_phase(phase)

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class superRes_res34(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

        self.preBlock_amp = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.preBlock_phase = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            GeneralELU(1 - pi),
        )

        self.maxpool_amp = nn.MaxPool2d(3, 2, 1)
        self.maxpool_phase = nn.MaxPool2d(3, 2, 1)

        # first block
        self.layer1_amp = nn.Sequential(
            ResBlock_amp(64, 64), ResBlock_amp(64, 64), ResBlock_amp(64, 64)
        )
        self.layer1_phase = nn.Sequential(
            ResBlock_phase(64, 64), ResBlock_phase(64, 64), ResBlock_phase(64, 64)
        )

        self.layer2_amp = nn.Sequential(
            ResBlock_amp(64, 128, stride=2),
            ResBlock_amp(128, 128),
            ResBlock_amp(128, 128),
            ResBlock_amp(128, 128),
        )
        self.layer2_phase = nn.Sequential(
            ResBlock_phase(64, 128, stride=2),
            ResBlock_phase(128, 128),
            ResBlock_phase(128, 128),
            ResBlock_phase(128, 128),
        )

        self.layer3_amp = nn.Sequential(
            ResBlock_amp(128, 256, stride=2),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
            ResBlock_amp(256, 256),
        )
        self.layer3_phase = nn.Sequential(
            ResBlock_phase(128, 256, stride=2),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
            ResBlock_phase(256, 256),
        )

        self.layer4_amp = nn.Sequential(
            ResBlock_amp(256, 512, stride=2),
            ResBlock_amp(512, 512),
            ResBlock_amp(512, 512),
        )
        self.layer4_phase = nn.Sequential(
            ResBlock_phase(256, 512, stride=2),
            ResBlock_phase(512, 512),
            ResBlock_phase(512, 512),
        )

        self.final_amp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )
        self.final_phase = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, img_size ** 2)
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.preBlock_amp(amp)
        phase = self.preBlock_phase(phase)

        amp = self.maxpool_amp(amp)
        phase = self.maxpool_phase(phase)

        amp = self.layer1_amp(amp)
        phase = self.layer1_phase(phase)

        amp = self.layer2_amp(amp)
        phase = self.layer2_phase(phase)

        amp = self.layer3_amp(amp)
        phase = self.layer3_phase(phase)

        amp = self.layer4_amp(amp)
        phase = self.layer4_phase(phase)

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class SRResNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        # torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock_amp = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4), nn.PReLU()
        )
        self.preBlock_phase = nn.Sequential(
            nn.Conv2d(1, 64, 9, stride=1, padding=4), nn.PReLU()
        )

        # ResBlock 16
        self.blocks_amp = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )
        self.blocks_phase = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock_amp = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )
        self.postBlock_phase = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final_amp = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4),
        )
        self.final_phase = nn.Sequential(
            nn.Conv2d(64, 1, 9, stride=1, padding=4),
        )

    def forward(self, x):
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        amp = self.preBlock_amp(amp)
        phase = self.preBlock_phase(phase)

        amp = amp + self.postBlock_amp(self.blocks_amp(amp))
        phase = phase + self.postBlock_phase(self.blocks_phase(phase))

        amp = self.final_amp(amp)
        phase = self.final_phase(phase)

        amp = amp.reshape(-1, 1, self.img_size, self.img_size)
        phase = phase.reshape(-1, 1, self.img_size, self.img_size)

        comb = torch.cat([amp, phase], dim=1)
        return comb


class SRResNet_corr(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 8
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),
        )

        #new symmetry

        self.symmetry = Lambda(better_symmetry)

        #pi layer
        self.pi = nn.Tanh()

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        # x[:,0][x[:,0]<0] = 0
        # x[:,0][x[:,0]>2] = 2
        # x[:,1] = np.pi*self.pi(x[:,1])

    

        return self.symmetry(x)

class SRResNet_sym(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        # torch.cuda.set_device(1)
        self.tf = Lambda(tf_shift)

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2),
        )

        #new symmetry

        self.btf = Lambda(btf_shift)

    def forward(self, x):
        x = self.tf(x)
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        return self.btf(x)

class SRResNet_sym_pad(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        # torch.cuda.set_device(1)
        self.tf = Lambda(tf_shift)

        self.preBlock = nn.Sequential(
            BetterShiftPad((4,4,4,4)),
            nn.Conv2d(2, 64, 9, stride=1, padding=0, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
            SRBlockPad(64, 64),
        )

        self.postBlock = nn.Sequential(
            BetterShiftPad((1,1,1,1)),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            BetterShiftPad((4,4,4,4)),
            nn.Conv2d(64, 2, 9, stride=1, padding=0, groups=2),
        )

        #new symmetry

        self.btf = Lambda(btf_shift)

    def forward(self, x):
        x = self.tf(x)
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        return self.btf(x)

class EDSRBase(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        # torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2)
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
            EDSRBaseBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2)
        )

    def forward(self, x):
        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        return x


class RDNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2, bias=False)
        )

        # ResBlock 6
        self.block1 = RDB(64, 32)
        self.block2 = RDB(64, 32)
        self.block3 = RDB(64, 32)
        self.block4 = RDB(64, 32)
        self.block5 = RDB(64, 32)
        self.block6 = RDB(64, 32)
        

        self.postBlock = nn.Sequential(
            nn.Conv2d(6*64, 64, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2, bias=False)
        )

    def forward(self, x):
        x = self.preBlock(x)

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        x = x + self.postBlock(torch.cat((x1,x2,x3,x4,x5,x6), dim=1))
        x = self.final(x)
        return x


class SRFBNet(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        # torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2, bias=False)
        )

        # ResBlock 6
        # self.block1 = FBB(64, 32, first=True)
        # self.block2 = FBB(64, 32)
        self.block1 = FBB(64, 32, first=True)

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2, bias=False)
        )

    def forward(self, x):
        x = self.preBlock(x)


        
        x1 = torch.zeros(x.shape).cuda()
        for i in range(4):
            x1 = self.block1(torch.cat((x,x1), dim=1))
            if i == 0:
                block = x1
            else:
                block = torch.cat((block,x1), dim=0)

        x = torch.cat((x,x,x,x), dim=0) + self.postBlock(block)
        x = self.final(x)
        return x


class vgg19_feature_maps(nn.Module):

    def __init__(self, i, j):
        super().__init__()
        # load pretrained vgg19
        # vgg19 = torchvision.models.vgg19(pretrained=True)
        # model = ut.load_pretrained_model(arch_name='vgg19_blackhole_group2', model_path='/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/models/vgg19_group2.model')
        # model = ut.load_pretrained_model(arch_name='vgg19_blackhole_group2_prelu', model_path='/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/models/vgg19_groups2_prelu.model')
        model = ut.load_pretrained_model(arch_name='vgg19_blackhole_fft', model_path='/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/models/vgg19_fft.model')
        
        vgg19 = model.vgg

        conv_counter = 0
        maxpool_counter = 0
        truncate_at = 0
        for layer in vgg19.features:
            truncate_at += 1

            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features)[:truncate_at + 1])

    def forward(self, x):
        amp_rescaled = (10 ** (10 * x[:,0]) - 1) / 10 ** 10
        phase = x[:,1]
        compl = amp_rescaled * torch.exp(1j * phase)
        ifft = torch.fft.ifft2(compl)
        img = torch.absolute(ifft)
        shift = torch.fft.fftshift(img)
        with torch.no_grad():
            feature = self.truncated_vgg19(img.unsqueeze(1))

        return feature



class vgg19_blackhole(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(1)
        vgg19 = torchvision.models.vgg19(pretrained=False)

        # customize vgg19
        vgg19.features[0] = nn.Conv2d(2, 64, 3, stride=1, padding=1)
        vgg19.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=6, bias=True))

        # for i, layer in enumerate(vgg19.features):
            # if isinstance(layer, nn.Conv2d):
            #     vgg19.features[i] = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, groups=2)
            # if isinstance(layer, nn.ReLU):
            #     vgg19.features[i] = nn.PReLU()

        self.vgg = vgg19
    
    def forward(self, x):
        return self.vgg(x)


class vgg19_blackhole_group2(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.cuda.set_device(1)
        vgg19 = torchvision.models.vgg19(pretrained=False)

        # customize vgg19

        vgg19.features[0] = nn.Conv2d(2, 64, 3, stride=1, padding=1)
        vgg19.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=6, bias=True))

        for i, layer in enumerate(vgg19.features):
            if isinstance(layer, nn.Conv2d):
                vgg19.features[i] = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, groups=2)
            # if isinstance(layer, nn.ReLU):
            #     vgg19.features[i] = nn.PReLU()

        self.vgg = vgg19
    
    def forward(self, x):
        #ifft
        return self.vgg(x)

class vgg19_blackhole_group2_prelu(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.cuda.set_device(1)
        vgg19 = torchvision.models.vgg19(pretrained=False)

        # customize vgg19
        vgg19.features[0] = nn.Conv2d(2, 64, 3, stride=1, padding=1)
        vgg19.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=6, bias=True))

        for i, layer in enumerate(vgg19.features):
            if isinstance(layer, nn.Conv2d):
                vgg19.features[i] = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, groups=2)
            if isinstance(layer, nn.ReLU):
                vgg19.features[i] = nn.PReLU()

        self.vgg = vgg19
    
    def forward(self, x):
        return self.vgg(x)

class vgg19_blackhole_fft(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.cuda.set_device(1)
        vgg19 = torchvision.models.vgg19(pretrained=False)

        # customize vgg19

        vgg19.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        vgg19.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=6, bias=True))

        # for i, layer in enumerate(vgg19.features):
        #     if isinstance(layer, nn.Conv2d):
        #         vgg19.features[i] = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, groups=2)
        #     # if isinstance(layer, nn.ReLU):
            #     vgg19.features[i] = nn.PReLU()

        self.vgg = vgg19
    
    def forward(self, x):
        #ifft
        amp_rescaled = (10 ** (10 * x[:,0]) - 1) / 10 ** 10
        phase = x[:,1]
        compl = amp_rescaled * torch.exp(1j * phase)
        ifft = torch.fft.ifft2(compl)
        img = torch.absolute(ifft)
        shift = torch.fft.fftshift(img)
        return self.vgg(img.unsqueeze(1))

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.LeakyReLU(0.2))

        self.block1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block5 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block6 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.block7 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.main = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7)

        self.postBlock = nn.Sequential(nn.Linear(512*4*4, 1024), nn.LeakyReLU(0.2), nn.Linear(1024,1), nn.Sigmoid())

    def forward(self, x):
        if x.shape[1] == 2:
            amp_x = (10 ** (10 * x[:,0]) - 1) / 10 ** 10
            phase_x = x[:,1]
            compl_x = amp_x * torch.exp(1j * phase_x)
            ifft_x = torch.fft.ifft2(compl_x)
            img_x = torch.absolute(ifft_x)
            shift_x = torch.fft.fftshift(img_x).unsqueeze(1)
            x = shift_x
        x = self.preBlock(x)
        x = self.main(x)
        x = torch.flatten(x, 1)
        x = self.postBlock(x)


        return x


class automap(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(1)
        self.fcs = nn.Sequential(nn.Linear(63*63*2,63*63),nn.Tanh(),nn.Linear(63*63,63*63),nn.Tanh())

        self.convs = nn.Sequential(nn.Conv2d(1,64,5, stride=1, padding=2),nn.ReLU(),nn.Conv2d(64,64,5,stride=1, padding=2),nn.ReLU(),nn.Conv2d(64,1,7,stride=1, padding=3))

    
    def forward(self, x):
        amp_x = (10 ** (10 * x[:,0]) - 1) / 10 ** 10
        phase_x = x[:,1]
        compl_x = amp_x * torch.exp(1j * phase_x)
        x[:,0] = compl_x.real
        x[:,1] = compl_x.imag
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        x = x.reshape((x.shape[0],1,63,63))
        x = self.convs(x)
        return x