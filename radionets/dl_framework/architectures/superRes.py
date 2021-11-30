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
    fft,
    gradFunc2,
    manual_grad,
    tf_shift,
    btf_shift,
    CirculationShiftPad,
    SRBlockPad,
    BetterShiftPad,
    Lambda,
    symmetry,
    SRBlock_noBias,
    HardDC,
    SoftDC,
    calc_DirtyBeam,
    gauss,
    ConvGRUCell,
    gradFunc,
    gradFunc2,
    gradFunc_putzky,
    fft_conv,
    ConvGRUCellBN,
)
from functools import partial
import torchvision
import radionets.evaluation.utils as ut
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# import irim.rim as rim

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
        # torch.cuda.set_device(1)
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
        # self.pi = nn.Tanh()

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
        # torch.cuda.set_device(1)
        # load pretrained vgg19
        # vgg19 = torchvision.models.vgg19(pretrained=True)
        # model = ut.load_pretrained_model(arch_name='vgg19_blackhole_group2', model_path='/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/models/vgg19_group2.model')
        # model = ut.load_pretrained_model(arch_name='vgg19_blackhole_group2_prelu', model_path='/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/models/vgg19_groups2_prelu.model')
        # model = ut.load_pretrained_model(arch_name='vgg19_blackhole_fft', model_path='/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/models/vgg19_fft.model')
        model = ut.load_pretrained_model(arch_name='vgg19_one_channel', model_path='/net/big-tank/POOL/projects/radio/simulations/jets/260521/model/temp_30.model')
        
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
        for param in self.truncated_vgg19.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.shape[1] == 2:
            amp_rescaled = (10 ** (10 * x[:,0]) - 1) / 10 ** 10
            phase = x[:,1]
            compl = amp_rescaled * torch.exp(1j * phase)
            ifft = torch.fft.ifft2(compl)
            img = torch.absolute(ifft)
            shift = torch.fft.fftshift(img)
            with torch.no_grad():
                feature = self.truncated_vgg19(img.unsqueeze(1))
        else:
            feature = self.truncated_vgg19(x) 
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

class vgg19_one_channel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(1)
        vgg19 = torchvision.models.vgg19(pretrained=False)

        # customize vgg19

        vgg19.features[0] = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        vgg19.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=2, bias=True))

        self.vgg = vgg19
    
    def forward(self, x):
        #ifft
        return self.vgg(x)

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(0)

        self.preBlock = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.LeakyReLU(0.2))

        self.block1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block5 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block6 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.block7 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.main = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7)

        # self.postBlock = nn.Sequential(nn.Linear(512*4*4, 1024), nn.LeakyReLU(0.2), nn.Linear(1024,1), nn.Sigmoid()) #GAN
        self.postBlock = nn.Sequential(nn.Linear(512*4*4, 1024), nn.LeakyReLU(0.2), nn.Linear(1024,1)) #WGAN

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) == 2:
                x = x[1]
            else:
                x = x[0]
            
        if x.shape[1] == 2:
            amp_x = (10 ** (10 * x[:,0]) - 1) / 10 ** 10
            phase_x = x[:,1]
            compl_x = amp_x * torch.exp(1j * phase_x)
            ifft_x = torch.fft.ifft2(compl_x)
            img_x = torch.absolute(ifft_x)
            shift_x = torch.fft.ifftshift(img_x).unsqueeze(1)
        else:
            shift_x = x
        # shift_x[torch.isnan(shift_x)] = 0
        pred = self.preBlock(shift_x)
        pred = self.main(pred)
        pred = torch.flatten(pred, 1)
        pred = self.postBlock(pred)
        return pred

class SRResNet_dirtyModel_pretrainedL1(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.model = ut.load_pretrained_model(arch_name='SRResNet_dirtyModel', model_path='/net/big-tank/POOL/users/sfroese/vipy/jets/models/l1_symmetry.model')

    def forward(self, x):
        return self.model(x)


class SRResNet_dirtyModel(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        torch.cuda.set_device(1)
        self.img_size = img_size

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=1), nn.PReLU()
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
        # self.upscale = nn.Sequential(
        #     nn.Conv2d(64, 256, 3, stride=1, padding = 1),
        #     nn.PixelShuffle(2),
        #     nn.PReLU()
        # )
        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=1),
        )

        self.relu = nn.Hardtanh(0,1.1)
        self.pi = nn.Hardtanh(-np.pi,np.pi)

        self.symmetry = Lambda(better_symmetry)


    def forward(self, x):

        amp = x[:,0].clone().detach()
        phase = x[:,1].clone().detach()
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase)
        ifft = torch.fft.ifft2(compl)
        dirty = torch.fft.ifftshift(torch.absolute(ifft))
        dirty = dirty.unsqueeze(1)


        pred = self.preBlock(x)

        pred = pred + self.postBlock(self.blocks(pred))
        # pred = self.postBlock(self.blocks(pred))


        # pred = self.upscale(pred)

        pred = self.final(pred)

        pred[:,0] = self.relu(pred[:,0].clone())
        pred[:,1] = self.pi(pred[:,1].clone())

        pred = self.symmetry(pred)


        # pred = self.relu(pred)
        # pred = nn.functional.interpolate(pred, scale_factor=0.5)

        return dirty, pred


class GANCS_generator_test(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(1)
        self.blocks = nn.Sequential(
            SRBlock(2, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.post = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(64, 2, 1, stride=1, padding=0)
        )

        self.DC = HardDC(45, 10)
    
    def forward(self, x):
        ap = x[0]
        base_mask = x[1]
        A = x[2]


        amp = ap[:,0].clone().detach()
        phase = ap[:,1].clone().detach()
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        ifft = torch.fft.ifft2(compl)
        spatial = torch.fft.ifftshift(ifft) 
        # change to two channels real/imag
        input = torch.zeros(ap.shape).to('cuda')
        input[:,0] = spatial.real
        input[:,1] = spatial.imag
        # dirty = input.clone().detach()


        pred = self.blocks(input)

        pred = self.post(pred)

        pred = self.DC(pred, compl.unsqueeze(1), A, base_mask)
        

        return pred

class GANCS_generator(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(1)
        self.blocks = nn.Sequential(
            SRBlock(2, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.post = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(64, 2, 1, stride=1, padding=0)
        )

        self.DC = HardDC(45, 10)
    
    def forward(self, x):
        ap = x[0]
        base_mask = x[1]
        A = x[2]


        amp = ap[:,0].clone().detach()
        phase = ap[:,1].clone().detach()
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        ifft = torch.fft.ifft2(compl)
        spatial = torch.fft.ifftshift(ifft) 
        # change to two channels real/imag
        input = torch.zeros(ap.shape).to('cuda')
        input[:,0] = spatial.real
        input[:,1] = spatial.imag
        # dirty = input.clone().detach()


        pred = self.blocks(input)

        pred = self.post(pred)

        pred = self.DC(pred, compl.unsqueeze(1), A, base_mask)
        

        return pred


class GANCS_critic(nn.Module):
    def __init__(self):
        super().__init__()
        # self.blocks = nn.Sequential(
        #     nn.Conv2d(2, 4, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(4, 8, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(8, 16, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(16, 32, 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 32, 3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 32, 1, stride=1, padding=0),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 1, 1, stride=1, padding=0),
        #     nn.AdaptiveAvgPool2d(1)
        # )
        self.block1 = nn.Sequential(nn.Conv2d(2, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block5 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.block6 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.block7 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        self.blocks = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7, nn.AdaptiveAvgPool2d(1))
            
    def forward(self, x):
        if x.shape[1] == 2:
            amp = x[:,0].clone().detach()
            phase = x[:,1].clone().detach()
            amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
            compl = amp_rescaled * torch.exp(1j * phase)
            ifft = torch.fft.ifft2(compl)
            x = torch.fft.ifftshift(ifft).unsqueeze(1)
        input = torch.zeros((x.shape[0],2,x.shape[2], x.shape[3])).to('cuda')
        input[:,0] = x.real.squeeze(1)
        input[:,1] = x.imag.squeeze(1)
        return self.blocks(input)


class GANCS_unrolled(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(0)
        self.block1 = nn.Sequential(
            SRBlock(2, 64),
            SRBlock(64, 64),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )
        self.DC1 = SoftDC(45, 10)
        self.block2 = nn.Sequential(
            SRBlock(2, 64),
            SRBlock(64, 64),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )
        self.DC2 = SoftDC(45, 10)
        self.block3 = nn.Sequential(
            SRBlock(2, 64),
            SRBlock(64, 64),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )
        self.DC3 = HardDC(45, 10)
        # self.block4 = nn.Sequential(
        #     SRBlock(2, 64),
        #     SRBlock(64, 64),
        #     nn.Conv2d(64, 2, 3, stride=1, padding=1),
        # )
        # self.DC4 = SoftDC(45, 10)
        # self.block5 = nn.Sequential(
        #     SRBlock(2, 64),
        #     SRBlock(64, 64),
        #     nn.Conv2d(64, 2, 3, stride=1, padding=1),
        # )
        # self.DC5 = SoftDC(45, 10)
    
    def forward(self, x):
        ap = x[0]
        base_mask = x[1]
        A = x[2]


        amp = ap[:,0].clone().detach()
        phase = ap[:,1].clone().detach()
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        ifft = torch.fft.ifft2(compl)
        spatial = torch.fft.ifftshift(ifft) 
        # change to two channels real/imag
        input = torch.zeros(ap.shape).to('cuda')
        input[:,0] = spatial.real
        input[:,1] = spatial.imag
        measured = input.clone().detach()
        # dirty = input.clone().detach()


        pred = self.block1(input)
        dc1 = self.DC1(pred, measured, A, base_mask)
        # pred[:,0] = dc1.real.squeeze(1)
        # pred[:,1] = dc1.imag.squeeze(1)
        pred = self.block2(pred)
        dc2 = self.DC2(pred, measured, A, base_mask)
        # pred[:,0] = dc2.real.squeeze(1)
        # pred[:,1] = dc2.imag.squeeze(1)
        pred = self.block3(pred)
        dc3 = self.DC3(pred, compl.unsqueeze(1), A, base_mask)
        pred[:,0] = dc3.real.squeeze(1)
        pred[:,1] = dc3.imag.squeeze(1)
        # pred = self.block4(pred)
        # pred = self.DC4(pred, measured, A, base_mask)
        # pred = self.block5(pred)
        # pred = self.DC5(pred, measured, A, base_mask)

        #pred = self.post(pred)

        #pred = self.DC(pred, compl.unsqueeze(1), A, base_mask)
        
        return (pred[:,0]+1j*pred[:,1]).unsqueeze(1)


class CLEANNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(0)

        self.blocks = nn.Sequential(
            SRBlock(2, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64)
        )

        self.post = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(64, 2, 1, stride=1, padding=0)
        )

        self.lamb = nn.Parameter(torch.tensor(1).float())

        # self.conv = nn.Conv2d(2, 2, 3, stride=1, padding=1, bias=False)

        # self.DC = HardDC(45, 10)
        # self.beamBlock = nn.Sequential(
        #     SRBlock(2, 64),
        #     SRBlock(64, 64),
        #     SRBlock(64, 64),
        #     SRBlock(64, 64),
        #     SRBlock(64, 64),
        #     nn.Conv2d(64, 2, 1, stride=1, padding=0)
        # )
        # self.block1 = nn.Sequential(nn.Conv2d(2, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.block3 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        # self.block5 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        # self.block6 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        # self.block7 = nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))

        # self.beamBlock = nn.Sequential(self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7, nn.Conv2d(512, 1, 3, stride=1, padding=1), nn.AdaptiveAvgPool2d(1), nn.Hardtanh(1,3))


    def forward(self, x):
        # print(len(x))
        ap = x[0]
        base_mask = x[1]
        A = x[2]
        M = x[3]

        amp = ap[:,0].clone().detach()
        phase = ap[:,1].clone().detach()
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        ifft = torch.fft.ifft2(compl)
        spatial = torch.fft.ifftshift(ifft) 
        # change to two channels real/imag
        input = torch.zeros(ap.shape).to('cuda')
        input[:,0] = spatial.real
        input[:,1] = spatial.imag
        # measured = input.clone().detach()

        #calculate Dirty Beam
        beam = calc_DirtyBeam(base_mask)
        # beam_copy = beam.clone().detach()

        # M = torch.zeros(input.shape).to('cuda')


        # for i in range(5):
        out_b = self.blocks(input)
        out_p = self.post(out_b)
    

        # residual = input - self.lamb*torch.einsum('bclm,bclm->bclm', out_p, beam)
        residual = input - self.lamb*out_p

        M = M + self.lamb*out_p

            # if i == 4:
            #     break

        # return (input[:,0]+1j*input[:,1]).unsqueeze(1)
        # return (M[:,0]+1j*M[:,1]).unsqueeze(1)

        # gauss_params = self.beamBlock(beam)

        # clean_beam = torch.fft.ifft2(torch.fft.fft2(torch.cat([gauss(63,s) for s in gauss_params]).reshape(-1,M.shape[2],M.shape[2])))

        # M = M + input
        # return clean_beam
        # M_compl = (M[:,0]+1j*M[:,1])
       

        # M_conv = torch.einsum('blm,blm->blm',  torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(M_compl))),  torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(clean_beam))))
        
        # M_conv = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(M_conv)))
        fft_residual = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(residual[:,0]+1j*residual[:,1])))

        res_amp = torch.absolute(fft_residual)
        res_phase = torch.angle(fft_residual)

        residual[:,0] = ((torch.log10(res_amp + 1e-10) / 10) + 1)
        residual[:,1] = res_phase

        return residual, M

class ConvRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, dilation=1, padding=2), #padding = dilation * (ks-1) // 2
            nn.ReLU(),
        )
        self.GRU1 = ConvGRUCell(64, 64, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, dilation=2, padding=2),
            nn.ReLU(),
        )
        self.GRU2 = ConvGRUCell(64, 64, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 2, 3, stride=1, dilation=1, padding=1, bias=False)
        )

    def forward(self, x, hx=None):
        if not hx:
            hx = [None]*2
        
        c1 = self.conv1(x)
        g1 = self.GRU1(c1, hx[0])
        c2 = self.conv2(g1)
        g2 = self.GRU2(c2, hx[1])
        c3 = self.conv3(g2)



        return c3, [g1.detach(), g2.detach()]

class RIM(nn.Module):
    def __init__(self, n_steps=20):
        super().__init__()
        torch.cuda.set_device(1)
        self.n_steps = n_steps
        self.cRNN = ConvRNN()
        # self.bn = nn.BatchNorm2d(2)
    

    def forward(self, x, hx=None):
        ap = x[0]
        amp = ap[:,0]
        phase = ap[:,1]
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        data = compl.clone().detach()
        compl_shift = torch.fft.fftshift(compl) # shift low freq to corner
        ifft = torch.fft.ifft2(compl_shift)
        ifft_shift = torch.fft.ifftshift(ifft) # shift low freq to center
        eta = torch.zeros(ap.shape).to('cuda')
        eta[:,0] = ifft_shift.real
        eta[:,1] = ifft_shift.imag


        
        etas = []

        for i in range(self.n_steps):

            grad = gradFunc(eta, data, x[2], x[1], 8, 45).detach()
            # bn = self.bn(grad)
            input = torch.cat((eta,grad), dim=1)

            delta, hx = self.cRNN(input, hx)
            eta = eta + delta
            # plt.imshow(torch.absolute(bn[0,0]+1j*bn[0,1]).cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
            etas.append(eta)
        
        return etas


class ConvRNN_deepClean(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 11, stride=3, dilation=1, padding=2), # use stride=4 for 63 px images
            nn.Tanh(),
        )
        self.GRU1 = ConvGRUCell(64, 64, 11)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 11, stride=3, dilation=1, padding=2),
            nn.Tanh(),
        )
        self.GRU2 = ConvGRUCell(64, 64, 11)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 2, 11, stride=1, dilation=1, padding=5, bias=False)
        )
        # self.weight = nn.Parameter(torch.tensor([0.25]))

    def forward(self, x, hx=None):
        if not hx:
            hx = [None]*2

        complex2channels = torch.cat((x[:,0].real.unsqueeze(1),x[:,0].imag.unsqueeze(1),x[:,1].real.unsqueeze(1),x[:,1].imag.unsqueeze(1)), dim=1)
        # print(complex2channels.shape)
        # print(complex2channels.dtype)

        c1 = self.conv1(complex2channels)
        g1 = self.GRU1(c1, hx[0])
        c2 = self.conv2(g1)
        g2 = self.GRU2(c2, hx[1])
        c3 = self.conv3(g2)

        # plt.imshow(torch.absolute(c2[0,0]+1j*c2[0,1]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        channels2complex = (c3[:,0]+1j*c3[:,1]).unsqueeze(1)

        return channels2complex, [g1.detach(), g2.detach()] # ??? detach() ???

class RIM_DC(nn.Module):
    def __init__(self, n_steps=10):
        super().__init__()
        torch.cuda.set_device(0)
        # torch.set_default_dtype(torch.float64) ## this is really important since we do a lot of ffts. otherwise torch.zeros is float32 and we can't save complex128 into it!
        self.n_steps = n_steps
        
        self.cRNN = ConvRNN_deepClean()
        # self.type(torch.complex64)
        # torch.backends.cudnn.enabled = False
        

    def forward(self, x, hx=None):
        ap = x[0]
        amp = ap[:,0]
        phase = ap[:,1]
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        

        data = compl.clone().detach().unsqueeze(1)
        compl_shift = torch.fft.fftshift(compl) # shift low freq to corner
        ifft = torch.fft.ifft2(compl_shift, norm="forward")
        eta = torch.fft.ifftshift(ifft).unsqueeze(1) # shift low freq to center
        # print(eta.shape)
        # eta = torch.zeros(ap.shape, dtype=torch.float64).to('cuda')
        # eta[:,0] = ifft_shift.real
        # eta[:,1] = ifft_shift.imag


        
        etas = []
        # plt.imshow(torch.abs(eta[0,0]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        for i in range(self.n_steps):

            grad = gradFunc_putzky(eta.detach(), [data, x[1]]).detach()
            # grad = manual_grad(eta.detach(), [data, x[1]]).detach()
            # plt.imshow(torch.absolute(grad[0,0]+1j*grad[0,1]).cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
            # break

            input = torch.cat((eta.detach(),grad), dim=1)
            delta, hx = self.cRNN(input, hx)
            # plt.imshow(torch.abs(grad[0,0]).cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
            # plt.imshow(torch.absolute(grad[0,0]).cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
            # print(hx[0].requires_grad)
            # plt.imshow(torch.abs(delta[0,0]).cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
            eta = eta.detach() + delta
            # plt.imshow(torch.abs(eta[0,0]).cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()

            etas.append(eta)


        
        return [eta/eta.shape[2]**2 for eta in etas]


class ConvRNN_deepClean_noDetach(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 11, stride=4, dilation=1, padding=2), # use stride=4 for 63 px images # for blackhole model use 4, 64, 11, stride=3, dilation=1, padding=2
            nn.Tanh(),
        )
        self.GRU1 = ConvGRUCell(64, 64, 11)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 11, stride=4, dilation=1, padding=2),
            nn.Tanh(),
        )
        self.GRU2 = ConvGRUCell(64, 64, 11)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 2, 11, stride=1, dilation=1, padding=5, bias=False)
        )
        # self.weight = nn.Parameter(torch.tensor([0.25]))

    def forward(self, x, hx=None):
        if not hx:
            hx = [None]*2

        complex2channels = torch.cat((x[:,0].real.unsqueeze(1),x[:,0].imag.unsqueeze(1),x[:,1].real.unsqueeze(1),x[:,1].imag.unsqueeze(1)), dim=1)
        # print(complex2channels.shape)
        # print(complex2channels.dtype)

        c1 = self.conv1(complex2channels)
        g1 = self.GRU1(c1, hx[0])
        c2 = self.conv2(g1)
        g2 = self.GRU2(c2, hx[1])
        c3 = self.conv3(g2)

        # plt.imshow(torch.absolute(c2[0,0]+1j*c2[0,1]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        channels2complex = (c3[:,0]+1j*c3[:,1]).unsqueeze(1)

        return channels2complex, [g1, g2] # ??? detach() ???

class RIM_DC_noDetach(nn.Module):
    def __init__(self, n_steps=10):
        super().__init__()
        torch.cuda.set_device(1)
        # torch.set_default_dtype(torch.float64) ## this is really important since we do a lot of ffts. otherwise torch.zeros is float32 and we can't save complex128 into it!
        self.n_steps = n_steps
        
        # self.cRNN = ConvRNN_deepClean_noDetach()
        self.cRNN = ConvRNN_deepClean_noDetach_smallKernel()
        # self.type(torch.complex64)
        # torch.backends.cudnn.enabled = False
        

    def forward(self, x, hx=None, factor=1):
        ap = x[0]
        amp = ap[:,0]
        uv_cov = amp.unsqueeze(1).clone().detach()
        phase = ap[:,1]
        amp_rescaled = (10 ** (10 * amp) - 1) / 10 ** 10
        compl = amp_rescaled * torch.exp(1j * phase) #k measured
        

        data = compl.clone().detach().unsqueeze(1)
        compl_shift = torch.fft.fftshift(compl) # shift low freq to corner
        ifft = torch.fft.ifft2(compl_shift, norm="forward")
        eta = torch.fft.ifftshift(ifft).unsqueeze(1)*factor # shift low freq to center
        #calc beam
        # uv_cov[uv_cov!=0] = 1
        # beam = abs(torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(uv_cov))))
        
        # beam = beam/torch.max(torch.max(beam,2)[0],2)[0][:,:,None,None]
        # plt.imshow(beam[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        
        
        etas = []

        for i in range(self.n_steps):

            grad = gradFunc_putzky(eta, [data, x[1]])

            # if i == 0:
            #     eta = fft_conv(eta,beam)
            input = torch.cat((eta,grad), dim=1)
            delta, hx = self.cRNN(input, hx)
            eta = eta + delta
            # plt.imshow(abs(eta[0,0].cpu().detach().numpy())/(64**2), cmap='hot')
            # plt.colorbar()
            # plt.show()
            # plt.imshow(abs(grad[0,0].cpu().detach().numpy()), cmap='hot')
            # plt.colorbar()
            # plt.show()
            etas.append(eta)

        # plt.imshow(abs(fft_conv(eta,beam)[0,0].cpu().detach().numpy())/(64**2), cmap='hot')
        # plt.colorbar()
        # plt.show()
        # return [fft_conv(eta,beam)/eta.shape[2]**2 for eta in etas]
        return [eta/eta.shape[2]**2 for eta in etas]



class ConvRNN_deepClean_noDetach_smallKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, dilation=2, padding=2),
            nn.Tanh(),
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(4, 64, 1, stride=1, dilation=2, padding=0),
            nn.Tanh(),
        )
        self.conv1c = nn.Sequential(
            nn.Conv2d(4, 64, 5, stride=1, dilation=2, padding=4),
            nn.Tanh(),
        )
        self.GRU1 = ConvGRUCell(64*3, 64*3, 3)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1, dilation=2, padding=2),
            nn.Tanh(),
        )
        self.conv2b = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 1, stride=1, dilation=2, padding=0),
            nn.Tanh(),
        )
        self.conv2c = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, stride=1, dilation=2, padding=4),
            nn.Tanh(),
        )
        self.GRU2 = ConvGRUCell(64*3, 64*3, 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*3, 2, 3, stride=1, dilation=2, padding=2, bias=False)
        )
        # self.weight = nn.Parameter(torch.tensor([0.25]))

    def forward(self, x, hx=None):
        if not hx:
            hx = [None]*2

        complex2channels = torch.cat((x[:,0].real.unsqueeze(1),x[:,0].imag.unsqueeze(1),x[:,1].real.unsqueeze(1),x[:,1].imag.unsqueeze(1)), dim=1)
        # print(complex2channels.shape)
        # print(complex2channels.dtype)

        c1 = self.conv1(complex2channels)
        c1b = self.conv1b(complex2channels)
        c1c = self.conv1c(complex2channels)
        comb = torch.cat((c1,c1b,c1c),dim=1)
        g1 = self.GRU1(comb, hx[0])
        g1abc = torch.split(g1,64,dim=1)
        c2 = self.conv2(g1abc[0])
        c2b = self.conv2(g1abc[1])
        c2c = self.conv2(g1abc[2])
        comb2 = torch.cat((c2,c2b,c2c),dim=1)
        g2 = self.GRU2(comb2, hx[1])
        c3 = self.conv3(g2)

        # plt.imshow(torch.absolute(c2[0,0]+1j*c2[0,1]).cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()
        channels2complex = (c3[:,0]+1j*c3[:,1]).unsqueeze(1)

        return channels2complex, [g1, g2] # ??? detach() ???
