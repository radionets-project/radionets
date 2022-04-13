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
