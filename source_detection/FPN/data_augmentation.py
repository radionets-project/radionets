#Same as the other one
import matplotlib.pyplot as plt
import numpy as np
from radionets.evaluation.utils import  load_pretrained_model, eval_model
import FPNtrain
from radionets.dl_framework.data import get_bundles
import torch
from tqdm import tqdm
from FPNeval import box_coord
import matplotlib.patches as patches
import random
import cv2
from matplotlib.pyplot import figure
import h5py
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel


class_labels = ('pointlike gaussian', 'diffuse gaussian', 'diamond', 'square', 'background')
color_map = ('r', 'g', 'w', 'y','brown')

class random_shear(object):
    def __init__(self, shear_factor = 0.5):
        
        self.shear_factor  = shear_factor
        self.shear_factor = (-self.shear_factor, self.shear_factor)
        shear_factor = random.uniform(*self.shear_factor)
        self.horizontal_flip
    def __call__(self, img, bboxes):
        shear_factor = random.uniform(*self.shear_factor)
        bboxes = bboxes * img.shape[1]
        w,h = img.shape[1], img.shape[0]

        if shear_factor < 0:
            img, bboxes = self.horizontal_flip(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

        nW =  img.shape[1] + abs(shear_factor*img.shape[0])

        bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 


        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        if shear_factor < 0:
            img, bboxes = self.horizontal_flip(img, bboxes)
    
        img = cv2.resize(img, (w,h))

        scale_factor_x = nW / w

        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 

        bboxes = bboxes / img.shape[1]
        return img, bboxes
    def horizontal_flip(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        img =  img[::-1,:]
        bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]])
        
        box_w = abs(bboxes[:,0] - bboxes[:,2])
         
        bboxes[:,0] -= box_w
        bboxes[:,2] += box_w
        
        return img, bboxes


def gaussian_noise(image, strength = 0.05):
    pixel = image.shape[1]
    bundle_size = 1
    x = np.linspace(0, pixel - 1, num=pixel)
    y = np.linspace(0, pixel - 1, num=pixel)
    X, Y = np.meshgrid(x, y)
    grid = np.array([np.random.normal(0,1,X.shape) *strength, X, Y])
    grid = np.repeat(
        grid[None, :, :, :],
        bundle_size,
        axis=0,
    )
    k = np.random.normal(0,1,X.shape) * image.max()/10
    return image+grid[0][0]

def psf_noise(image, psf_path = '/net/big-tank/POOL/projects/ska/0836-BAND8_POSTAIPS.UVFITS'):
    f = fits.open(psf_path)
    u = f[0].data['UU--']
    v = f[0].data['VV--']
    U = np.append(u,-u)
    V = np.append(v,-v)
    u[(u<-0.002)&(u>0.002)] = 0
    v[(v<-0.002)&(v>0.002)] = 0
    
    gaussian2dkernel = Gaussian2DKernel(x_size = 2999, y_size = 2999, x_stddev = 75, y_stddev = 75)
    
    gaussian2dkernel = gaussian2dkernel.array/gaussian2dkernel.array.max()
    uv_hist, _, _ = np.histogram2d(U,V ,bins=3000)
    uv_hist[uv_hist >0] = 1
    psf = np.abs(np.fft.fftshift(np.fft.fft2(uv_hist)))
    psfnorm = psf/psf.max() 
    psfcut = psfnorm[0:2999,0:2999]
    psfcut = psfcut / psfcut.sum()
    psfcut = psfcut *gaussian2dkernel
    psfcut[psfcut < 1e-12] = 1e-12
    noisy_image = convolve_fft(image, psfcut, normalize_kernel = False, boundary = 'wrap')
    
    return noisy_image
