from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch.nn.functional as F
import torch
import h5py
import bdsf
from tqdm import tqdm
import matplotlib.patches as patches
import numpy as np
import scipy
from scipy.spatial import distance
#from data_augmentation import gaussian_noise, psf_noise 


class sdc_dataset():
    
    def __init__(self, num_images, label_path, img_path, img_size):
        
        self.num_images = num_images
        
        self.label_path = label_path
        
        self.img_path = img_path
        
        self.img_size = img_size
        
    def create_image(self, x, y, image, img_size):
            img = image[0][0][y:y+img_size,x:x+img_size].astype('float64')
            img = img#/img.max()
            return img
    
    def get_coords(self):
        classes = np.loadtxt(self.label_path, skiprows = 18, usecols = (11) )
        cx = np.loadtxt(self.label_path, skiprows = 18, usecols = (13) )
        cy = np.loadtxt(self.label_path, skiprows = 18, usecols = (14) )
        use = np.loadtxt(self.label_path, dtype = 'bool', skiprows = 18, usecols = (12) )
        flux = np.loadtxt(self.label_path, skiprows = 18, usecols = (5) )
        cx = cx[use]
        cy = cy[use]
        flux = flux[use]
        oid = np.linspace(0,cx.shape[0], cx.shape[0], dtype = 'int32')
        cy = np.vstack((oid,cy))
        cx = np.vstack((oid,cx))
        flux = np.vstack((oid,flux))
        
        all_coords = cx, cy, flux
        return all_coords
    
    def RMS(self, image, img_size): 
        img_save = image.reshape(1,1,img_size,img_size)
        g = fits.open('/net/big-tank/POOL/users/pblomenkamp/sdc1/dataset.fits')
        g[0].header
        g[0].data = img_save
        g[0].header['CRPIX1'] = img_size//2
        g[0].header['CRPIX2'] = img_size//2
        g[0].header['CRVAL3'] = 560000000
        g[0].header['BMAJ'] = 4.16666676756E-04 
        g[0].header['BMIN']=4.16666676756E-04 
        g[0].header['BPA'] = 0
        g.writeto('temp.fits',overwrite=True)

        img = bdsf.process_image('./temp.fits', rms_map = True,quiet=True)
        return img.rms_arr[0][0]
    def create_labels(self,imgx,imgy, RMSfactor, mean, all_coords, image):
        coords = []
        x, y, flux = all_coords
        for i in range(x.shape[1]):
            if imgx < np.round(x[1][i]).astype('int') < imgx+self.img_size:
                #print(x[0][i])
                woy = np.where(x[0][i] == y[0])
                if imgy < np.round(y[1][woy[0].item()]).astype('int') < imgy+self.img_size:
                    c = np.array([x[1][i]-imgx,y[1][woy[0].item()]-imgy])
                    c = np.round(c).astype('int')
                    if  image[int(y[1][woy[0].item()]-imgy),int(x[1][i]-imgx)] > RMSfactor:
                        coords.append(c)
        return coords
    def forward(self, save_path):
        all_images = []
        all_labels = []
        all_pos = []
        f = fits.open(self.img_path)
        img = f[0].data
        f.close()
        all_coords = self.get_coords()
        mean = img.mean()  #NEEDS PROPER CALCULATION
        with h5py.File(save_path+'.h5', "w") as hf:
            for i in tqdm(range(self.num_images)):
                x = np.random.randint(16350 ,19900)
                y = np.random.randint(16700,19950)
                pos = (x,y) 
                image = self.create_image(x,y,img, self.img_size)
                RMSfactor = 2*self.RMS(image, self.img_size) #SNR
                #print(RMSfactor)
                labels = self.create_labels(x,y, RMSfactor, mean, all_coords, image)
                image = image/image.max()
                all_images.append(image)
                all_labels.append(labels)
                all_pos.append((x,y))
                hf.create_dataset('x'+str(i), data=image)
                hf.create_dataset('y'+str(i), data=labels)
                hf.create_dataset('z'+str(i), data=pos)
        hf.close()    


