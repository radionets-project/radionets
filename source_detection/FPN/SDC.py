from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch.nn.functional as F
import torch
import h5py
#import bdsf
from tqdm import tqdm
import matplotlib.patches as patches
import numpy as np
from  FPNeval import detect_sources, box_coord,image_detection, open_bundle_pack, annotate
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
            if imgx < x[1][i] < imgx+self.img_size:
                #print(x[0][i])
                woy = np.where(x[0][i] == y[0])
                if imgy < y[1][woy[0].item()] < imgy+self.img_size:
                    c = np.array([x[1][i]-imgx,y[1][woy[0].item()]-imgy])
                    c = np.round(c).astype('int')
                    print(image[int(x[1][i]-imgx),int(y[1][woy[0].item()]-imgy)])
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
                RMSfactor = self.RMS(image, self.img_size)
                labels = self.create_labels(x,y, RMSfactor, mean, all_coords, image)
                all_images.append(image)
                all_labels.append(labels)
                all_pos.append((x,y))
                hf.create_dataset('x'+str(i), data=image)
                hf.create_dataset('y'+str(i), data=labels)
                hf.create_dataset('z'+str(i), data=pos)
        hf.close()    


class sdceval():
    
    def __init__(self,checkpoint, sdcset_path):
        
        self.checkpoint = checkpoint
        
        self.sdcset_path = sdcset_path
    
    def boxtoxy(self,boxes, img_size = 50):
        coords = np.array([boxes[:,2]-(boxes[:,2]-boxes[:,0])/2, boxes[:,3]-(boxes[:,3]-boxes[:,1])/2]).T 
        
        return coords
    def image_comp(self,image_number):
    
        f = h5py.File(self.sdcset_path, "r")
        img = np.array(f["x"+str(image_number)])
        true_labels = np.array(f["y"+str(image_number)])
        img_pos = np.array(f["z"+str(image_number)])
        f.close()

        fig, axs = plt.subplots(1,2,figsize=(12,8))
        axs[0].imshow(img,cmap = 'gist_heat')
        for i in range(len(true_labels)):
            axs[0].scatter(true_labels[i][0], true_labels[i][1], color='b', s=20)

        dimg = torch.FloatTensor(img)
        dimg = dimg.unsqueeze(0)
        dimg = dimg.unsqueeze(0)
        uimg = F.interpolate(dimg, size = (300,300) , mode = 'bilinear')
        uimg = uimg[0][0].detach().numpy()

        b,l,s = image_detection(self.checkpoint,uimg)

        bbox = b[0]
        labels = l[0].cpu().detach().numpy()
        img_size = img.shape[0]

        for j in range(bbox.shape[0]):
            true_label = labels[j]
            trux, truy, truw, truh = box_coord(bbox[j],img_size)
            trurect = patches.Rectangle((trux, truy), truw, truh, linewidth=1, edgecolor='w', facecolor='none')
            #ax2.text(trux,(truy+truh-7),true_label, color = 'k',fontsize=8,backgroundcolor = color)
            axs[1].add_patch(trurect)
        im = axs[1].imshow(img, cmap = 'gist_heat')
        
        for ax in axs.flat:
            ax.set(xlabel='Pixels', ylabel='Pixels')
            ax.label_outer()
        fig.tight_layout()
        cbar_ax = fig.add_axes([0.99, 0.155, 0.03, 0.69])
        fig.colorbar(im, cax=cbar_ax,label = 'Normalized Intensity')
        plt.rcParams.update({'font.size': 20})
        plt.savefig('./sdc_example.pdf', bbox_inches = 'tight')
    
    def precisioneval(self):
        
        num_sources = []
        f = h5py.File(self.sdcset_path, "r")
        sum_true_sources = 0
        sum_detected_sources  = 0
        false_pos = 0
        for image_number in range(len(f.keys())//3):
            
            img = np.array(f["x"+str(image_number)])
            true_labels = np.array(f["y"+str(image_number)])
            img_pos = np.array(f["z"+str(image_number)])
            img_size = img.shape[0]

            dimg = torch.FloatTensor(img)
            dimg = dimg.unsqueeze(0)
            dimg = dimg.unsqueeze(0)
            uimg = F.interpolate(dimg, size = (300,300) , mode = 'bilinear')
            uimg = uimg[0][0].detach().numpy()

            b,l,s = image_detection(self.checkpoint,uimg)
            coords = self.boxtoxy(b[0].cpu().detach().numpy()*img_size)
            if np.array(true_labels).shape[0] == 0:
                false_pos += coords.shape[0]
                num_sources.append((0, 0))
               # print('ay caramba',coords.shape[0])
                continue
            coords = np.round(coords).astype('int')
            
            dist = scipy.spatial.distance.cdist(coords, np.array(true_labels))
            
            dist = torch.tensor(dist)
            
            sort_dist, sort_ind = dist.sort(dim = 1, descending = False)
            
            sort_dist = sort_dist.detach().numpy()
            sort_ind = sort_ind.detach().numpy()
            keep_mask = [True]*sort_dist.shape[0]
            for i in range(sort_dist.shape[0]):
                for j in range(sort_dist.shape[0]):
                    if keep_mask[j] == False: #If already thrown out: Continue
                        continue
                    if sort_ind[:,0][j] == sort_ind[:,0][i]: # If i and j are closest to the same object
                        if sort_dist[:,0][j] > sort_dist[:,0][i]: #keep only the one that is closest
                            keep_mask[j] = False
                            false_pos += 1
                            continue
                    if sort_dist[:,0][j] > img_size*0.05:
                            keep_mask[j] = False
                            false_pos += 1
            detected_sources = sort_dist[keep_mask].shape[0]
            true_sources = true_labels.shape[0]
            sum_detected_sources = sum_detected_sources + detected_sources
            sum_true_sources = sum_true_sources + true_sources  
            num_sources.append((true_sources, detected_sources))
        sum_sources = (sum_true_sources, sum_detected_sources)
        f.close()
        return sum_sources, false_pos
