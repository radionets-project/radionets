from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch.nn.functional as F
import torch
import h5py
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
            img = img/img.max()
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
    def create_labels(self,imgx,imgy, SNRfactor, mean, all_coords):
        coords = []
        x, y, flux = all_coords
        print(mean)
        for i in range(x.shape[1]):
            if flux[1][i] > 70*mean:
                if imgx < x[1][i] < imgx+self.img_size:
                    #print(x[0][i])
                    woy = np.where(x[0][i] == y[0])
                    if imgy < y[1][woy[0].item()] < imgy+self.img_size:
                        c = np.array([x[1][i]-imgx,y[1][woy[0].item()]-imgy])
                        c = np.round(c).astype('int')
                        coords.append(c)
        return coords
    def forward(self, SNRfactor, save_path):
        all_images = []
        all_labels = []
        all_pos = []
        f = fits.open(self.img_path)
        img = f[0].data
        f.close()
        all_coords = self.get_coords()
        mean = img.mean()  #NEEDS PROPER CALCULATION
        with h5py.File(save_path+'.h5', "w") as hf:
            for i in range(self.num_images):
                x = np.random.randint(16350 ,19900)
                y = np.random.randint(16700,19950)
                pos = (x,y) 
                image = self.create_image(x,y,img, self.img_size)
                labels = self.create_labels(x,y, SNRfactor, mean, all_coords)
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

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,8))
        ax1.imshow(img,cmap = 'hot')
        for i in range(len(true_labels)):
            ax1.scatter(true_labels[i][0], true_labels[i][1], color='b', s=20)

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
            ax2.add_patch(trurect)
        ax2.imshow(img, cmap = 'hot')
    
    def precisioneval(self):
        
        num_sources = []
        f = h5py.File(self.sdcset_path, "r")
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
            
            coords = np.round(coords).astype('int')
            
            dist = scipy.spatial.distance.cdist(coords, np.array(true_labels))
            
            dist = torch.tensor(dist)
            
            sort_dist, sort_ind = dist.sort(dim = 1, descending = False)
            
            sort_dist = sort_dist.detach().numpy()
            sort_ind = sort_ind.detach().numpy()
            keep_mask = [True]*sort_dist.shape[0]
            for i in range(sort_dist.shape[0]):
                for j in range(sort_dist.shape[0]):
                    if keep_mask[j] == False:
                        continue
                    if sort_ind[:,0][j] == sort_ind[:,0][i]:
                        if sort_dist[:,0][j] > sort_dist[:,0][i]:
                            keep_mask[j] = False
                            continue
                    if sort_dist[:,0][j] > img_size*0.05:
                            keep_mask[j] = False
            detected_sources = sort_dist[keep_mask].shape[0]
            true_sources = true_labels.shape[0]
            num_sources.append((true_sources, detected_sources))
        f.close()
        return num_sources
