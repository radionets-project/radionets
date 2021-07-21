from radionets.simulations.gaussians import create_grid, create_gauss, create_diamond, create_square
from radionets.dl_framework.data import save_fft_pair, open_fft_pair
from scipy import ndimage
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import h5py

from torchvision import transforms
#om utils import *
from PIL import Image, ImageDraw, ImageFont
from data_augmentation import random_shear, gaussian_noise, psf_noise, rough_gaussian_noise
import oldgaussian


def detector_data(img_size, bundle_size, num_bundles,path):
    for t in tqdm(range(num_bundles)):
        with h5py.File(path+str(t)+'.h5', "w") as hf:
            all_images = []
            all_bboxes = []
            all_labels = []
            for r in range(bundle_size):
                grid = create_grid(img_size, 1)
                num_objects = np.random.randint(1,5)  #noise
                bboxes = np.zeros((num_objects,4))
                labels = np.zeros((num_objects,1))
                if num_objects == 0:
                    bboxes = np.zeros((1,4))
                    labels = np.zeros((1,1))
                    g = grid[:,0]
                    bboxes[0] = np.array([0, 0, 1, 1], dtype = float)
                    labels[0] = np.array([4])
                else:
                    for i in range(num_objects): 
                        rand = 0#np.random.randint(0,1)
                        if rand == 0:
                            #g,c,s,theta = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True, spherical = True)
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([0])
                            bboxes[i] = bbox 
                            labels[i] = label
                            
                            #xmin = (c[0]-(4*np.sqrt(s[0])/2))/img_size
                            #ymin = (c[1]-(4*np.sqrt(s[1])/2))/img_size
                            #xmax = (c[0]+(4*np.sqrt(s[0])/2))/img_size
                            #ymax = (c[1]+(4*np.sqrt(s[1])/2))/img_size
                            #w = xmax-xmin
                            #h = ymax-ymin
                            #wn = np.sqrt((w*np.cos(theta))**2+(h*np.sin(theta))**2)
                            #hn = np.sqrt((h*np.cos(theta))**2+(w*np.sin(theta))**2)
                            #xmin = c[0]/img_size-wn/2
                            #xmax = c[0]/img_size+wn/2
                            #ymin = c[1]/img_size-hn/2
                            #ymax = c[1]/img_size+hn/2
                            #bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            #label = np.array([0])
                            #bboxes[i] = bbox 
                            #labels[i] = label
                        elif rand == 1:
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, True, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([1])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 2:
                            g,c,s = create_diamond(grid[:, 0], 1, 1, img_size,True)
                            xmin = (c[0]-2*s[0])/img_size
                            ymin = (c[1]-2*s[1])/img_size
                            xmax = (c[0]+2*s[0])/img_size
                            ymax = (c[1]+2*s[1])/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([2]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 3:
                            g,c = create_square(grid[:, 0],1, 1, img_size, True)
                            xmin = (c[0]-(img_size/50+1))/img_size
                            ymin = (c[1]-(img_size/50+1))/img_size
                            xmax = (c[0]+(img_size/50+1))/img_size
                            ymax = (c[1]+(img_size/50+1))/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([3]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
                g = g/g.max()
                hf.create_dataset('x'+str(r), data=g)
                hf.create_dataset('y'+str(r), data=bboxes)
                hf.create_dataset('z'+str(r), data=labels)
        hf.close()

def create_mosaic(img_size, bundle_size, num_bundles,path):
    for t in range(num_bundles):
        with h5py.File(path+str(t)+'.h5', "w") as hf:
            all_images = []
            all_bboxes = []
            all_labels = []
            for r in range(bundle_size):
                grid = create_grid(img_size*10, 1)
                num_objects = 20#np.random.randint(150,200)
                bboxes = np.zeros((num_objects,4))
                labels = np.zeros((num_objects,1))
                diffuse_limiter = 0
                if num_objects == 0:
                    bboxes = np.zeros((1,4))
                    labels = np.zeros((1,1))
                    g = grid[:,0]
                    bboxes[0] = np.array([0, 0, 1, 1], dtype = float)
                    labels[0] = np.array([4])
                else:
                    for i in range(num_objects): 
                        rand = 0#np.random.randint(0,2)
                        if rand == 0:
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True, 10, spherical = True)
                            print(c)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size*10)/(img_size*10)
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size*10)/(img_size*10)
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size*10)/(img_size*10)
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size*10)/(img_size*10)
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([0])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 1:
                            diffuse_limiter +=1
                            if diffuse_limiter > 0:
                                continue
                            print(diffuse_limiter)
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, True, True, 10)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([1])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 2:
                            g,c,s = create_diamond(grid[:, 0], 1, 1, img_size,True, 10)
                            xmin = (c[0]-2*s[0])/img_size
                            ymin = (c[1]-2*s[1])/img_size
                            xmax = (c[0]+2*s[0])/img_size
                            ymax = (c[1]+2*s[1])/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([2]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 3:
                            g,c = create_square(grid[:, 0],1, 1, img_size, True, 10)
                            xmin = (c[0]-(img_size/50+1))/img_size
                            ymin = (c[1]-(img_size/50+1))/img_size
                            xmax = (c[0]+(img_size/50+1))/img_size
                            ymax = (c[1]+(img_size/50+1))/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([3]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
                g = g/g.max()
                hf.create_dataset('x'+str(r), data=g)
                hf.create_dataset('y'+str(r), data=bboxes)
                hf.create_dataset('z'+str(r), data=labels)
        hf.close()



def feature_data(num_gauss, num_diff, num_diamond, num_square, img_size, num_files, path):
    for j in tqdm(range(num_files)):
        with h5py.File(path+str(j)+'.h5', "w") as hf:
            gauss_grid = create_grid(img_size, num_gauss)
            diff_grid = create_grid(img_size, num_diff)
            diamond_grid = create_grid(img_size, num_diamond)
            square_grid =  create_grid(img_size, num_square)
            gaussians = create_gauss(gauss_grid[:, 0], num_gauss, 1, False,img_size, False, spherical = True)
            y_gauss = np.array([0]*len(gaussians))
            diff = create_gauss(diff_grid[:, 0], num_diff, 1, False,img_size, True)
            y_diff = np.array([1]*len(diff))
            diamonds = create_diamond(diamond_grid[:, 0], num_diamond, 1, img_size)
            y_diamond = np.array([2]*len(diamonds))
            squares = create_square(square_grid[:, 0],num_square, 1, img_size)
            y_square = np.array([3]*len(squares))
            arr = np.concatenate((gaussians, diff, diamonds, squares), axis=0)
            keys = np.concatenate((y_gauss, y_diff, y_diamond, y_square), axis=0)
            shuff = np.random.permutation(len(arr))
            hf.create_dataset('x', data=arr[shuff])
            hf.create_dataset('y', data=keys[shuff])
        hf.close()


def noisy_data(img_size, mosaic_scale, bundle_size, num_bundles,path): #TEMP CHANGES: NO NOISE!
    for t in tqdm(range(num_bundles)):
        with h5py.File(path+str(t)+'.h5', "w") as hf:
            all_images = []
            all_bboxes = []
            all_labels = []
            grid = create_grid(img_size*mosaic_scale, 1)
            num_objects = 180
            bboxes = np.zeros((num_objects,5))
            diffuse_limiter = 0
            bundle = 0
            if num_objects == 0:
                bboxes = np.zeros((1,5))
                labels = np.zeros((1,1))
                g = grid[:,0]
                bboxes[0] = np.array([0, 0, 1, 1], dtype = float)
                labels[0] = np.array([4])
            else:
                for i in range(num_objects): 
                    #rand = np.random.randint(0,2)
                    rand = 0
                    #if diffuse_limiter >= 3:
                    #        rand = 0
                    if rand == 0:
                        g,c,s,theta = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True,mosaic_scale, spherical = False)
                        xmin = (c[0]-(4*np.sqrt(s[0])/2))/(img_size*mosaic_scale)
                        ymin = (c[1]-(4*np.sqrt(s[1])/2))/(img_size*mosaic_scale)
                        xmax = (c[0]+(4*np.sqrt(s[0])/2))/(img_size*mosaic_scale)
                        ymax = (c[1]+(4*np.sqrt(s[1])/2))/(img_size*mosaic_scale)
                        w = xmax-xmin
                        h = ymax-ymin
                        wn = np.sqrt((w*np.cos(theta))**2+(h*np.sin(theta))**2)
                        hn = np.sqrt((h*np.cos(theta))**2+(w*np.sin(theta))**2)
                        xmin = c[0]/(img_size*mosaic_scale)-wn/2
                        xmax = c[0]/(img_size*mosaic_scale)+wn/2
                        ymin = c[1]/(img_size*mosaic_scale)-hn/2
                        ymax = c[1]/(img_size*mosaic_scale)+hn/2
                        bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1),0], dtype = float)
                        bboxes[i] = bbox 
                    elif rand == 1:
                        diffuse_limiter +=1
                        g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, True, True, mosaic_scale)
                        xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1),1], dtype = float)
                        bboxes[i] = bbox 
                image = g[0]
                image = gaussian_noise(image)
                image = psf_noise(image)
                image = image/image.max()
                while bundle < bundle_size:
                    x_crop = np.random.randint(0,((img_size*mosaic_scale))-img_size)
                    y_crop = np.random.randint(0,((img_size*mosaic_scale))-img_size)
                   
                    crop_image = image[y_crop:y_crop+img_size,x_crop:x_crop+img_size]
                    boxes = np.array(bboxes)
                    boxes[:,:4] = boxes[:,:4]*(img_size*mosaic_scale)

                    box_crop = boxes[boxes[:,0] < x_crop+img_size-(boxes[:,2]-boxes[:,0])/2]
                    box_crop = box_crop[box_crop[:,2] > x_crop+(box_crop[:,2]-box_crop[:,0])/2]
                    box_crop = box_crop[box_crop[:,1] < y_crop+img_size-(box_crop[:,3]-box_crop[:,1])/2]
                    box_crop = box_crop[box_crop[:,3] > y_crop+(box_crop[:,3]-box_crop[:,1])/2]
                    
                    if len(box_crop) == 0:
                        continue
                    box_crop[:,0] = (box_crop[:,0]-x_crop)/img_size
                    box_crop[:,1] = (box_crop[:,1]-y_crop)/img_size
                    box_crop[:,2] = (box_crop[:,2]-x_crop)/img_size
                    box_crop[:,3] = (box_crop[:,3]-y_crop)/img_size
                    labels = np.zeros((box_crop.shape[0],1))
                    for g in range(box_crop.shape[0]):
                        labels[g] = box_crop[g,4].astype(int)
                    final_image = np.expand_dims(crop_image, axis=0)
                    #final_image = rough_gaussian_noise(final_image)
                    hf.create_dataset('x'+str(bundle), data=final_image)
                    hf.create_dataset('y'+str(bundle), data=box_crop[:,:4])
                    hf.create_dataset('z'+str(bundle), data=labels)
                    bundle += 1
        hf.close()


def noisy_feature_data(num_gauss, num_diff, num_diamond, num_square, img_size, num_files, path):
    for j in tqdm(range(num_files)):
        with h5py.File(path+str(j)+'.h5', "w") as hf:
            gauss_grid = create_grid(img_size, num_gauss)
            diff_grid = create_grid(img_size, num_diff)
            diamond_grid = create_grid(img_size, num_diamond)
            square_grid =  create_grid(img_size, num_square)
            gaussians = create_gauss(gauss_grid[:, 0], num_gauss, 1, False,img_size, False, spherical = False)
            y_gauss = np.array([0]*len(gaussians))
            diff = create_gauss(diff_grid[:, 0], num_diff, 1, False,img_size, True)
            y_diff = np.array([1]*len(diff))
            diamonds = create_diamond(diamond_grid[:, 0], num_diamond, 1, img_size)
            y_diamond = np.array([2]*len(diamonds))
            squares = create_square(square_grid[:, 0],num_square, 1, img_size)
            y_square = np.array([3]*len(squares))
            arr = np.concatenate((gaussians, diff, diamonds, squares), axis=0)
            num_superimages = arr.shape[0]//100
            for k in range(num_superimages):
                super_grid = create_grid(img_size*10, 1)
                for i in range(10):
                    for o in range(10):
                         super_grid[0,0][o*img_size:img_size+o*img_size,i*img_size:img_size+i*img_size] = arr[100*k+10*i+o]
            
                #super_grid[0,0] = gaussian_noise(super_grid[0,0])  
                #super_grid[0,0] = psf_noise(super_grid[0,0])
                for p in range(10):
                    for a in range(10):
                         arr[10*p+a+100*k] = super_grid[0,0][a*img_size:img_size+a*img_size,p*img_size:img_size+p*img_size] 
            keys = np.concatenate((y_gauss, y_diff, y_diamond, y_square), axis=0)
            shuff = np.random.permutation(len(arr))
            
            hf.create_dataset('x', data=arr[shuff])
            hf.create_dataset('y', data=keys[shuff])
        hf.close()


def NoisyTBCData(img_size, mosaic_scale, bundle_size, num_bundles,path):
    for t in tqdm(range(num_bundles)):
        with h5py.File(path+str(t)+'.h5', "w") as hf:
            images = []
            truth = []
            grid = create_grid(img_size*mosaic_scale, 1)
            num_objects = 90
            bboxes = np.zeros((num_objects,5))
            diffuse_limiter = 0
            bundle = 0
            if num_objects == 0:
                bboxes = np.zeros((1,5))
                labels = np.zeros((1,1))
                g = grid[:,0]
                bboxes[0] = np.array([0, 0, 1, 1], dtype = float)
                labels[0] = np.array([4])
            else:
                for i in range(num_objects): 
                    rand = np.random.randint(0,2)
                    rand = 0
                    if diffuse_limiter >= 1:
                            rand = 0
                    if rand == 0:
                        g,c,s,theta = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True,mosaic_scale, spherical = False)
                           
                            
                       
                    elif rand == 1:
                        diffuse_limiter +=1
                        g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, True, True, mosaic_scale)
                        xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,(img_size*mosaic_scale))/(img_size*mosaic_scale)
                        bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1),1], dtype = float)
                        bboxes[i] = bbox 
                image = g[0]
                image = gaussian_noise(image)
                image = psf_noise(image)
                while bundle < bundle_size:
                    x_crop = np.random.randint(0,((img_size*mosaic_scale))-img_size)
                    y_crop = np.random.randint(0,((img_size*mosaic_scale))-img_size)
                   
                    crop_image = image[y_crop:y_crop+img_size,x_crop:x_crop+img_size]
                    boxes = np.array(bboxes)
                    boxes[:,:4] = boxes[:,:4]*(img_size*mosaic_scale)

                    box_crop = boxes[boxes[:,0] < x_crop+img_size-(boxes[:,2]-boxes[:,0])/2]
                    box_crop = box_crop[box_crop[:,2] > x_crop+(box_crop[:,2]-box_crop[:,0])/2]
                    box_crop = box_crop[box_crop[:,1] < y_crop+img_size-(box_crop[:,3]-box_crop[:,1])/2]
                    box_crop = box_crop[box_crop[:,3] > y_crop+(box_crop[:,3]-box_crop[:,1])/2]
                    
                    if len(box_crop) == 0:
                        continue
                    box_crop[:,0] = (box_crop[:,0]-x_crop)/img_size
                    box_crop[:,1] = (box_crop[:,1]-y_crop)/img_size
                    box_crop[:,2] = (box_crop[:,2]-x_crop)/img_size
                    box_crop[:,3] = (box_crop[:,3]-y_crop)/img_size
                    labels = np.zeros((box_crop.shape[0],1))
                    for g in range(box_crop.shape[0]):
                        labels[g] = box_crop[g,4].astype(int)
                    final_image = np.expand_dims(crop_image, axis=0)
                    hf.create_dataset('x'+str(bundle), data=final_image)
                    hf.create_dataset('y'+str(bundle), data=box_crop[:,:4])
                    hf.create_dataset('z'+str(bundle), data=labels)
                    bundle += 1
        hf.close()


def old_detector_data(img_size, bundle_size, num_bundles,path):
    for t in tqdm(range(num_bundles)):
        with h5py.File(path+str(t)+'.h5', "w") as hf:
            all_images = []
            all_bboxes = []
            all_labels = []
            for r in range(bundle_size):
                grid = create_grid(img_size, 1)
                num_objects = np.random.randint(1,5)  #noise
                bboxes = np.zeros((num_objects,4))
                labels = np.zeros((num_objects,1))
                if num_objects == 0:
                    bboxes = np.zeros((1,4))
                    labels = np.zeros((1,1))
                    g = grid[:,0]
                    bboxes[0] = np.array([0, 0, 1, 1], dtype = float)
                    labels[0] = np.array([4])
                else:
                    for i in range(num_objects): 
                        rand = np.random.randint(0,2)
                        rand = 0  #noise
                        if rand == 0:
                            g,c,s = oldgaussian.create_gauss(grid[:, 0], 1, 1, False,img_size, False, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([0])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 1:
                            g,c,s = oldgaussian.create_gauss(grid[:, 0], 1, 1, False,img_size, True, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([1])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 2:
                            g,c,s = create_diamond(grid[:, 0], 1, 1, img_size,True)
                            xmin = (c[0]-2*s[0])/img_size
                            ymin = (c[1]-2*s[1])/img_size
                            xmax = (c[0]+2*s[0])/img_size
                            ymax = (c[1]+2*s[1])/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([2]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 3:
                            g,c = create_square(grid[:, 0],1, 1, img_size, True)
                            xmin = (c[0]-(img_size/50+1))/img_size
                            ymin = (c[1]-(img_size/50+1))/img_size
                            xmax = (c[0]+(img_size/50+1))/img_size
                            ymax = (c[1]+(img_size/50+1))/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([3]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
               
                hf.create_dataset('x'+str(r), data=g)
                hf.create_dataset('y'+str(r), data=bboxes)
                hf.create_dataset('z'+str(r), data=labels)
        hf.close()


def old_feature_data(num_gauss, num_diff, num_diamond, num_square, img_size, num_files, path):
    for j in tqdm(range(num_files)):
        with h5py.File(path+str(j)+'.h5', "w") as hf:
            gauss_grid = create_grid(img_size, num_gauss)
            diff_grid = create_grid(img_size, num_diff)
            diamond_grid = create_grid(img_size, num_diamond)
            square_grid =  create_grid(img_size, num_square)
            gaussians = oldgaussian.create_gauss(gauss_grid[:, 0], num_gauss, 1, False,img_size, False)
            y_gauss = np.array([0]*len(gaussians))
            diff = oldgaussian.create_gauss(diff_grid[:, 0], num_diff, 1, False,img_size, True)
            y_diff = np.array([1]*len(diff))
            diamonds = create_diamond(diamond_grid[:, 0], num_diamond, 1, img_size)
            y_diamond = np.array([2]*len(diamonds))
            squares = create_square(square_grid[:, 0],num_square, 1, img_size)
            y_square = np.array([3]*len(squares))
            arr = np.concatenate((gaussians, diff, diamonds, squares), axis=0)
            keys = np.concatenate((y_gauss, y_diff, y_diamond, y_square), axis=0)
            shuff = np.random.permutation(len(arr))
            hf.create_dataset('x', data=arr[shuff])
            hf.create_dataset('y', data=keys[shuff])
        hf.close()
