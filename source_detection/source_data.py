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


def detector_data(img_size, bundle_size, num_bundles,path):
    for t in tqdm(range(num_bundles)):
        with h5py.File(path+str(t)+'.h5', "w") as hf:
            all_images = []
            all_bboxes = []
            all_labels = []
            for r in range(bundle_size):
                grid = create_grid(img_size, 1)
                num_objects = np.random.randint(1,2)  #noise
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
                        rand = np.random.randint(0,4)
                        rand = 0  #noise
                        if rand == 0:
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([0])
                            bboxes[i] = bbox 
                            labels[i] = label
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
                num_objects = np.random.randint(150,200)
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
                        rand = np.random.randint(0,4)
                        if rand == 0:
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, False, True, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([0])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 1:
                            diffuse_limiter +=1
                            if diffuse_limiter > 5:
                                continue
                            print(diffuse_limiter)
                            g,c,s = create_gauss(grid[:, 0], 1, 1, False,img_size, True, True, True)
                            xmin = (c[0]-(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymin = (c[1]-(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            xmax = (c[0]+(3*np.sqrt(s[0])/2)).clip(0,img_size)/img_size
                            ymax = (c[1]+(3*np.sqrt(s[1])/2)).clip(0,img_size)/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([1])
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 2:
                            g,c,s = create_diamond(grid[:, 0], 1, 1, img_size,True, True)
                            xmin = (c[0]-2*s[0])/img_size
                            ymin = (c[1]-2*s[1])/img_size
                            xmax = (c[0]+2*s[0])/img_size
                            ymax = (c[1]+2*s[1])/img_size
                            bbox = np.array([xmin.clip(0,1), ymin.clip(0,1), xmax.clip(0,1), ymax.clip(0,1)], dtype = float)
                            label = np.array([2]) #nodiff
                            bboxes[i] = bbox 
                            labels[i] = label
                        elif rand == 3:
                            g,c = create_square(grid[:, 0],1, 1, img_size, True, True)
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


def feature_data(num_gauss, num_diff, num_diamond, num_square, img_size, num_files, path):
    for j in tqdm(range(num_files)):
        with h5py.File(path+str(j)+'.h5', "w") as hf:
            gauss_grid = create_grid(img_size, num_gauss)
            diff_grid = create_grid(img_size, num_diff)
            diamond_grid = create_grid(img_size, num_diamond)
            square_grid =  create_grid(img_size, num_square)
            gaussians = create_gauss(gauss_grid[:, 0], num_gauss, 1, False,img_size, False)
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
