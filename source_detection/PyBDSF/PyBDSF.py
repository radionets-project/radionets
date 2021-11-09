import bdsf
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import h5py
import scipy
from scipy.spatial import distance
import torch

def image_eval(image): #Evaluate an image with PyBDSF
    image = np.flip(image, 0)
    img_save = image.reshape(1,1,50,50)
    g = fits.open('/net/big-tank/POOL/users/pblomenkamp/sdc1/dataset.fits')
    g[0].header
    g[0].data = img_save
    g[0].header['CRPIX1'] = 25
    g[0].header['CRPIX2'] = 25
    g[0].header['CRVAL3'] = 560000000
    g[0].header['BMAJ'] = 4.16666676756E-04 
    g[0].header['BMIN']=4.16666676756E-04 
    g[0].header['BPA'] = 0
    g.writeto('temp.fits',overwrite=True)
    
    img = bdsf.process_image('./temp.fits', rms_map = True)
    img.show_fit()
    img.export_image(img_format = 'fits', img_type = 'gaus_model')
    
    
    
def precisioneval(sdcset_path):
    #Evaluates an entire SKA Data Challenge Dataset. PyBDSF detections are made on every image and are evaluated. If the predicted source position is further away than 5% of the image resolution, the detection is considered a false positive.
    #Creates a temporary FITS file to since PyBDSF only accepts those.
    num_sources = []
    f = h5py.File(sdcset_path, "r")
    g = fits.open('/net/big-tank/POOL/users/pblomenkamp/sdc1/dataset.fits')
    sum_true_sources = 0
    sum_detected_sources  = 0
    false_pos = 0
    for image_number in range(len(f.keys())//3):
        img = np.array(f["x"+str(image_number)])
        true_labels = np.array(f["y"+str(image_number)])
        img_pos = np.array(f["z"+str(image_number)])
        img_size = img.shape[0]
        img_save = img.reshape(1,1,50,50)

        g[0].header
        g[0].data = img_save
        g[0].header['CRPIX1'] = 25
        g[0].header['CRPIX2'] = 25
        g[0].header['BPA'] = 0
        g.writeto('TEMP.fits',overwrite=True)

        img = bdsf.process_image('TEMP.fits',quiet=True)
        img.write_catalog(format='fits', catalog_type='gaul', clobber = True)
        l = fits.open('TEMP.pybdsm.gaul.fits')
        xcoords = l[1].data.field(12)
        ycoords = l[1].data.field(14)
        coords = np.vstack((xcoords,ycoords)).T

        coords = np.round(coords).astype('int')
        if np.array(true_labels).shape[0] == 0:
            false_pos += coords.shape[0]
            num_sources.append((0, coords.shape[0]))
            continue
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
                        false_pos += 1
                        continue
                if sort_dist[:,0][j] > img_size*0.05: #FP threshold
                        keep_mask[j] = False
                        false_pos += 1
        detected_sources = sort_dist[keep_mask].shape[0]
        true_sources = true_labels.shape[0]
        num_sources.append((true_sources, detected_sources))
        sum_detected_sources = sum_detected_sources + detected_sources
        sum_true_sources = sum_true_sources + true_sources  
    sum_sources = (sum_true_sources, sum_detected_sources)
    f.close()
    g.close()
    print('True Sources',sum_sources[0], 'True Positives',sum_sources[1], 'False Positives', false_pos, 'False Negatives',
          sum_sources[0]-sum_sources[1] )
    return num_sources,  sum_sources, false_pos
          #True number of sources, Number of True Positives, Number of false positives