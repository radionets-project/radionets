import os
from tqdm import tqdm
from numpy import savez_compressed
from radionets.simulations.utils import (
    get_fft_bundle_paths,
    get_real_bundle_paths,
    prepare_fft_images,
    interpol,
)
from radionets.dl_framework.data import (
    open_fft_bundle,
    save_fft_pair,
    save_fft_pair_list,
)
from radionets.simulations.uv_simulations import sample_freqs
import h5py
import numpy as np
from astropy.io import fits
from PIL import Image
import cv2
import radionets.dl_framework.data as dt
import re
from natsort import natsorted
from PIL import Image


def process_data(
    data_path,
    # amp_phase,
    # real_imag,
    # fourier,
    # compressed,
    # interpolation,
    # specific_mask,
    # antenna_config,
    # lon=None,
    # lat=None,
    # steps=None,
):
    
    print(f"\n Loading VLBI data set.\n")
    bundles = dt.get_bundles('/net/big-tank/POOL/users/sfroese/vipy/eht/m87/')
    freq = 227297*10**6 # hard code #eht 227297
    bundles_target = dt.get_bundles(bundles[0])
    bundles_input = dt.get_bundles(bundles[1])
    bundle_paths_target = natsorted(bundles_target)
    bundle_paths_input = natsorted(bundles_input)
    size = len(bundle_paths_target)
    img = np.zeros((size,256,256))
    samps = np.zeros((size,4,21036)) # hard code
    for i in tqdm(range(size)):
        sampled = bundle_paths_input[i]
        target = bundle_paths_target[i]

        img[i] = np.asarray(Image.open(str(target)))
        # img[i] = img[i]/np.sum(img[i])
    
        with fits.open(sampled) as hdul:
            data = hdul[0].data
            cmplx = data['DATA'] 
            x = cmplx[...,0,0]
            y = cmplx[...,0,1]
            w = cmplx[...,0,2]
            x = np.squeeze(x)
            y = np.squeeze(y)
            w = np.squeeze(w)
            ap = np.sqrt(x**2+y**2)
            ph = np.angle(x+1j*y)
            samps[i] = [np.append(data['UU--']*freq,-data['UU--']*freq),np.append(data['VV--']*freq,-data['VV--']*freq),np.append(ap,ap),np.append(ph,-ph)]

    print(f"\n Gridding VLBI data set.\n")

    # Generate Mask
    u_0 = samps[0][0]
    v_0 = samps[0][1]
    N = 63 # hard code
    mask = np.zeros((N,N,u_0.shape[0]))
    fov = 0.00018382*np.pi/(3600*180) # hard code #default 0.00018382
    # delta_u = 1/(fov*N/256) # hard code
    delta_u = 1/(fov)
    for i in range(N):
        for j in range(N):
            u_cell = (j-N/2)*delta_u
            v_cell = (i-N/2)*delta_u
            mask[i,j] = ((u_cell <= u_0) & (u_0 <= u_cell+delta_u)) & ((v_cell <= v_0) & (v_0 <= v_cell+delta_u))

    mask = np.flip(mask, [0])
    points = np.sum(mask, 2)
    points[points==0] = 1
    samp_img = np.zeros((size,2,N,N))
    img_resized = np.zeros((size,N,N))
    for i in tqdm(range(samps.shape[0])):
        samp_img[i][0] = np.matmul(mask, samps[i][2].T)/points
        samp_img[i][0] = (np.log10(samp_img[i][0] + 1e-10) / 10) + 1 
        samp_img[i][1] = np.matmul(mask, samps[i][3].T)/points
        img_resized[i] = cv2.resize(img[i], (N,N))
        img_resized[i] = img_resized[i]/np.sum(img_resized[i])

    # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])
    truth_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_resized, axes=(1,2)), axes=(1,2)), axes=(1,2))
    fft_scaled_truth = prepare_fft_images(truth_fft, True, False)

    out = data_path + "/samp_train0.h5"
    save_fft_pair(out, samp_img[:500], fft_scaled_truth[:500])
    out = data_path + "/samp_valid0.h5"
    save_fft_pair(out, samp_img[500:], fft_scaled_truth[500:])
