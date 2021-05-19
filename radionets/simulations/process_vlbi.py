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
    save_fft_pair_with_response,
)
from radionets.simulations.uv_simulations import sample_freqs
import h5py
import numpy as np
from astropy.io import fits
from PIL import Image
import cv2
import radionets.dl_framework.data as dt
import re
from natsort import natsorted, ns
from PIL import Image
import os
import vipy.simulation.utils as ut
import vipy.layouts.layouts as layouts
import astropy.constants as const
from astropy import units as un
import vipy.simulation.scan as scan

# set env flags to catch BLAS used for scipy/numpy 
# to only use 1 cpu, n_cpus will be totally controlled by csky
# flags from mirco
os.environ['MKL_NUM_THREADS'] = "12"
os.environ['NUMEXPR_NUM_THREADS'] = "12"
os.environ['OMP_NUM_THREADS'] = "12"
os.environ['OPENBLAS_NUM_THREADS'] = "12"
os.environ['VECLIB_MAXIMUM_THREADS'] = "12"


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
    bundles = dt.get_bundles('/net/big-tank/POOL/users/sfroese/vipy/eht/m87/blackhole/')
    freq = 227297*10**6 # hard code #eht 227297
    bundles_target = dt.get_bundles(bundles[1])
    bundles_input = dt.get_bundles(bundles[0])
    bundle_paths_target = natsorted(bundles_target)
    bundle_paths_input = natsorted(bundles_input)
    size = len(bundle_paths_target)
    img = np.zeros((size,256,256))
    samps = np.zeros((size,4,21000)) # hard code
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
    save_fft_pair(out, samp_img[:2300], fft_scaled_truth[:2300])
    out = data_path + "/samp_valid0.h5"
    save_fft_pair(out, samp_img[2300:], fft_scaled_truth[2300:])



def process_data_dirty_model(data_path, freq, n_positions, fov_asec):
    
    print(f"\n Loading VLBI data set.\n")
    bundles = dt.get_bundles(data_path)
    freq = freq*10**6 # hard code #eht 227297
    uvfits = dt.get_bundles(bundles[3])
    imgs = dt.get_bundles(bundles[2])
    configs = dt.get_bundles(bundles[0])
    uv_srt = natsorted(uvfits, alg=ns.PATH)
    img_srt = natsorted(imgs, alg=ns.PATH)
    size = 1000
    for p in tqdm(range(n_positions)):
        N = 63 # hard code
        with fits.open(uv_srt[p*1000]) as hdul:
            n_sampled = hdul[0].data.shape[0] #number of sampled points
            baselines = hdul[0].data['Baseline']
            baselines = np.append(baselines,baselines)
            unique_telescopes = hdul[3].data.shape[0]
            unique_baselines = (unique_telescopes**2 - unique_telescopes)/2

        # response matrices
        A = response(configs[p], N, unique_telescopes, unique_baselines)

        img = np.zeros((size,256,256))
        samps = np.zeros((size,4,n_sampled*2))
        print(f"\n Load subset.\n")
        for i in np.arange(p*1000, p*1000+1000):
            sampled = uv_srt[i]
            target = img_srt[i]

            img[i-p*1000] = np.asarray(Image.open(str(target)))
        
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
                samps[i-p*1000] = [np.append(data['UU--']*freq,-data['UU--']*freq),np.append(data['VV--']*freq,-data['VV--']*freq),np.append(ap,ap),np.append(ph,-ph)]

        print(f"\n Gridding VLBI data set.\n")

        # Generate Mask
        u_0 = samps[0][0]
        v_0 = samps[0][1]
        mask = np.zeros((N,N,u_0.shape[0]))
        
        base_mask = np.zeros((N,N,int(unique_baselines)))
        
        fov = fov_asec*np.pi/(3600*180) # hard code #default 0.00018382
        # delta_u = 1/(fov*N/256) # hard code
        delta_u = 1/(fov) # with a set N this is the same as zooming in since N*delta_u can be smaller than u_max
        delta_u = (2*max(np.max(u_0),np.max(v_0))/N) # test gridding pixel size
        # print(delta_u)
        for i in range(N):
            for j in range(N):
                u_cell = (j-N/2)*delta_u
                v_cell = (i-N/2)*delta_u
                mask[i,j] = ((u_cell <= u_0) & (u_0 <= u_cell+delta_u)) & ((v_cell <= v_0) & (v_0 <= v_cell+delta_u))
                
                base = np.unique(baselines[mask[i,j].astype(bool)])
                base_mask[i,j,:base.shape[0]] = base

        mask = np.flip(mask, [0])
        points = np.sum(mask, 2)
        points[points==0] = 1
        samp_img = np.zeros((size,2,N,N))
        img_resized = np.zeros((size,N,N))
        for i in range(samps.shape[0]):
            samp_img[i][0] = np.matmul(mask, samps[i][2].T)/points
            samp_img[i][0] = (np.log10(samp_img[i][0] + 1e-10) / 10) + 1 
            samp_img[i][1] = np.matmul(mask, samps[i][3].T)/points
            img_resized[i] = cv2.resize(img[i], (N,N))
            img_resized[i] = img_resized[i]/np.sum(img_resized[i])

        # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])
        truth_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_resized, axes=(1,2)), axes=(1,2)), axes=(1,2))
        fft_scaled_truth = prepare_fft_images(truth_fft, True, False)

        out = data_path + "/h5/samp_train"+ str(p) + ".h5"
        save_fft_pair_with_response(out, samp_img[:800], fft_scaled_truth[:800], base_mask, A)
        out = data_path + "/h5/samp_valid"+ str(p) + ".h5"
        save_fft_pair_with_response(out, samp_img[800:], fft_scaled_truth[800:], base_mask, A)


def response(config, N, unique_telescopes, unique_baselines):
    rc = ut.read_config(config)
    array_layout = layouts.get_array_layout('vlba')
    src_crd = rc['src_coord']

    wave = const.c/((float(rc['channel'].split(':')[0])/2)*10**6/un.second)/un.meter
    rd = scan.rd_grid(rc['fov_size']*np.pi/(3600*180),N, src_crd)
    E = scan.getE(rd, array_layout, wave, src_crd)
    A = np.zeros((N,N,int(unique_baselines)))
    counter = 0
    for i in range(int(unique_telescopes)):
        for j in range(int(unique_telescopes)):
            if i == j or j < i:
                continue
            A[:,:,counter] = E[:,:,i]*E[:,:,j]
            counter += 1
    
    return A
        
