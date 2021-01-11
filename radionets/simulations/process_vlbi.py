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
    bundle_paths = get_real_bundle_paths(data_path)
    size = len(bundle_paths[0])
    img = np.zeros((size,256,256))
    samps = np.zeros((size,4,21000))
    for i in tqdm(range(size)):
        sampled = bundle_paths[0][i]
        target = bundle_paths[1][i]

        with fits.open(target) as hdul:
            img[i] = hdul[0].data
        
        with fits.open(sampled) as hdul:
            data = hdul[4].data
            samps[i] = [np.append(data['UCOORD']/hdul[1].data['EFF_WAVE'],-data['UCOORD']/hdul[1].data['EFF_WAVE']),np.append(data['VCOORD']/hdul[1].data['EFF_WAVE'],-data['VCOORD']/hdul[1].data['EFF_WAVE']),np.append(data['VISAMP'],data['VISAMP']),np.append(data['VISPHI'],-data['VISPHI'])]

    print(f"\n Gridding VLBI data set.\n")

    # Generate Mask
    u_0 = samps[0][0]
    v_0 = samps[0][1]
    N = 127
    mask = np.zeros((N,N,21000))
    umax = max(u_0)
    delta_u = 2*umax/N
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
        samp_img[i][1] = np.deg2rad(np.matmul(mask, samps[i][3].T)/points)
        img_resized[i] = cv2.resize(img[i], (N,N))

    truth_fft = np.array([np.fft.fftshift(np.fft.fft2(im)) for im in img_resized])
    fft_scaled_truth = prepare_fft_images(truth_fft, True, False)

    out = data_path + "/samp_train0.h5"
    save_fft_pair(out, samp_img[:100], fft_scaled_truth[:100])
    out = data_path + "/samp_valid0.h5"
    save_fft_pair(out, samp_img[100:], fft_scaled_truth[100:])

    # return samp_img, fft_scaled_truth
        # f = h5py.File(path, "r")
        # z = np.array(f["z"])
        # size = fft.shape[-1]

        # fft_scaled = prepare_fft_images(fft.copy(), amp_phase, real_imag)
        # truth_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in truth])
        # fft_scaled_truth = prepare_fft_images(truth_fft, amp_phase, real_imag)

        # if specific_mask is True:
        #     fft_samp = sample_freqs(
        #         fft_scaled.copy(),
        #         antenna_config,
        #         size,
        #         lon,
        #         lat,
        #         steps,
        #         plot=False,
        #         test=False,
        #     )
        # else:
        #     fft_samp = sample_freqs(
        #         fft_scaled.copy(),
        #         antenna_config,
        #         size=size,
        #         specific_mask=False,
        #     )
        # if interpolation:
        #     for i in range(len(fft_samp[:, 0, 0, 0])):
        #         fft_samp[i] = interpol(fft_samp[i])

        # out = data_path + "/samp_" + path.name.split("_")[-1]

        # if fourier:
        #     if compressed:
        #         savez_compressed(out, x=fft_samp, y=fft_scaled)
        #         os.remove(path)
        #     else:
        #         save_fft_pair(out, fft_samp, fft_scaled_truth)
        # else:
        #     save_fft_pair_list(out, fft_samp, truth, z)
