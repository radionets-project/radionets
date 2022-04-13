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



def process_data_dirty_model(data_path, freq, n_positions, fov_asec, layout):
    
    print(f"\n Loading VLBI data set.\n")
    bundles = dt.get_bundles(data_path)
    freq = freq*10**6 # mhz hard code #eht 227297
    uvfits = dt.get_bundles(bundles[2])
    imgs = dt.get_bundles(bundles[4])
    configs = dt.get_bundles(bundles[1])
    uv_srt = natsorted(uvfits, alg=ns.PATH)
    img_srt = natsorted(imgs, alg=ns.PATH)
    size = 1000
    for p in tqdm(range(n_positions)):
        N = 64 # hard code
        with fits.open(uv_srt[p*1000]) as hdul:
            n_sampled = hdul[0].data.shape[0] #number of sampled points
            baselines = hdul[0].data['Baseline']
            baselines = np.append(baselines,baselines)
            unique_telescopes = hdul[3].data.shape[0]
            unique_baselines = (unique_telescopes**2 - unique_telescopes)/2

        # response matrices
        A = response(configs[p], N, unique_telescopes, unique_baselines, layout)

        img = np.zeros((size,128,128))
        samps = np.zeros((size,4,n_sampled*2))
        print(f"\n Load subset.\n")
        for i in np.arange(p*1000, p*1000+1000):
            # print(i)
            sampled = uv_srt[i]
            target = img_srt[i] # +1000 because I had to only grid images from 1000-1999

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
        # delta_u = (2*max(np.max(u_0),np.max(v_0))/N) # test gridding pixel size
        # biggest_baselines = 8611*1e3 
        # wave = const.c/(freq/un.second)/un.meter
        # uv_max = biggest_baselines/wave
        # delta_u = uv_max/N
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

        out = data_path + "/h5/samp_train"+ str(p) +".h5"
        save_fft_pair_with_response(out, samp_img[:800], fft_scaled_truth[:800], np.expand_dims(base_mask,0), np.expand_dims(A,0))
        out = data_path + "/h5/samp_valid"+ str(p) + ".h5"
        save_fft_pair_with_response(out, samp_img[800:], fft_scaled_truth[800:], np.expand_dims(base_mask,0), np.expand_dims(A,0))


def response(config, N, unique_telescopes, unique_baselines, layout='vlba'):
    rc = ut.read_config(config)
    array_layout = layouts.get_array_layout(layout)
    src_crd = rc['src_coord']

    wave = const.c/((float(rc['channel'].split(':')[0]))*10**6/un.second)/un.meter
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
    

def process_measurement(data_path, file, config, fov_asec):
    
    print(f"\n Loading VLBI data set.\n")
    configs = config
    size = 1
    N=64
    with fits.open(file) as hdul:
        n_sampled = hdul[0].data.shape[0] #number of sampled points
        baselines = hdul[0].data['Baseline']
        
        unique_telescopes = hdul[3].data.shape[0]
        unique_baselines = (unique_telescopes**2 - unique_telescopes)/2
        freq = hdul[0].header[37]
        offset = hdul[2].data['IF FREQ']
        for o in offset[0][1:]:
            break
            baselines = np.append(baselines,hdul[0].data['Baseline'])
        baselines = np.append(baselines,baselines)
    # response matrices
    A = response(configs, N, unique_telescopes, unique_baselines, 'vlba')

    samps = np.zeros((size,4,n_sampled*2))
    print(f"\n Load subset.\n")


    with fits.open(file) as hdul:
        data = hdul[0].data
        cmplx = data['DATA'] 
        x = cmplx[...,0,0]
        y = cmplx[...,0,1]
        w = cmplx[...,0,2]
        x = np.squeeze(x)[:,0]
        y = np.squeeze(y)[:,0]
        w = np.squeeze(w)[:,0]
        ap = np.sqrt(x**2+y**2)
        ph = np.angle(x+1j*y)
        u = np.array([])
        v = np.array([])
        for f in offset[0]:
            u = np.append(u,data['UU--']*(freq+f))
            v = np.append(v,data['VV--']*(freq+f))
            break
        samps = [np.append(u,-u),np.append(v,-v),np.append(ap,ap),np.append(ph,-ph)]
    import matplotlib.pyplot as plt
    plt.plot(samps[0], samps[1], 'x')
    plt.show()
    print(f"\n Gridding VLBI data set.\n")

    # Generate Mask
    u_0 = samps[0]
    v_0 = samps[1]
    mask = np.zeros((N,N,u_0.shape[0]))
    
    base_mask = np.zeros((N,N,int(unique_baselines)))
    
    fov = fov_asec*np.pi/(3600*180) # hard code #default 0.00018382
    delta_u = 1/(fov) # with a set N this is the same as zooming in since N*delta_u can be smaller than u_max

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
    img_resized = np.zeros((size,N,N))

    samp_img = np.zeros((size,2,N,N))
    print(mask.shape)
    print(samps[2].shape)
    samp_img[0,0] = np.matmul(mask, samps[2].T)/points
    samp_img[0,0] = (np.log10(samp_img[0,0] + 1e-10) / 10) + 1 
    samp_img[0,1] = np.matmul(mask, samps[3].T)/points


    # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])

    out = data_path + "/h5/samp_meas.h5"
    save_fft_pair_with_response(out, samp_img, samp_img, np.expand_dims(base_mask,0), np.expand_dims(A,0))


def process_eht(data_path, file, config, fov_asec):
    
    print(f"\n Loading VLBI data set.\n")
    configs = config
    size = 1
    N=64
    with fits.open(file) as hdul:
        baselines = hdul[0].data['Baseline']
        
        unique_telescopes = 8
        unique_baselines = (unique_telescopes**2 - unique_telescopes)/2
        freq = 229071e6#hdul[0].header[37]
        baselines = np.append(baselines,baselines)
    # response matrices
    A = response(configs, N, unique_telescopes, unique_baselines, 'eht')

    # samps = np.zeros((size,4,n_sampled*2))
    print(f"\n Load subset.\n")


    with fits.open(file) as hdul:
        data = hdul[0].data
        cmplx = data['DATA'] 
        x = cmplx[...,0,0]
        y = cmplx[...,0,1]
        w = cmplx[...,0,2]
        x = np.squeeze(x)
        y = np.squeeze(y)
        w = np.squeeze(w)
        ap = np.sqrt((x*w)**2+(y*w)**2)
        ph = np.angle(x*w+1j*y*w)
        u = np.array([])
        v = np.array([])
        u = np.append(u,data['UU---SIN']*(freq))
        v = np.append(v,data['VV---SIN']*(freq))
        samps = [np.append(u,-u),np.append(v,-v),np.append(ap,ap),np.append(ph,-ph)]
    
    # plt.plot(samps[0], samps[1], 'x')
    # plt.show()
    print(f"\n Gridding VLBI data set.\n")

    # Generate Mask
    u_0 = samps[0]
    v_0 = samps[1]
    mask = np.zeros((N,N,u_0.shape[0]))
    
    base_mask = np.zeros((N,N,int(unique_baselines)))
    
    fov = fov_asec*np.pi/(3600*180) # hard code #default 0.00018382
    delta_u = 1/(fov) # with a set N this is the same as zooming in since N*delta_u can be smaller than u_max

    for i in range(N):
        for j in range(N):
            u_cell = (j-N/2)*delta_u
            v_cell = (i-N/2)*delta_u
            mask[i,j] = ((u_cell <= u_0) & (u_0 <= u_cell+delta_u)) & ((v_cell <= v_0) & (v_0 <= v_cell+delta_u))
            
            base = np.unique(baselines[mask[i,j].astype(bool)])
            base_mask[i,j,:base.shape[0]] = base

    mask = np.flip(mask, [0])
    import matplotlib.pyplot as plt
    # plt.imshow(np.sum(mask, 2))
    # plt.show()
    points = np.sum(mask, 2)
    points[points==0] = 1

    samp_img = np.zeros((size,2,N,N))
    print(mask.shape)
    print(samps[2].shape)
    samp_img[0,0] = np.matmul(mask, samps[2].T)/points
    samp_img[0,0] = (np.log10(samp_img[0,0] + 1e-10) / 10) + 1 
    samp_img[0,1] = np.matmul(mask, samps[3].T)/points
    plt.imshow(samp_img[0,0])
    plt.colorbar()
    plt.show()
    plt.imshow(samp_img[0,1])
    plt.colorbar()
    plt.show()


    # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])

    out = data_path + "/eht_hi_test_DPG_startmod.h5"
    save_fft_pair_with_response(out, samp_img, samp_img, np.expand_dims(base_mask,0), np.expand_dims(A,0))

def process_eht_hist(data_path, file, config, fov_asec):
    
    print(f"\n Loading VLBI data set.\n")
    configs = config
    size = 1
    N=64
    with fits.open(file) as hdul:
        baselines = hdul[0].data['Baseline']
        
        unique_telescopes = 8
        unique_baselines = (unique_telescopes**2 - unique_telescopes)/2
        freq = 229071e6#hdul[0].header[37]
        baselines = np.append(baselines,baselines)
    # response matrices
    A = response(configs, N, unique_telescopes, unique_baselines, 'eht')

    # samps = np.zeros((size,4,n_sampled*2))
    print(f"\n Load subset.\n")


    with fits.open(file) as hdul:
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
        u = np.array([])
        v = np.array([])
        u = np.append(u,data['UU---SIN']*(freq))
        v = np.append(v,data['VV---SIN']*(freq))
        samps = [np.append(u,-u),np.append(v,-v),np.append(ap,ap),np.append(ph,-ph)]
    
    # plt.plot(samps[0], samps[1], 'x')
    # plt.show()
    print(f"\n Gridding VLBI data set.\n")

    # Generate Mask
    u_0 = samps[0]
    v_0 = samps[1]
    mask = np.zeros((N,N,u_0.shape[0]))
    
    base_mask = np.zeros((N,N,int(unique_baselines)))
    
    fov = fov_asec*np.pi/(3600*180) # hard code #default 0.00018382
    delta_u = 1/(fov) # with a set N this is the same as zooming in since N*delta_u can be smaller than u_max
    binpos = np.arange(N//2+1)*delta_u
    binsneg = -np.flip(np.arange(N//2+1)*delta_u)
    bins = np.append(binsneg,binpos)
    bins = np.unique(bins)
    for i in range(N):
        for j in range(N):
            u_cell = (j-N/2)*delta_u
            v_cell = (i-N/2)*delta_u
            mask[i,j] = ((u_cell <= u_0) & (u_0 <= u_cell+delta_u)) & ((v_cell <= v_0) & (v_0 <= v_cell+delta_u))
            
            base = np.unique(baselines[mask[i,j].astype(bool)])
            base_mask[i,j,:base.shape[0]] = base

    import matplotlib.pyplot as plt
    # plt.imshow(np.sum(mask, 2))
    # plt.show()
    amp_cal,_,_,_ = plt.hist2d(samps[1],samps[0],bins=bins, weights=np.append(ap,ap))
    phase_cal,_,_,_ = plt.hist2d(samps[1],samps[0],bins=bins, weights=np.append(ph,-ph))
    points_cal,_,_,_ = plt.hist2d(samps[1],samps[0],bins=bins)
    points_cal[points_cal==0]=1
    amp_cal = amp_cal/points_cal
    phase_cal = phase_cal/points_cal

    samp_img = np.zeros((size,2,N,N))
    print(mask.shape)
    print(samps[2].shape)
    samp_img[0,0] = amp_cal
    samp_img[0,0] = (np.log10(samp_img[0,0] + 1e-10) / 10) + 1 
    samp_img[0,1] = phase_cal
    plt.imshow(points_cal)
    plt.colorbar()
    plt.show()
    plt.imshow(samp_img[0,0])
    plt.colorbar()
    plt.show()
    plt.imshow(samp_img[0,1])
    plt.colorbar()
    plt.show()


    # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])

    out = data_path + "/eht_hi_test_DPG.h5"
    save_fft_pair_with_response(out, samp_img, samp_img, np.expand_dims(base_mask,0), np.expand_dims(A,0))




def process_data_dirty_model_noisy(data_path, freq, n_positions, fov_asec, layout):
    
    print(f"\n Loading VLBI data set.\n")
    bundles = dt.get_bundles(data_path)
    freq = freq*10**6 # mhz hard code #eht 227297
    uvfits = dt.get_bundles(bundles[3])
    imgs = dt.get_bundles(bundles[2])
    configs = dt.get_bundles(bundles[0])
    uv_srt = natsorted(uvfits, alg=ns.PATH)[50000:]
    img_srt = natsorted(imgs, alg=ns.PATH)[50000:]
    configs = natsorted(configs, alg=ns.PATH)[50:]
    size = 1000
    for p in tqdm(range(n_positions)):
        N = 64 # hard code
        with fits.open(uv_srt[p*1000]) as hdul:
            n_sampled = hdul[0].data.shape[0] #number of sampled points
            baselines = hdul[0].data['Baseline']
            baselines = np.append(baselines,baselines)
            unique_telescopes = hdul[3].data.shape[0]
            unique_baselines = (unique_telescopes**2 - unique_telescopes)/2

        # response matrices
        A = response(configs[p], N, unique_telescopes, unique_baselines, layout)

        img = np.zeros((size,256,256))
        samps = np.zeros((size,4,n_sampled*2))
        print(f"\n Load subset.\n")
        for i in np.arange(p*1000, p*1000+1000):
            # print(i)
            sampled = uv_srt[i]
            target = img_srt[i] # +1000 because I had to only grid images from 1000-1999

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
        # delta_u = (2*max(np.max(u_0),np.max(v_0))/N) # test gridding pixel size
        # biggest_baselines = 8611*1e3 
        # wave = const.c/(freq/un.second)/un.meter
        # uv_max = biggest_baselines/wave
        # delta_u = uv_max/N
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
            # samp_img[i][0] = (np.log10(samp_img[i][0] + 1e-10) / 10) + 1 
            samp_img[i][1] = np.matmul(mask, samps[i][3].T)/points
            img_resized[i] = cv2.resize(img[i], (N,N))
            img_resized[i] = img_resized[i]/np.sum(img_resized[i])

        ### nooiiiiiseeeee
        np.random.seed(42)
        noise = np.random.normal(size=(size, N, N))
        m = np.zeros((1000,64,64))
        m[:] = np.sum(mask, 2)
        m[m != 0] = 1
        ft_noise = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(noise)))
        ft_noise[m == 0] = 0
        noise_dirty = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ft_noise)))

        compl = samp_img[:,0]*np.exp(1j*samp_img[:,1])    
        dirty_img = abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(compl))))
        for idx, d in enumerate(dirty_img):
            max = np.max(d)
            std = np.std(noise_dirty[idx])
            snr = np.random.uniform(2,10)
            alpha = max/(std*snr)
            dirty_img[idx] = dirty_img[idx] + abs(noise_dirty[idx]*alpha)

        measured = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(dirty_img)))
        mask = np.sum(mask, 2)
        for idx, m in enumerate(measured):
            samp_img[idx][0] = (np.log10(np.abs(m) + 1e-10) / 10) + 1 
            samp_img[idx][0][mask == 0] = 0
            samp_img[idx][1] = np.angle(m)
            samp_img[idx][1][mask == 0] = 0
        import matplotlib.pyplot as plt
        plt.imshow(samp_img[0][0])
        plt.colorbar()
        plt.show()


        # truth_fft = np.array([np.fft.fft2(np.fft.fftshift(img)) for im in img_resized])
        truth_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_resized, axes=(1,2)), axes=(1,2)), axes=(1,2))
        fft_scaled_truth = prepare_fft_images(truth_fft, True, False)

        out = data_path + "/h5/bh/samp_train"+ str(p) +".h5"
        save_fft_pair_with_response(out, samp_img[:800], fft_scaled_truth[:800], np.expand_dims(base_mask,0), np.expand_dims(A,0))
        out = data_path + "/h5/bh/samp_valid"+ str(p) + ".h5"
        save_fft_pair_with_response(out, samp_img[800:], fft_scaled_truth[800:], np.expand_dims(base_mask,0), np.expand_dims(A,0))

def process_data_dirty_model_noisy_pointSource(data_path, freq, n_positions, fov_asec, layout):
    
    print(f"\n Loading VLBI data set.\n")
    bundles = dt.get_bundles(data_path)
    print(bundles)
    freq = freq*10**6 # mhz hard code #eht 227297
    uvfits = dt.get_bundles(bundles[2])
    imgs = dt.get_bundles(bundles[4])
    configs = dt.get_bundles(bundles[1])
    uv_srt = natsorted(uvfits, alg=ns.PATH)
    img_srt = natsorted(imgs, alg=ns.PATH)
    size = 1000
    for p in tqdm(range(n_positions)):
        N = 64 # hard code
        with fits.open(uv_srt[p*1000]) as hdul:
            n_sampled = hdul[0].data.shape[0] #number of sampled points
            baselines = hdul[0].data['Baseline']
            baselines = np.append(baselines,baselines)
            unique_telescopes = hdul[3].data.shape[0]
            unique_baselines = (unique_telescopes**2 - unique_telescopes)/2

        # response matrices
        A = response(configs[p], N, unique_telescopes, unique_baselines, layout)

        img = np.zeros((size,128,128))
        samps = np.zeros((size,4,n_sampled*2))
        print(f"\n Load subset.\n")
        for i in np.arange(p*1000, p*1000+1000):
            # print(i)
            sampled = uv_srt[i]
            target = img_srt[i] # +1000 because I had to only grid images from 1000-1999

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
        # delta_u = (2*max(np.max(u_0),np.max(v_0))/N) # test gridding pixel size
        # biggest_baselines = 8611*1e3 
        # wave = const.c/(freq/un.second)/un.meter
        # uv_max = biggest_baselines/wave
        # delta_u = uv_max/N
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
            # samp_img[i][0] = (np.log10(samp_img[i][0] + 1e-10) / 10) + 1 
            samp_img[i][1] = np.matmul(mask, samps[i][3].T)/points
            img_resized[i] = cv2.resize(img[i], (N,N))
            img_resized[i] = img_resized[i]/np.sum(img_resized[i])

        ### nooiiiiiseeeee
        np.random.seed(42)
        noise = np.random.normal(size=(size, N, N))
        m = np.zeros((1000,64,64))
        m[:] = np.sum(mask, 2)
        m[m != 0] = 1
        ft_noise = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(noise)))
        ft_noise[m == 0] = 0
        noise_dirty = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ft_noise)))

        compl = samp_img[:,0]*np.exp(1j*samp_img[:,1])    
        dirty_img = abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(compl))))
        for idx, d in enumerate(dirty_img):
            max = np.max(d)
            std = np.std(noise_dirty[idx])
            snr = np.random.uniform(2,10)
            alpha = max/(std*snr)
            dirty_img[idx] = dirty_img[idx] + abs(noise_dirty[idx]*alpha)

        measured = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(dirty_img)))
        mask = np.sum(mask, 2)
        for idx, m in enumerate(measured):
            samp_img[idx][0] = (np.log10(np.abs(m) + 1e-10) / 10) + 1 
            samp_img[idx][0][mask == 0] = 0
            samp_img[idx][1] = np.angle(m)
            samp_img[idx][1][mask == 0] = 0
        import matplotlib.pyplot as plt
        plt.imshow(samp_img[0][0])
        plt.colorbar()
        plt.show()


        #point source label
        position = np.zeros((size,N,N))
        result = np.array([np.unravel_index(np.argmax(r), r.shape) for r in img_resized])
        for i in range(2):
            position[i,result[i][0],result[i][1]] = 1
        plt.imshow(position[0])
        plt.colorbar()
        plt.show()


        truth_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_resized, axes=(1,2)), axes=(1,2)), axes=(1,2))
        fft_scaled_truth = prepare_fft_images(truth_fft, True, False)

        out = data_path + "/h5/samp_train"+ str(p) +".h5"
        save_fft_pair_with_response(out, samp_img[:800], fft_scaled_truth[:800], np.expand_dims(base_mask,0), np.expand_dims(A,0))
        out = data_path + "/h5/samp_valid"+ str(p) + ".h5"
        save_fft_pair_with_response(out, samp_img[800:], fft_scaled_truth[800:], np.expand_dims(base_mask,0), np.expand_dims(A,0))
