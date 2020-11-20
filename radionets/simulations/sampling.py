import os
from tqdm import tqdm
from numpy import savez_compressed
from radionets.simulations.utils import (
    get_fft_bundle_paths,
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


def sample_frequencies(
    data_path,
    amp_phase,
    real_imag,
    fourier,
    compressed,
    interpolation,
    specific_mask,
    antenna_config,
    lon=None,
    lat=None,
    steps=None,
):
    for mode in ["train", "valid", "test"]:
        print(f"\n Sampling {mode} data set.\n")

        bundle_paths = get_fft_bundle_paths(data_path, "fft", mode)

        for path in tqdm(bundle_paths):
            fft, truth = open_fft_bundle(path)
            f = h5py.File(path, "r")
            z = np.array(f["z"])
            size = fft.shape[-1]

            fft_scaled = prepare_fft_images(fft.copy(), amp_phase, real_imag)
            truth_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in truth])
            fft_scaled_truth = prepare_fft_images(truth_fft, amp_phase, real_imag)

            if specific_mask is True:
                fft_samp = sample_freqs(
                    fft_scaled.copy(),
                    antenna_config,
                    size,
                    lon,
                    lat,
                    steps,
                    plot=False,
                    test=False,
                )
            else:
                fft_samp = sample_freqs(
                    fft_scaled.copy(),
                    antenna_config,
                    size=size,
                    specific_mask=False,
                )
            if interpolation:
                for i in range(len(fft_samp[:, 0, 0, 0])):
                    fft_samp[i] = interpol(fft_samp[i])

            out = data_path + "/samp_" + path.name.split("_")[-1]

            if fourier:
                if compressed:
                    savez_compressed(out, x=fft_samp, y=fft_scaled)
                    os.remove(path)
                else:
                    save_fft_pair(out, fft_samp, fft_scaled_truth)
            else:
                save_fft_pair_list(out, fft_samp, truth, z)
