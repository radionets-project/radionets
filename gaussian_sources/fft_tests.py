# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from dl_framework.data import open_bundle, get_bundles, save_fft_pair
from simulations.uv_simulations import sample_freqs
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

path = 'data/'
bundles = get_bundles(path)
bundles = [path for path in bundles if re.findall('gaussian_sources', path.name)]
bundles

# %%time
for path in bundles:
    bundle = open_bundle(path)
    bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in bundle])
    out = 'data/fft' + path.name.split('_')[-1]
    save_fft_pair(out, bundle_fft, bundle)
    print(path)
    print(bundle.shape)
    print(bundle_fft.shape)
    print(out)

# +
i = 9

f = plt.figure(figsize=(10,10))
ax1 = plt.subplot(331)
ax1.imshow(np.abs(bundle[i]))

ax2 = plt.subplot(332)
ax2.imshow(np.abs(bundle_fft[i]))

ax3 = plt.subplot(333)
ax3.imshow(np.angle(bundle_fft[i]))

# +
# %%time
specific_mask = True
lon = -80
lat = 50
steps = 50
antenna_config_path = '../simulations/layouts/vlba.txt'

if specific_mask is True:
    bundle_fft_samp = np.array([sample_freqs(img, antenna_config_path, 128, lon, lat, steps)
                                for img in tqdm(bundle_fft)])
else:
    bundle_fft_samp = np.array([sample_freqs(img, antenna_config_path, size=128)
                                for img in tqdm(bundle_fft)])

# +
i = 9

f = plt.figure(figsize=(10,10))
ax1 = plt.subplot(331)
ax1.imshow(np.abs(bundle[i]))

ax2 = plt.subplot(332)
ax2.imshow(np.abs(bundle_fft[i]))

ax3 = plt.subplot(333)
ax3.imshow(np.angle(bundle_fft[i]))
# -






