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

from dl_framework.data import open_bundle, get_bundles
import numpy as np
import matplotlib.pyplot as plt

path = 'data/'
bundles = get_bundles(path)
bundles

# %%time
for path in bundles:
    bundle = open_bundle(path)
    bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in bundle])
    print(path)
    print(bundle.shape)
    print(bundle_fft.shape)

# +
f = plt.figure(figsize=(10,10))
ax1 = plt.subplot(331)
ax1.imshow(np.abs(bundle[8]))

ax2 = plt.subplot(332)
ax2.imshow(np.abs(bundle_fft[8]))

ax3 = plt.subplot(333)
ax3.imshow(np.angle(bundle_fft[8]))
# -










