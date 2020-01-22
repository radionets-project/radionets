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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

from simulations.source_simulations import create_grid, gaussian_source

s = gaussian_source(128)
plt.imshow(s, norm=LogNorm(vmin=1e-8, vmax=10))
plt.colorbar()


def save_bundle(path, bundle, counter, name='gs_bundle'):
    with h5py.File(path + str(counter) + '.h5', 'w') as hf:
        hf.create_dataset(name,  data=bundle)
        hf.close()


def open_bundle(path):
    f = h5py.File(path, 'r')
    bundle = np.array(f['gs_bundle'])
    return bundle


def running_stats(path, num_bundles):
    means = np.array([])
    stds = np.array([])
    
    for i in range(num_bundles):
        bundle_path = path + str(i) + '.h5'
        bundle = open_bundle(bundle_path)
        bundle_mean = bundle.mean()
        bundle_std = bundle.std()
        means = np.append(bundle_mean, means)
        stds = np.append(bundle_std, stds)
    mean = means.mean()
    std = stds.mean()
    return mean, std


# %%time
for i in range(1024):
    s = gaussian_source(128)

# %%time
bundle = np.array([gaussian_source() for i in range(1024)])
print(bundle.shape)

# %%time
path = 'gaussian_sources/bundle_'
for j in range(20):
    bundle = np.array([gaussian_source() for i in range(1024)])
    save_bundle(path, bundle, j)

mean, std = running_stats('gaussian_sources/bundle_', 20)
mean,std

from pathlib import Path


def get_bundles(path):
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


path = 'gaussian_sources'
bundles = get_bundles(path)
bundles


class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))


from dl_framework.data import DataLoader
from functools import partial


class h5_dataset():
    def __init__(self, bundles_x, bundles_y):
        self.x = bundles_x
        self.y = bundles_y

    def __call__(self):
        return print('This is the h5_dataset class.')
        
    def __len__(self):
        return len(self.x) * len(self.open_bundle(self.x[0]))

    def __getitem__(self, i):
        x = self.open_image(self.x, i)
        y = self.open_image(self.y, i)
        return x, y

    def open_bundle(self, bundle):
        bundle = h5py.File(bundle, 'r')
        data = bundle['gs_bundle']
        return data
    
    def open_image(self, bundle, i):
        bundle_i = i // 1024
        image_i = i - bundle_i * 1024
        bundle = h5py.File(bundle[bundle_i], 'r')
        data = bundle['gs_bundle'][image_i]
        return data


train_ds = h5_dataset(bundles, bundles)

len(train_ds)

train_ds()

bs = 64

next(iter(train_ds.x))

val = next(iter(train_ds))
plt.imshow(val[0])

data = DataBunch(*get_dls(train_ds, train_ds, bs))

x, y = next(iter(data.train_dl))
plt.imshow(x[0])

# +
#loader = get_dls(train_ds, train_ds, bs)

# +
#loader

# +
#x = next(iter(loader[0]))
#img = x[0][0]
#plt.imshow(img)
# -

data.train_ds[125]






























