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

data_path = 'data/'
bundles = get_bundles(data_path)
bundles = [path for path in bundles if re.findall('gaussian_sources', path.name)]
bundles

# +
# %%time
samp = True
specific_mask = True
lon = -80
lat = 50
steps = 50
antenna_config_path = '../simulations/layouts/vlba.txt'
out_path = 'data/fft_samp'

for path in bundles[0:1]:
    bundle = open_bundle(path)
    bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in bundle])
    if samp is True:
        if specific_mask is True:
            bundle_fft = np.array([sample_freqs(img, antenna_config_path, 128, lon, lat, steps)
                                   for img in tqdm(bundle_fft)])
        else:
            bundle_fft = np.array([sample_freqs(img, antenna_config_path, size=128)
                                   for img in tqdm(bundle_fft)])
    out = out_path + path.name.split('_')[-1]
    # save_fft_pair(out, bundle_fft, bundle)
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
# -
data_path = 'data/'
bundle_paths = get_bundles(data_path)
bundle_paths = [path for path in bundle_paths if re.findall('fft_samp', path.name)]
bundle_paths

# +
from gaussian_sources.preprocessing import split_amp_phase, mean_and_std
from dl_framework.data import open_fft_pair

means_amp = np.array([])
stds_amp = np.array([])
means_phase = np.array([])
stds_phase = np.array([])

for path in tqdm(bundle_paths):
    x, _ = open_fft_pair(path)
    x_amp, x_phase = split_amp_phase(x)
    mean_amp, std_amp = mean_and_std(x_amp)
    mean_phase, std_phase = mean_and_std(x_phase)
    means_amp = np.append(mean_amp, means_amp)
    means_phase = np.append(mean_phase, means_phase)
    stds_amp = np.append(std_amp, stds_amp)
    stds_phase = np.append(std_phase, stds_phase)

mean_amp = means_amp.mean()
std_amp = stds_amp.mean()
mean_phase = means_phase.mean()
std_phase = stds_phase.mean()


d = {'train_mean_amp': [mean_amp],
     'train_std_amp': [std_amp],
     'train_mean_phase': [mean_phase],
     'train_std_phase': [std_phase]
     }
# -

d

bundle_paths

from dl_framework.data import DataBunch, get_dls, h5_dataset
import h5py
import torch


class h5_dataset():
    def __init__(self, bundle_paths):
        self.bundles = bundle_paths

    def __call__(self):
        return print('This is the h5_dataset class.')

    def __len__(self):
        return len(self.bundles) * len(self.open_bundle(self.bundles[0], 'x'))

    def __getitem__(self, i):
        x = self.open_image('x', i)
        y = self.open_image('y', i)
        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, 'r')
        data = bundle[str(var)]
        return data

    def open_image(self, var, i):
        bundle_i = i // 1024
        image_i = i - bundle_i * 1024
        bundle = h5py.File(self.bundles[bundle_i], 'r')
        data = bundle[str(var)][image_i]
        if var == 'x':
            data_amp, data_phase = split_amp_phase(data)
            data_channel = combine_and_swap_axes(data_amp, data_phase).reshape(-1,16384)
        else:
            data_channel = data.reshape(16384)
        return torch.tensor(data_channel).float()



def combine_and_swap_axes(array1, array2):
    return np.swapaxes(np.dstack((array1, array2)), 2, 0)


from gaussian_sources.preprocessing import split_amp_phase

train_ds = h5_dataset(bundle_paths)
valid_ds = h5_dataset(bundle_paths)

bs = 64
data = DataBunch(*get_dls(train_ds, valid_ds, bs))

data.train_ds[0][1].shape

next(iter(data.train_dl))[1].shape

# +
import dl_framework.architectures as architecture
from dl_framework.learner import get_learner
from dl_framework.optimizer import StatefulOptimizer, weight_decay,\
                                   AverageGrad
from dl_framework.optimizer import adam_step, AverageSqrGrad, StepCount
from dl_framework.param_scheduling import sched_no
from dl_framework.callbacks import Recorder, AvgStatsCallback,\
                                   BatchTransformXCallback, CudaCallback,\
                                   SaveCallback, view_tfm, ParamScheduler,\
                                   normalize_tfm
from functools import partial
import torch.nn as nn

# Define architecture
arch = getattr(architecture, 'cnn')()

# Define resize for mnist data
mnist_view = view_tfm(2, 128, 128)

# make normalisation
norm = normalize_tfm('./data/normalization_factors.csv')


# Define callback functions
cbfs = [
    Recorder,
    # test for use of multiple Metrics or Loss functions
    partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
    CudaCallback,
    partial(BatchTransformXCallback, norm),
    partial(BatchTransformXCallback, mnist_view),
    SaveCallback,
]

# Define optimiser function
adam_opt = partial(StatefulOptimizer, steppers=[adam_step, weight_decay],
                   stats=[AverageGrad(dampening=True), AverageSqrGrad(),
                   StepCount()])

# Combine model and data in learner
learn = get_learner(data, arch, 1e-3, opt_func=adam_opt,  cb_funcs=cbfs)

# Print model architecture
print(learn.model, '\n')

# Train the model, make it possible to stop at any given time
learn.fit(3)
# -
learn.recorder.plot_loss()















class h5_dataset():
    def __init__(self, bundle_paths):
        self.bundles = bundle_paths

    def __call__(self):
        return print('This is the h5_dataset class.')

    def __len__(self):
        return len(self.bundles) * len(self.open_bundle(self.bundles[0], 'x'))

    def __getitem__(self, i):
        x = self.open_image('x', i)
        y = self.open_image('y', i)
        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, 'r')
        data = bundle[str(var)]
        return data

    def open_image(self, var, i):
        bundle_i = i // 1024
        image_i = i - bundle_i * 1024
        bundle = h5py.File(self.bundles[bundle_i], 'r')
        data = bundle[str(var)][image_i]
        return data
