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

from matplotlib.colors import LogNorm
test = open_bundle(bundles[0])
img = test[55]
plt.imshow(img)#, norm=LogNorm(vmin=1e-5))
plt.colorbar()

from dl_framework.model import fft

import numpy as np
img_fft = np.fft.fft2(img)

plt.imshow(np.fft.fftshift(img_fft.real))
plt.colorbar()

import torch
arr_real = torch.tensor(img).float()
arr_imag = torch.zeros(img.shape).float()

arr = torch.stack((arr_real, arr_imag), dim=-1).expand(1, -1, -1, -1)
arr.shape

torch.transpose()

arr_fft = torch.ifft(arr, 2).permute(0, 3, 2, 1).transpose(2, 3)
arr_fft.shape

axes = tuple(range(arr_fft.ndim))
shift = [-(dim // 2) for dim in arr_fft.shape]
arr_shift = torch.roll(arr_fft, shift, axes)
arr_shift.shape

plt.imshow(arr_shift[0][1])
plt.colorbar()



plt.imshow(shift[0][0].reshape(128,128))







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
train = [path for path in bundle_paths if re.findall('fft_samp_train', path.name)]
valid = [path for path in bundle_paths if re.findall('fft_samp_valid', path.name)]

train, valid

# +
from gaussian_sources.preprocessing import split_amp_phase, mean_and_std, split_real_imag
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

from dl_framework.data import DataBunch, get_dls, h5_dataset, split_real_imag
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
            data_amp, data_phase = split_real_imag(data)
            data_channel = combine_and_swap_axes(data_amp, data_phase).reshape(-1,4096)
        else:
            data_channel = data.reshape(4096)
        return torch.tensor(data_channel).float()


from dl_framework.model import Lambda, fft

import numpy as np
img_fft = np.fft.fft2(img)

plt.imshow(np.fft.fftshift(img_fft.real))

arr_real = torch.tensor(img).float()
arr_imag = torch.zeros(img.shape).float()

arr = torch.stack((arr_real, arr_imag), dim=-1).flatten().expand(1, -1)
arr.shape

shift = fft(arr)
shift.shape

plt.imshow(shift[0][0].reshape(128,128))

plt.imshow(shift[0][0])





def combine_and_swap_axes(array1, array2):
    return np.swapaxes(np.dstack((array1, array2)), 2, 0)


train_ds = h5_dataset(train)
valid_ds = h5_dataset(valid)

bs = 512
data = DataBunch(*get_dls(train_ds, valid_ds, bs))

plt.imshow(data.valid_ds[60][0].reshape(64,64))
plt.colorbar()

data.valid_ds[60][1].min()

plt.imshow(data.valid_ds[0][1].reshape(64,64))

next(iter(data.train_dl))[1].shape

np.mean([10**2, 20**2, 70**2])

# +
test = torch.tensor([1e-1]).float()
target = torch.tensor([1e-2]).float()

mse_log(test, target), mse(test, target)


# +
def mse(x, y):
    return ((x - y)**2).mean()

def mse_log(x, y):
    y[y <=0] = 1e-8
    return ((torch.log10(x) - torch.log10(y))**2).mean()


# +
import dl_framework.architectures as architecture
from dl_framework.learner import get_learner
from dl_framework.callbacks import Recorder, AvgStatsCallback,\
                                   BatchTransformXCallback, CudaCallback,\
                                   SaveCallback, view_tfm, ParamScheduler,\
                                   normalize_tfm
from functools import partial
import torch.nn as nn

from dl_framework.model import init_cnn

# Define architecture
arch = getattr(architecture, 'UNet_fft')()
init_cnn(arch)

# Define resize for mnist data
mnist_view = view_tfm(2, 64, 64)

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
adam_opt = torch.optim.Adam

# Combine model and data in learner
learn = get_learner(data, arch, 1e-3, opt_func=adam_opt,  cb_funcs=cbfs)

# Print model architecture
print(learn.model, '\n')

# Train the model, make it possible to stop at any given time
learn.fit(10)
# -
learn.recorder.plot_loss()

import pandas as pd
from dl_framework.data import do_normalisation
evaluate_model(data.valid_ds, learn.model, 'data/normalization_factors.csv', nrows=10)
plt.savefig('gauss_test.pdf')

learn.fit(5)



# +
from matplotlib.colors import LogNorm
def get_eval_img(valid_ds, model, norm_path):
    rand = np.random.randint(0, len(valid_ds))
    img = valid_ds[rand][0].cuda()
    norm = pd.read_csv(norm_path)
    img = do_normalisation(img, norm)
    h = int(np.sqrt(img.shape[1]))
    img = img.view(-1, h, h).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()
    return img, pred, h, rand


def evaluate_model(valid_ds, model, norm_path, nrows=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 6*nrows),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    for i in range(nrows):
        img, pred, h, rand = get_eval_img(valid_ds, model, norm_path)
        axes[i][0].set_title('x')
        axes[i][0].imshow(img[:, 0].view(h, h).cpu(), cmap='RdGy_r',
                          vmax=img.max(), vmin=-img.max())
        axes[i][1].set_title('y_pred')
        im = axes[i][1].imshow(pred.view(h, h),
                               #norm=LogNorm(vmin=1e-8),
                               #vmin=valid_ds[rand][1].min(),
                               #vmax=valid_ds[rand][1].max()
                              )
        axes[i][2].set_title('y_true')
        axes[i][2].imshow(valid_ds[rand][1].view(h, h),
                          #norm=LogNorm(
                          #    vmin=1e-8,
                          #    vmax=valid_ds[rand][1].max()
                          #),
                          #vmin=valid_ds[rand][1].min(),
                          #vmax=valid_ds[rand][1].max()
                         )
        fig.colorbar(im, cax=axes[i][3])
    plt.tight_layout()
# -




