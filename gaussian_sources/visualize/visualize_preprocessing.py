import sys
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np
from mnist_cnn.preprocessing import prepare_dataset, get_dls, DataBunch
from mnist_cnn.utils import get_h5_data
from sampling.uv_simulations import sample_freqs
from mnist_cnn.visualize.utils import plot_mnist
from tqdm import tqdm

# Load train and valid data
path_train = '../data/mnist_train.h5'
x_train, y_train = get_h5_data(path_train, columns=['x_train', 'y_train'])
path_valid = '../data/mnist_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])

# Create train and valid datasets
train_ds, valid_ds = prepare_dataset(x_train[0:16], y_train[0:16], x_valid[0:16], y_valid[0:16],
                                    log=True, use_mask=True)

# Create databunch with definde batchsize
bs = 16
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)

c = 0
vals = np.random.choice(range(16), size=4, replace=False)
for i in tqdm(vals):
    # Plot y
    y = data.train_ds.y[i].reshape(64, 64)
    plt.figure(figsize=(6, 4))
    plot_mnist(y)
    plt.savefig("build/y" + str(c) + ".pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

    # Plot x_log
    plt.figure(figsize=(6, 4))
    x_log = data.train_ds.x[i].reshape(64, 64)
    plot_mnist(x_log)
    plt.savefig("build/x_log" + str(c) + ".pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

    # Plot x
    x = np.exp(data.train_ds.x[i].reshape(64, 64))
    plt.figure(figsize=(6, 4))
    plot_mnist(x)
    plt.savefig("build/x" + str(c) + ".pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

    # Plot x_samp
    x_samp, mask = sample_freqs(x_log.reshape(4096), '../../sampling/layouts/vlba.txt', plot=True)
    img_samp = x_log.numpy()
    img_samp[~mask] = 0
    plt.figure(figsize=(6, 4))
    plot_mnist(img_samp)
    plt.savefig("build/x_samp" + str(c) + ".pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)

    # Plot y_samp
    img_traf = np.abs(np.fft.fftshift(np.fft.fft2(img_samp)))
    plt.figure(figsize=(6, 4))
    plot_mnist(img_traf)
    plt.savefig("build/y_samp" + str(c) + ".pdf", dpi=100, bbox_inches='tight', pad_inches=0.01)
    c += 1