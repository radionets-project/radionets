import gzip
import pickle
import os
import sys
import torch
from torch import nn
import numpy as np
from functools import partial
from skimage.transform import resize
from simulations.gaussian_simulations import add_noise
from dl_framework.data import save_fft_pair
from dl_framework.learner import get_learner
from dl_framework.loss_functions import init_feature_loss

from dl_framework.callbacks import (
    AvgStatsCallback,
    BatchTransformXCallback,
    CudaCallback,
    Recorder,
    SaveCallback,
    Recorder_lr_find,
    LR_Find,
)


def open_mnist(path):
    """
    Open MNIST data set pickle file.

    Parameters
    ----------
    path: str
        path to MNIST pickle file

    Returns
    -------
    train_x: 2d array
        50000 x 784 images
    valid_x: 2d array
        10000 x 784 images
    """
    with gzip.open(path, "rb") as f:
        ((train_x, _), (valid_x, _), _) = pickle.load(f, encoding="latin-1")
    return train_x, valid_x


def adjust_outpath(path, option):
    """
    Add number to out path when filename already exists.

    Parameters
    ----------
    path: str
        path to save directory
    option: str
        additional keyword to add to path

    Returns
    -------
    out: str
        adjusted path
    """
    counter = 0
    filename = path + "/fft_bundle_" + option + "{}.h5"
    while os.path.isfile(filename.format(counter)):
        counter += 1
    out = filename.format(counter)
    return out


def prepare_mnist_bundles(bundle, path, option, noise=False, pixel=63):
    """
    Resize images to specific squared pixel size and calculate fft

    Parameters
    ----------
    image: 2d array
        input image
    pixel: int
        number of pixel in x and y

    Returns
    -------
    x: 2d array
        fft of rescaled input image
    y: 2d array
        rescaled input image
    """
    y = [
        resize(bund, (pixel, pixel), anti_aliasing=True, mode="constant",)
        for bund in bundle
    ]
    y_prep = y.copy()
    if noise:
        y_prep = add_noise(y_prep)
    x = np.fft.fftshift(np.fft.fft2(y_prep))
    path = adjust_outpath(path, option)
    save_fft_pair(path, x, y)


def define_learner(
    data,
    arch,
    norm,
    loss_func,
    lr=1e-3,
    model_name='',
    model_path='',
    max_iter=400,
    max_lr=1e-1,
    min_lr=1e-6,
    test=False,
    lropt_func=torch.optim.Adam,
):
    if test:
        cbfs = [
            Recorder,
            partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
            partial(BatchTransformXCallback, norm),
            partial(SaveCallback, model_path=model_path),
            partial(LR_Find, max_iter=max_iter, max_lr=max_lr, min_lr=min_lr),
            Recorder_lr_find,
        ]
    else:
        from dl_framework.callbacks import LoggerCallback
        cbfs = [
            Recorder,
            partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
            CudaCallback,
            partial(BatchTransformXCallback, norm),
            partial(SaveCallback, model_path=model_path),
            partial(LoggerCallback, model_name=model_name),
        ]

    if loss_func == "feature_loss":
        loss_func = init_feature_loss()
    elif loss_func == "l1":
        loss_func = nn.L1Loss()
    elif loss_func == "mse":
        loss_func = nn.MSELoss()
    else:
        print("\n No matching loss function! Exiting. \n")
        sys.exit(1)

    # Combine model and data in learner
    learn = get_learner(
        data, arch, lr=lr, opt_func=torch.optim.Adam, cb_funcs=cbfs, loss_func=loss_func
    )
    return learn
