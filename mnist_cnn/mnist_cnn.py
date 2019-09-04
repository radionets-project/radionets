# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

from preprocessing import get_h5_data, prepare_dataset, get_dls, DataBunch

# +
# Load train and valid data
path_train = 'data/mnist_train.h5'
x_train, y_train = get_h5_data(path_train, columns=['x_train', 'y_train'])
path_valid = 'data/mnist_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])

# Create train and valid datasets
train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid, log=True)

# Create databunch with definde batchsize
bs = 128
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)
# -




