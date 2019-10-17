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
from model import *
from training import view_tfm, sched_no, Recorder, AvgStatsCallback, ParamScheduler, BatchTransformXCallback, CudaCallback, Runner
from training import model_summary
from training import find_lr

# +
# Load train and valid data
path_train = 'data/mnist_samp_train.h5'
x_train, y_train = get_h5_data(path_train, columns=['x_train', 'y_train'])
path_valid = 'data/mnist_samp_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])

# Create train and valid datasets
train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid, log=True, quantile=True, positive=True)

# Create databunch with definde batchsize
bs = 128
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)
# + {}
# import numpy as np
# a = data.train_ds.x.reshape(10, 4096)
# print(a.shape)
# a.mean()

# +
# from preprocessing import noramlize_data
# a = data.train_ds.x.reshape(10, 4096)
# b = data.valid_ds.x.reshape(10, 4096)
# x_train, x_valid = noramlize_data(a, b)
# -

img = data.train_ds.x[4]
plt.imshow(img.reshape(64, 64), cmap='RdGy_r', vmax=img.max(), vmin=-img.max())
plt.xlabel('u')
plt.ylabel('v')
plt.colorbar(label='Amplitude')

# +
# Define loss function
loss_func = nn.MSELoss()

# Define model
def get_model(data, lr=1e-1): #1e-1
    model = nn.Sequential(
        *conv(1, 4, (3,3), 1),
        *conv(4, 8, (3,3), 1),
        *conv(8, 16, (3,3), 1),
        nn.MaxPool2d((3,3)),
        *conv(16, 32, (2,2), 1),
        *conv(32, 64, (2,2), 1),
        nn.MaxPool2d((2,2)),
        Lambda(flatten),
        nn.Linear(4096, data.c)
    )
    return model, optim.SGD(model.parameters(), lr=lr)


# +
# Combine model and data in learner
learn = Learner(*get_model(data), loss_func, data)

# Initialize convolutional layers
init_cnn(learn.model)


# +
# Find learning rate
# find_lr(learn)

# +
from training import MixUp

# Define resize for mnist data
mnist_view = view_tfm(1, 64, 64)

# Define schedueled learning rate
sched = sched_no(9e-2, 9e-2)

# Define callback functions
cbfs = [
    Recorder,
    partial(AvgStatsCallback, loss_func),
    partial(ParamScheduler, 'lr', sched),
    CudaCallback,
    partial(BatchTransformXCallback, mnist_view),
]

# Define runner
run = Runner(cb_funcs=cbfs)

# +
# Show model summary
# model_summary(run, learn, data)
# -



# Train model
run.fit(10, learn)

# %debug

# +
# Evaluate model
from inspection import evaluate_model

evaluate_model(valid_ds, learn.model, nrows=2)
plt.savefig('mnist_samp_results2.pdf', dpi=100, bbox_inches='tight', pad_inches=0.01)

# +
# Save model
# state = learn.model.state_dict()
# torch.save(state, './mnist_cnn_big_1.model')
# -

# Load model
m = learn.model
m.load_state_dict((torch.load('./models/cnn_samp_leak_fixed_mask.model')))
learn.model.cuda()



