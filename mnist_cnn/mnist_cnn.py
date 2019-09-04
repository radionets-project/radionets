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
path_train = 'data/mnist_train.h5'
x_train, y_train = get_h5_data(path_train, columns=['x_train', 'y_train'])
path_valid = 'data/mnist_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])

# Create train and valid datasets
train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid, log=True)

# Create databunch with definde batchsize
bs = 128
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)
# + {}
# Define cuda device else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define loss function
loss_func = nn.MSELoss()

# Define model
def get_model(data, lr=1e-1):
    model = nn.Sequential(
        *conv(1, 4, (3,3), 1),
        *conv(4, 8, (3,3), 1),
        nn.MaxPool2d((3,3)),
        Lambda(flatten),
        nn.Linear(3200, data.c)
    ).to(device)
    return model, optim.SGD(model.parameters(), lr=lr)


# +
# Combine model and data in learner
learn = Learner(*get_model(data), loss_func, data)

# Initialize convolutional layers
init_cnn(learn.model)
# -


# Find learning rate
find_lr(learn)

# +
# Define resize for mnist data
mnist_view = view_tfm(1, 64, 64)

# Define schedueled learning rate
sched = sched_no(9e-1, 9e-1)

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
# -

# Show model summary
model_summary(run, learn, data)

# Train model
run.fit(4000, learn)

# Evaluate model
evaluate(valid_ds, learn.model)

# Save model
state = learn.model.state_dict()
torch.save(state, './mnist_cnn_small_1')
