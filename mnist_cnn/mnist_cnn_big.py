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
import sys
sys.path.append('..')

# +
import matplotlib.pyplot as plt

from preprocessing import get_h5_data, prepare_dataset, get_dls, DataBunch
from torch import nn
from dl_framework.learner import Learner
from dl_framework.optimizer import sgd_opt
from dl_framework.model import conv, Lambda, flatten, init_cnn
from dl_framework.callbacks import Recorder, AvgStatsCallback, ParamScheduler, CudaCallback, BatchTransformXCallback, view_tfm, SaveCallback
from functools import partial

# +
# Load train and valid data
path_train = 'data/mnist_samp_train.h5'
x_train, y_train = get_h5_data(path_train, columns=['x_train', 'y_train'])
path_valid = 'data/mnist_samp_valid.h5'
x_valid, y_valid = get_h5_data(path_valid, columns=['x_valid', 'y_valid'])

# Create train and valid datasets
# train_ds, valid_ds = prepare_dataset(x_train[0:8], y_train[0:8], x_valid[0:8], y_valid[0:8], log=True, quantile=True, positive=True)
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
from torch import optim
# Define model
def get_model(data): #1e-1
    model = nn.Sequential(
        *conv(1, 4, (3,3), 2, 3//2),
        *conv(4, 8, (3,3), 2, 3//2),
        *conv(8, 16, (3,3), 2, 3//2),
        nn.MaxPool2d((3,3)),
        *conv(16, 32, (2,2), 2, 1),
        *conv(32, 64, (2,2), 2, 1),
        nn.MaxPool2d((2,2)),
        Lambda(flatten),
        nn.Linear(64, data.c)
    )
    return model

def get_learner(data, lr, loss_func=nn.MSELoss(),
                cb_funcs=None, opt_func=sgd_opt, **kwargs):
    model = get_model(data, **kwargs)
    init_cnn(model)
    return Learner(model, data, loss_func, lr=lr, cb_funcs=cb_funcs, opt_func=opt_func)


# +
# from dl_framework.callbacks import LR_Find
# from dl_framework.optimizer import StatefulOptimizer, momentum_step, weight_decay, AverageGrad

# sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step, weight_decay],
#                      stats=AverageGrad(), wd=0.01)

# def find_lr(data):
#     mnist_view = view_tfm(1, 64, 64)
#     cbs = [
#         CudaCallback,
#         partial(BatchTransformXCallback, mnist_view),
#         LR_Find,
#         Recorder
#     ]
#     lr_find = get_learner(data, 1e-1, opt_func=adam_opt, cb_funcs=cbs)
#     print(lr_find.opt_func)
#     lr_find.fit(2)
#     lr_find.recorder.plot()

# # Find learning rate
# find_lr(data)

# +
# Combine model and data in learner
from dl_framework.optimizer import StatefulOptimizer, momentum_step, weight_decay, AverageGrad

sgd_mom_opt = partial(StatefulOptimizer, steppers=[momentum_step, weight_decay],
                     stats=AverageGrad(), wd=0.01)

from dl_framework.optimizer import adam_step, AverageSqrGrad, StepCount
from dl_framework.utils import listify

xtra_step=None

adam_opt = partial(StatefulOptimizer, steppers=[adam_step,weight_decay],
                   stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()])

from dl_framework.param_scheduling import sched_cos, combine_scheds
from dl_framework.callbacks import MixUp

# sched = combine_scheds([0.3,0.7], [sched_cos(1e-3, 5e-2), sched_cos(5e-2, 8e-4)])
sched = combine_scheds([0.2,0.8], [sched_cos(8e-4, 7e-3), sched_cos(7e-3, 8e-5)])

mnist_view = view_tfm(1, 64, 64)

cbfs = [
    Recorder,
    partial(AvgStatsCallback, nn.MSELoss()),
    partial(ParamScheduler, 'lr', sched),
    CudaCallback,
    partial(BatchTransformXCallback, mnist_view),
    MixUp,
    SaveCallback,
]

learn = get_learner(data, 1e-1, opt_func=adam_opt, cb_funcs=cbfs)
# -


learn.opt

learn.fit(2500)

# +
# Evaluate model
from inspection import evaluate_model

evaluate_model(valid_ds, learn.model, nrows=2)
# -
learn.recorder.plot_loss()
plt.yscale('log')


learn.recorder.plot_lr()
plt.yscale('log')

from dl_framework.utils import get_batch

test = learn.xb


def get_batch(dl, run):
    run.xb,run.yb = next(iter(dl))
    run.in_train=True
    for cb in run.cbs: cb.set_runner(run), print(cb)
    run('begin_batch')
#     run.in_train=False
    return run.xb,run.yb


x, y = get_batch(learn.data.train_dl, learn)
x.shape

x.shape

# img = x[3].squeeze(0).cpu() 
img = y[3].reshape(64, 64).cpu()
plt.imshow(img, cmap='RdBu', vmin=-img.max(), vmax=img.max())
plt.colorbar()

# +
# Show model summary
# model_summary(run, learn, data)
# -
import numpy as np



plt.imshow(img)

img_trans = np.abs(np.fft.fftshift(np.fft.fft2(img)))
plt.imshow(img_trans)

img_true = y[3].reshape(64, 64).cpu()
plt.imshow(img_true)

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
import torch
m = learn.model
m.load_state_dict((torch.load('./models/mnist_mixup_adam_leaky.model')))
learn.model.cuda()



