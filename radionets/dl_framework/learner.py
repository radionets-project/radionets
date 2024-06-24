import torch
import torch.nn as nn
from fastai.callback.schedule import ParamScheduler, combined_cos
from fastai.data.core import DataLoaders
from miniai.learner import Learner, MetricsCB, TrainCB, DeviceCB, ProgressCB
from fastai.optimizer import Adam
from functools import partial
from miniai.sgd import BatchSchedCB

import radionets.dl_framework.loss_functions as loss_functions
from radionets.dl_framework.callbacks import (
    AvgLossCallback,
    CometCallback,
    CudaCallback,
    DataAug,
    Normalize,
    SaveTempCallback,
    SwitchLoss,
    MixedPrecision,
)
from radionets.dl_framework.model import init_cnn
from functools import partial


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=partial(torch.optim.AdamW, eps=1e-4), **kwargs
):
    init_cnn(arch)
    dls = DataLoaders.from_dsets(
        data.train_ds, data.valid_ds, bs=data.train_dl.batch_size
    )
    return Learner(arch, dls, loss_func=loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def define_learner(data, arch, train_conf, lr_find=False, plot_loss=False):
    cbfs = [
        TrainCB(),
        DeviceCB(),
        MetricsCB(),
        ProgressCB(plot=False),
    ]
    model_path = train_conf["model_path"]
    lr = train_conf["lr"]
    opt_func = partial(torch.optim.AdamW, eps=1e-4)

    if train_conf["param_scheduling"]:
        sched = {
            "lr": combined_cos(
                train_conf["lr_ratio"],
                train_conf["lr_start"],
                train_conf["lr_max"],
                train_conf["lr_stop"],
            )
        }
        tmax = train_conf["num_epochs"] * len(data.train_ds)
        sched = partial(torch.optim.lr_scheduler.OneCycleLR,
            max_lr=train_conf["lr_max"],
            total_steps=tmax,
        )
        cbfs.extend([BatchSchedCB(sched)])

    if train_conf["gpu"]:
        cbfs.extend([CudaCallback])

    cbfs.extend(
        [
            SaveTempCallback(model_path=model_path),
            AvgLossCallback,
            # DataAug,
            # MixedPrecision(),
        ]
    )

    # use switch loss
    if train_conf["switch_loss"]:
        cbfs.extend(
            [
                SwitchLoss(
                    second_loss=getattr(loss_functions, "comb_likelihood"),
                    when_switch=train_conf["when_switch"],
                ),
            ]
        )

    if train_conf["comet_ml"] and not lr_find and not plot_loss:
        cbfs.extend(
            [
                CometCallback(
                    name=train_conf["project_name"],
                    test_data=train_conf["data_path"],
                    plot_n_epochs=train_conf["plot_n_epochs"],
                    amp_phase=train_conf["amp_phase"],
                    scale=train_conf["scale"],
                ),
            ]
        )

    #if not plot_loss and train_conf["normalize"] != "none":
    #    cbfs.extend([Normalize(train_conf)])
    # get loss func
    if train_conf["loss_func"] == "feature_loss":
        loss_func = loss_functions.init_feature_loss()
    else:
        loss_func = getattr(loss_functions, train_conf["loss_func"])

    # Combine model and data in learner
    learn = get_learner(
        data, arch, lr=lr, opt_func=opt_func, cb_funcs=cbfs, loss_func=loss_func
    )
    return learn
