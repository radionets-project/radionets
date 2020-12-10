import torch.nn as nn
from radionets.dl_framework.model import init_cnn
from radionets.dl_framework.callbacks import (
    NormCallback,
    SaveTempCallback,
    TelegramLoggerCallback,
    DataAug,
    AvgLossCallback,
)
from fastai.optimizer import Adam
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.data import CudaCallback
from fastai.callback.schedule import ParamScheduler, combined_cos
import radionets.dl_framework.loss_functions as loss_functions


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=Adam, **kwargs
):
    init_cnn(arch)
    dls = DataLoaders.from_dsets(
        data.train_ds,
        data.valid_ds,
    )
    return Learner(dls, arch, loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def define_learner(
    data,
    arch,
    train_conf,
    cbfs=[],
    test=False,
    lr_find=False,
):
    model_path = train_conf["model_path"]
    model_name = (
        model_path.split("build/")[-1].split("/")[-1].split("/")[0].split(".")[0]
    )
    lr = train_conf["lr"]
    opt_func = Adam
    if train_conf["norm_path"] != "none":
        cbfs.extend(
            [
                NormCallback(train_conf["norm_path"]),
            ]
        )
    if train_conf["param_scheduling"]:
        sched = {
            "lr": combined_cos(
                0.25,
                train_conf["lr_start"],
                train_conf["lr_max"],
                train_conf["lr_stop"],
            )
        }
        cbfs.extend([ParamScheduler(sched)])
    if train_conf["gpu"]:
        cbfs.extend(
            [
                CudaCallback,
            ]
        )
    if not test:
        cbfs.extend(
            [
                SaveTempCallback(model_path=model_path),
                AvgLossCallback,
                DataAug,
            ]
        )
    if train_conf["telegram_logger"] and not lr_find:
        cbfs.extend(
            [
                TelegramLoggerCallback(model_name=model_name),
            ]
        )

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
