import torch.nn as nn
from radionets.dl_framework.model import init_cnn
from radionets.dl_framework.callbacks import (
    SaveTempCallback,
    DataAug,
    AvgLossCallback,
    SwitchLoss,
    CudaCallback,
    CometCallback,
)
from fastai.optimizer import Adam
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.schedule import ParamScheduler, combined_cos
import radionets.dl_framework.loss_functions as loss_functions


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=Adam, **kwargs
):
    init_cnn(arch)
    dls = DataLoaders.from_dsets(
        data.train_ds, data.valid_ds, bs=data.train_dl.batch_size
    )
    return Learner(dls, arch, loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def define_learner(data, arch, train_conf, lr_find=False, plot_loss=False):
    cbfs = []
    model_path = train_conf["model_path"]
    lr = train_conf["lr"]
    opt_func = Adam

    if train_conf["param_scheduling"]:
        sched = {
            "lr": combined_cos(
                train_conf["lr_ratio"],
                train_conf["lr_start"],
                train_conf["lr_max"],
                train_conf["lr_stop"],
            )
        }
        cbfs.extend([ParamScheduler(sched)])

    if train_conf["gpu"]:
        cbfs.extend([CudaCallback])

    cbfs.extend(
        [
            SaveTempCallback(model_path=model_path),
            AvgLossCallback,
            DataAug,
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
