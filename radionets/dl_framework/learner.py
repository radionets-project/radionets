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
from fastai.vision import gan, models
import torchvision


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=Adam, **kwargs
):
    init_cnn(arch)
    dls = DataLoaders.from_dsets(
        data.train_ds,
        data.valid_ds,
    )
    return Learner(dls, arch, loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def get_learner_gan(
    data, generator, discriminator, lr, gen_loss_func, crit_loss_func, cb_funcs=None, opt_func=Adam, **kwargs
):
    init_cnn(generator)
    init_cnn(discriminator)
    dls = DataLoaders.from_dsets(
        data.train_ds,
        data.valid_ds,
    )
    return gan.GANLearner(dls, generator, discriminator, gen_loss_func, crit_loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def define_learner(
    data,
    arch,
    train_conf,
    cbfs=[],
    test=False,
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
        # lr_max = train_conf["lr_max"]
        # div = 25.
        # div_final = 1e5
        # pct_start = 0.25
        # moms = (0.95, 0.85)
        # sched = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
        #       'mom': combined_cos(pct_start, moms[0], moms[1], moms[0])}
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
                SaveTempCallback(model_path=model_path, gan=train_conf["gan"]),
                AvgLossCallback,
                # DataAug,
            ]
        )
    if train_conf["telegram_logger"]:
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
    if train_conf["gan"]:
        gl = getattr(loss_functions, "gen_loss")
        dl = getattr(loss_functions, "disc_loss")
        # vgg = torchvision.models.vgg19_bn()
        # vgg.to(device)
        # print(arch)
        learn = get_learner_gan(
            data, arch[0], arch[1], lr=lr, gen_loss_func=gl, crit_loss_func=dl, opt_func=opt_func, cb_funcs=cbfs
        )
    else:
        learn = get_learner(
            data, arch, lr=lr, opt_func=opt_func, cb_funcs=cbfs, loss_func=loss_func
        )
    return learn
