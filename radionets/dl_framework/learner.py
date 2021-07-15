import torch.nn as nn
from radionets.dl_framework.model import init_cnn
from radionets.dl_framework.callbacks import (
    NormCallback,
    SaveTempCallback,
    TelegramLoggerCallback,
    DataAug,
    AvgLossCallback,
    OverwriteOneBatch_CLEAN,
)
from fastai.optimizer import Adam, RMSProp
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.callback.data import CudaCallback
from fastai.callback.schedule import ParamScheduler, combined_cos
import radionets.dl_framework.loss_functions as loss_functions
from fastai.vision import models
# from radionets.dl_framework.architectures import superRes
import torchvision
from radionets.dl_training.utils import define_arch
from fastai.vision.gan import GANLearner, FixedGANSwitcher, _tk_diff, GANDiscriminativeLR
from fastai.callback.mixup import MixUp


def get_learner(
    data, arch, lr, loss_func=nn.MSELoss(), cb_funcs=None, opt_func=Adam, **kwargs
):
    init_cnn(arch)
    dls = DataLoaders.from_dsets(
        data.train_ds,
        data.valid_ds,
        bs=data.train_dl.batch_size,
    )
    return Learner(dls, arch, loss_func, lr=lr, cbs=cb_funcs, opt_func=opt_func)


def define_learner(
    data,
    arch,
    train_conf,
    cbfs=[],
    test=False,
    lr_find=False,
    gan=False,
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
    if not test and not gan:
        cbfs.extend(
            [
                SaveTempCallback(model_path=model_path),
                AvgLossCallback,
                DataAug(vgg=train_conf["vgg"], physics_informed=train_conf["physics_informed"]),
                # OverwriteOneBatch_CLEAN(5),
                # OverwriteOneBatch_CLEAN(10),
                MixUp(),
            ]
        )
    if gan:
        cbfs.extend(
            [
                SaveTempCallback(model_path=model_path, gan=gan),
                AvgLossCallback,
                DataAug(vgg=train_conf["vgg"], physics_informed=train_conf["physics_informed"]),
                # WGANL1Callback,
                # GANDiscriminativeLR,
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

    if gan:
        # gen_loss_func = getattr(loss_functions, 'gen_loss_func') #non physics informed
        gen_loss_func = getattr(loss_functions, 'l1_wgan_GANCS')
        crit_loss_func = getattr(loss_functions, 'crit_loss_func')

        generator = arch
        critic = define_arch(
            arch_name='GANCS_critic', img_size=train_conf["image_size"]
        )
        # init_cnn(generator)
        init_cnn(critic)
        dls = DataLoaders.from_dsets(
            data.train_ds,
            data.valid_ds,
            bs=data.train_dl.batch_size,
        )
        switcher = FixedGANSwitcher(n_crit=1, n_gen=1) #GAN
        # learn = GANLearner(dls, generator, critic, gen_loss_func, crit_loss_func, lr=lr, cbs=cbfs, opt_func=opt_func, switcher=switcher) #GAN
        # learn = GANLearner.wgan(dls, generator, critic, lr=lr, cbs=cbfs, opt_func=RMSProp) #WGAN
        learn = GANLearner(dls, generator, critic, gen_loss_func, _tk_diff, clip=0.01, switch_eval=False, lr=lr, cbs=cbfs, opt_func=RMSProp) #WGAN-l1
        return learn

    # Combine model and data in learner
    learn = get_learner(
        data, arch, lr=lr, opt_func=opt_func, cb_funcs=cbfs, loss_func=loss_func
    )
    return learn
