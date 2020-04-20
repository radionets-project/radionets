import re
import sys
from functools import partial

import click

import dl_framework.architectures as architecture
import torch
import torch.nn as nn
from dl_framework.callbacks import (
    BatchTransformXCallback,
    CudaCallback,
    LR_Find,
    Recorder_lr_find,
    normalize_tfm,
    view_tfm,
)
from dl_framework.data import DataBunch, get_bundles, get_dls, h5_dataset
from dl_framework.learner import get_learner, define_learner
from dl_framework.model import load_pre_model
from inspection import plot_lr_loss


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("loss_func", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option(
    "-max_iter", type=float, required=True, help="maximal iterations for lr_find"
)
@click.option(
    "-min_lr", type=float, required=True, help="minimal learning rate for lr_find"
)
@click.option(
    "-max_lr", type=float, required=True, help="maximal learning rate for lr_find"
)
@click.option(
    "-fourier",
    type=bool,
    required=True,
    help="true, if target variables are fourier transformed",
)
@click.option(
    "-amp_phase",
    type=bool,
    required=True,
    help="true, if amplitude and phase splitting instead of real and imaginary",
)
@click.option("-log", type=bool, required=False, help="use of logarith")
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-save", type=bool, required=False, help="save the lr vs loss plot")
def main(
    data_path,
    arch,
    norm_path,
    loss_func,
    max_iter,
    min_lr,
    max_lr,
    fourier,
    amp_phase,
    log=True,
    pretrained=False,
    pretrained_model=None,
    save=False,
):
    """
    Train the neural network with existing training and validation data.
    TRAIN_PATH is the path to the training data\n
    VALID_PATH ist the path to the validation data\n
    ARCH is the name of the architecture which is used\n
    NORM_PATH is the path to the normalisation factors\n
    PRETRAINED_MODEL is the path to a pretrained model, which is
                     loaded at the beginning of the training\n
    """
    bundle_paths = get_bundles(data_path)
    train = [path for path in bundle_paths if re.findall("fft_samp_train", path.name)]
    valid = [path for path in bundle_paths if re.findall("fft_samp_valid", path.name)]

    # Create train and valid datasets
    train_ds = h5_dataset(train, tar_fourier=fourier, amp_phase=amp_phase)
    valid_ds = h5_dataset(valid, tar_fourier=fourier, amp_phase=amp_phase)

    # Create databunch with defined batchsize
    bs = 256
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    # First guess for max_iter
    print("\nTotal number of batches ~ ", len(data.train_ds) * 2 // bs)

    # Define model
    arch_name = arch
    arch = getattr(architecture, arch)()

    # Define resize based on the length of an input image
    img = train_ds[0][0]
    mnist_view = view_tfm(1, img.shape[0], img.shape[0])

    # make normalisation
    norm = normalize_tfm(norm_path)

    # Define callback functions
    cbfs = [
        partial(LR_Find, max_iter=max_iter, max_lr=max_lr, min_lr=min_lr),
        Recorder_lr_find,
    ]

    # if loss_func == "l1":
    #     loss_func = nn.L1Loss()
    # elif loss_func == "mse":
    #     loss_func = nn.MSELoss()
    # else:
    #     print("\n No matching loss function! Exiting. \n")
    #     sys.exit(1)
    # Combine model and data in learner
    # learn = get_learner(
    #     data, arch, 1e-3, opt_func=torch.optim.Adam, cb_funcs=cbfs, loss_func=loss_func
    # )

    learn = define_learner(
        data,
        arch,
        norm,
        loss_func,
        cbfs=cbfs,
    )

    # def loss(x, y, learn=learn):
    #     xb = learn.xb[-1, 0]
    #     unc = x[-1, 1][xb == -1]
    #     y_pred = x[-1, 0][xb == -1]
    #     loss = (
    #         (
    #             2 * torch.log(unc)
    #             + ((y.reshape(-1, 63, 63)[:, xb == -1] - y_pred) ** 2 / unc ** 2)
    #         )
    #     ).mean()
    #     return loss

    # learn.loss_func = loss

    # use pre-trained model if asked
    if pretrained is True:
        # Load model
        load_pre_model(learn, pretrained_model, lr_find=True)

    learn.fit(2)
    if save:
        plot_lr_loss(learn, arch_name, skip_last=5)
    else:
        learn.recorder_lr_find.plot(skip_last=5)


if __name__ == "__main__":
    main()
