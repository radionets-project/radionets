import sys
import numpy as np
from functools import partial

import click
import matplotlib.pyplot as plt

import dl_framework.architectures as architecture
import torch
import torch.nn as nn
from dl_framework.callbacks import (
    AvgStatsCallback,
    BatchTransformXCallback,
    CudaCallback,
    Recorder,
    SaveCallback,
    normalize_tfm,
    view_tfm,
    LoggerCallback,
)
from dl_framework.learner import get_learner
from dl_framework.loss_functions import init_feature_loss
from dl_framework.model import load_pre_model, save_model
from inspection import evaluate_model, plot_loss
from dl_framework.data import DataBunch, get_dls, h5_dataset, get_bundles
import re


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("num_epochs", type=int)
@click.argument("lr", type=float)
@click.argument("loss_func", type=str)
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option(
    "-fourier",
    type=bool,
    required=False,
    help="true, if target variables get fourier transformed",
)
@click.option("-log", type=bool, required=True, help="use of logarithm")
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-inspection", type=bool, required=False, help="make an inspection plot")
def main(
    data_path,
    model_path,
    arch,
    norm_path,
    num_epochs,
    lr,
    loss_func,
    fourier,
    log=True,
    pretrained=False,
    pretrained_model=None,
    inspection=False,
):
    """
    Train the neural network with existing training and validation data.

    TRAIN_PATH is the path to the training data\n
    VALID_PATH ist the path to the validation data\n
    MODEL_PATH is the Path to which the model is saved\n
    ARCH is the name of the architecture which is used\n
    NORM_PATH is the path to the normalisation factors\n
    NUM_EPOCHS is the number of epochs\n
    LR is the learning rate\n
    PRETRAINED_MODEL is the path to a pretrained model, which is
                     loaded at the beginning of the training\n
    """
    # Load data
    bundle_paths = get_bundles(data_path)
    train = [path for path in bundle_paths if re.findall("fft_samp_train", path.name)]
    valid = [path for path in bundle_paths if re.findall("fft_samp_valid", path.name)]

    # Create train and valid datasets
    train_ds = h5_dataset(train, tar_fourier=fourier)
    valid_ds = h5_dataset(valid, tar_fourier=fourier)

    # Create databunch with defined batchsize
    bs = 256
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    # Define model
    arch = getattr(architecture, arch)()

    # Define resize based on the length of an input image
    img = train_ds[0][0]
    mnist_view = view_tfm(2, int(np.sqrt(img.shape[1])), int(np.sqrt(img.shape[1])))

    # make normalisation
    norm = normalize_tfm(norm_path)

    # get model name for recording in LoggerCallback
    model_name = model_path.split("models/")[-1].split("/")[0]

    # Define callback functions
    cbfs = [
        Recorder,
        partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
        CudaCallback,
        partial(BatchTransformXCallback, norm),
        partial(BatchTransformXCallback, mnist_view),
        partial(SaveCallback, model_path=model_path),
        partial(LoggerCallback, model_name=model_name),
    ]

    if loss_func == "feature_loss":
        loss_func = init_feature_loss()
    elif loss_func == "l1":
        loss_func = nn.L1Loss()
    elif loss_func == "mse":
        loss_func = nn.MSELoss()
    else:
        print("\n No matching loss function! Exiting. \n")
        sys.exit(1)

    # Combine model and data in learner
    learn = get_learner(
        data,
        arch,
        lr=lr,
        opt_func=torch.optim.Adam,
        cb_funcs=cbfs,
        loss_func=loss_func,
    )

    # use pre-trained model if asked
    if pretrained is True:
        # Load model
        load_pre_model(learn, pretrained_model)

    # Print model architecture
    # print(learn.model, "\n")

    # Train the model, make it possible to stop at any given time
    try:
        learn.fit(num_epochs)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)")
        save = str(input())
        if save == "y":
            # saving the model if asked
            print("Saving the model after epoch {}".format(learn.epoch))
            save_model(learn, model_path)

            # Plot loss
            plot_loss(learn, model_path)

            # Plot input, prediction and true image if asked
            if inspection is True:
                evaluate_model(valid_ds, learn.model, norm_path)
                plt.savefig(
                    "inspection_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01
                )
        else:
            print("Stopping after epoch {}".format(learn.epoch))
        sys.exit(1)

    # Save model
    save_model(learn, model_path)

    # Plot loss
    plot_loss(learn, model_path)

    # Plot input, prediction and true image if asked
    if inspection is True:
        evaluate_model(valid_ds, learn.model, norm_path, nrows=10)
        plt.savefig(
            "inspection_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01
        )


if __name__ == "__main__":
    main()
