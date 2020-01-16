import sys
from functools import partial

import click
import matplotlib.pyplot as plt

import dl_framework.architectures as architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from dl_framework.callbacks import (
    AvgStatsCallback,
    BatchTransformXCallback,
    CudaCallback,
    ParamScheduler,
    Recorder,
    SaveCallback,
    normalize_tfm,
    view_tfm,
)
from dl_framework.learner import get_learner
from dl_framework.loss_functions import FeatureLoss
from dl_framework.model import load_pre_model
from dl_framework.optimizer import (
    AverageGrad,
    AverageSqrGrad,
    StatefulOptimizer,
    StepCount,
    adam_step,
    weight_decay,
)
from dl_framework.param_scheduling import sched_no
from dl_framework.utils import children
from inspection import evaluate_model, plot_loss
from mnist_cnn.utils import get_h5_data
from preprocessing import DataBunch, get_dls, prepare_dataset
from torchvision.models import vgg16_bn


@click.command()
@click.argument("train_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("valid_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("num_epochs", type=int)
@click.argument("lr", type=float)
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option("-log", type=bool, required=False, help="use of logarith")
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-inspection", type=bool, required=False, help="make an inspection plot")
def main(
    train_path,
    valid_path,
    model_path,
    arch,
    norm_path,
    num_epochs,
    lr,
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
    x_train, y_train = get_h5_data(train_path, columns=["x_train", "y_train"])
    x_valid, y_valid = get_h5_data(valid_path, columns=["x_valid", "y_valid"])

    # Create train and valid datasets
    train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid, log=log)

    # Create databunch with defined batchsize
    bs = 256
    data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)

    # Define model
    arch = getattr(architecture, arch)()

    # Define resize for mnist data
    mnist_view = view_tfm(2, 64, 64)

    # make normalisation
    norm = normalize_tfm(norm_path)

    # Define scheduled learning rate
    sched = sched_no(lr, lr)

    # Define callback functions
    cbfs = [
        Recorder,
        # test for use of multiple Metrics or Loss functions
        partial(AvgStatsCallback, metrics=[nn.MSELoss(), nn.L1Loss()]),
        partial(ParamScheduler, "lr", sched),
        CudaCallback,
        partial(BatchTransformXCallback, norm),
        partial(BatchTransformXCallback, mnist_view),
        SaveCallback,
    ]

    # Define optimiser function
    adam_opt = partial(
        StatefulOptimizer,
        steppers=[adam_step, weight_decay],
        stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()],
    )

    # feature_loss
    ###########################################################################
    vgg_m = vgg16_bn(True).features.cuda().eval()
    for param in vgg_m.parameters():
        param.requires_grad = False
    # requires_grad(vgg_m, False)
    blocks = [
        i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)
    ]
    feat_loss = FeatureLoss(vgg_m, F.l1_loss, blocks[2:5], [5, 15, 2])
    ###########################################################################

    # Combine model and data in learner
    learn = get_learner(
        data, arch, 1e-3, opt_func=adam_opt, cb_funcs=cbfs, loss_func=feat_loss
    )

    # use pre-trained model if asked
    if pretrained is True:
        # Load model
        load_pre_model(learn.model, pretrained_model)

    # Print model architecture
    print(learn.model, "\n")

    # Train the model, make it possible to stop at any given time
    try:
        learn.fit(num_epochs)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)")
        save = str(input())
        if save == "y":
            print("Saving the model after epoch {}".format(learn.epoch))
            state = learn.model.state_dict()
            torch.save(state, model_path)
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
    state = learn.model.state_dict()
    torch.save(state, model_path)

    # Plot loss
    plot_loss(learn, model_path)

    # Plot input, prediction and true image if asked
    if inspection is True:
        evaluate_model(valid_ds, learn.model, norm_path)
        plt.savefig(
            "inspection_plot.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01
        )


if __name__ == "__main__":
    main()
