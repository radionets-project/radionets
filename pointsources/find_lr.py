from functools import partial

import click

import dl_framework.architectures as architecture
from dl_framework.callbacks import (
    BatchTransformXCallback,
    CudaCallback,
    Recorder_lr_find,
    SaveCallback,
    normalize_tfm,
    view_tfm,
    LR_Find,
)
from dl_framework.learner import get_learner
from dl_framework.model import load_pre_model
from dl_framework.optimizer import (
    AverageGrad,
    AverageSqrGrad,
    StatefulOptimizer,
    StepCount,
    adam_step,
    weight_decay,
)
from mnist_cnn.utils import get_h5_data, write_h5
from preprocessing import DataBunch, get_dls, prepare_dataset
from inspection import plot_lr_loss


@click.command()
@click.argument("train_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("valid_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option("-log", type=bool, required=False, help="use of logarith")
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option("-save", type=bool, required=False, help="save the lr vs loss plot")
def main(
    train_path,
    valid_path,
    model_path,
    arch,
    norm_path,
    log=True,
    pretrained=False,
    pretrained_model=None,
    save=False,
):
    """
    Train the neural network with existing training and validation data.
    TRAIN_PATH is the path to the training data\n
    VALID_PATH ist the path to the validation data\n
    MODEL_PATH is the Path to which the model is saved\n
    ARCH is the name of the architecture which is used\n
    NORM_PATH is the path to the normalisation factors\n
    PRETRAINED_MODEL is the path to a pretrained model, which is
                     loaded at the beginning of the training\n
    """
    # Load data
    x_train, y_train = get_h5_data(train_path, columns=["x_train", "y_train"])
    x_valid, y_valid = get_h5_data(valid_path, columns=["x_valid", "y_valid"])

    # Create train and valid datasets
    train_ds, valid_ds = prepare_dataset(x_train, y_train, x_valid, y_valid, log=log)

    # Create databunch with defined batchsize
    bs = 128
    data = DataBunch(*get_dls(train_ds, valid_ds, bs), c=train_ds.c)

    # First guess for max_iter
    print("\nTotal number of batches ~ ", data.train_ds.x.size(0)*2//bs)

    # Define model
    arch = getattr(architecture, arch)()

    # Define resize for mnist data
    mnist_view = view_tfm(1, 64, 64)

    # make normalisation
    norm = normalize_tfm(norm_path)

    # Define callback functions
    cbfs = [
        partial(LR_Find, max_iter=800, max_lr=0.1),
        Recorder_lr_find,
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
    # Combine model and data in learner
    learn = get_learner(data, arch, 1e-3, opt_func=adam_opt, cb_funcs=cbfs,)

    # use pre-trained model if asked
    if pretrained is True:
        # Load model
        load_pre_model(learn, pretrained_model)

    learn.fit(2)
    if save:
        plot_lr_loss(learn, model_path, skip_last=5)
    else:
        learn.recorder_lr_find.plot(skip_last=5)


if __name__ == "__main__":
    main()
