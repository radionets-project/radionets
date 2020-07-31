import sys
import click
from functools import partial
from pathlib import Path
import dl_framework.architectures as architecture
from dl_framework.callbacks import normalize_tfm
from dl_framework.model import load_pre_model, save_model
from dl_framework.data import get_dls, DataBunch, load_data
from dl_framework.inspection import eval_model, plot_loss, get_images, reshape_2d
from dl_framework.learner import define_learner
from mnist_cnn.scripts.visualize import plot_results
from dl_framework.callbacks import (
    Recorder,
    SaveCallback,
)


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("model_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("arch", type=str)
@click.argument("norm_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("num_epochs", type=int)
@click.argument("lr", type=float)
@click.argument("loss_func", type=str)
@click.argument("batch_size", type=int)
@click.argument(
    "pretrained_model", type=click.Path(exists=True, dir_okay=True), required=False
)
@click.option(
    "-fourier",
    type=bool,
    required=False,
    help="true, if target variables get fourier transformed",
)
@click.option(
    "-pretrained", type=bool, required=False, help="use of a pretrained model"
)
@click.option(
    "-test", type=bool, default=False, required=False, help="Disable logger in tests"
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
    batch_size,
    fourier=False,
    pretrained=False,
    pretrained_model=None,
    inspection=False,
    test=False,
):
    """
    Train neural network with training and validation data.
    Progress can be visualized with test data set.

    Parameters
    ----------
    data_path: str
        path to data directory
    model_path:
        path to save model
    arch: str
        name of architecture
    norm_path: str
        path to normalization factors
    num_epochs: int
        number of training epochs
    lr: float
        learning rate
    loss_func: str
        name of loss function
    batch_size: int
        number of images in one batch
    pretrained_model: str
        path to model, when using option 'pretrained'

    Options
    -------
    fourier: bool
        if True: train in Fourier space, default is False
    pretrained: bool
        if True: load pretrained model before training, default is False
    inspection: bool
        if True: create inspection plot after training, default is False
    test: bool
        if True: load a 'smaller' learner, use for test cases, default is False
    """
    # Load data and create train and valid datasets
    train_ds = load_data(data_path, "train", fourier=False)
    valid_ds = load_data(data_path, "valid", fourier=False)

    # Create databunch with defined batchsize
    bs = batch_size
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))

    # Define model
    arch = getattr(architecture, arch)()

    # Define normalisation
    norm = normalize_tfm(norm_path)

    # Define model name for recording in LoggerCallback
    model_name = model_path.split("models/")[-1].split("/")[0]

    # Define learner
    learn = define_learner(
        data,
        arch,
        norm,
        loss_func,
        lr=lr,
        model_name=model_name,
        model_path=model_path,
        test=test,
    )

    # Load pre-trained model if asked
    if pretrained is True:
        load_pre_model(learn, pretrained_model)

    # Train the model, make it possible to stop at any given time
    try:
        learn.fit(num_epochs)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)")
        save = str(input())
        if save == "y":
            # Saving the model if asked
            print("Saving the model after epoch {}".format(learn.epoch))
            save_model(learn, model_path)

            # Plot loss
            plot_loss(learn, model_path)

            # Plot input, prediction and true image if asked
            if inspection is True:
                test_ds = load_data(data_path, "test", fourier=False)
                img_test, img_true = get_images(test_ds, 5, norm_path)
                pred = eval_model(img_test, learn.model)
                out_path = Path(model_path).parent
                plot_results(
                    img_test,
                    reshape_2d(pred),
                    reshape_2d(img_true),
                    out_path,
                    save=True,
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
        test_ds = load_data(data_path, "test", fourier=False)
        img_test, img_true = get_images(test_ds, 5, norm_path)
        pred = eval_model(img_test, learn.model.cpu())
        out_path = Path(model_path).parent
        plot_results(
            img_test, reshape_2d(pred), reshape_2d(img_true), out_path, save=True
        )


if __name__ == "__main__":
    main()
