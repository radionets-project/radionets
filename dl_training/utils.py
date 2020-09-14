import sys
import click
from pathlib import Path
from dl_framework.data import load_data, DataBunch, get_dls
import dl_framework.architectures as architecture
from dl_framework.inspection import plot_loss
from dl_framework.model import save_model


def create_databunch(data_path, fourier, batch_size):
    # Load data sets
    train_ds = load_data(data_path, "train", fourier=fourier)
    valid_ds = load_data(data_path, "valid", fourier=fourier)

    # Create databunch with defined batchsize
    bs = batch_size
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))
    return data


def read_config(config):
    train_conf = {}
    train_conf["data_path"] = config["paths"]["data_path"]
    train_conf["model_path"] = config["paths"]["model_path"]
    train_conf["pre_model"] = config["paths"]["pre_model"]
    train_conf["norm_path"] = config["paths"]["norm_path"]

    train_conf["bs"] = config["hypers"]["batch_size"]
    train_conf["lr"] = config["hypers"]["lr"]

    train_conf["fourier"] = config["general"]["fourier"]
    train_conf["arch_name"] = config["general"]["arch_name"]
    train_conf["loss_func"] = config["general"]["loss_func"]
    train_conf["num_epochs"] = config["general"]["num_epochs"]
    return train_conf


def check_outpath(model_path):
    path = Path(model_path)
    exists = path.exists()
    if exists:
        if click.confirm(
            "Do you really want to overwrite existing model file?", abort=False
        ):
            click.echo("Overwriting existing model file!")
            path.unlink()

    return None


def define_arch(arch_name, img_size):
    if (
        arch_name == "filter_deep"
        or arch_name == "filter_deep_amp"
        or arch_name == "filter_deep_phase"
    ):
        arch = getattr(architecture, arch_name)(img_size)
    else:
        arch = getattr(architecture, arch_name)()
    return arch


def pop_interrupt(learn, model_path):
    print("\nKeyboardInterrupt, do you wanna save the model: yes-(y), no-(n)")
    save = str(input())
    if save == "y":
        # saving the model if asked
        print("Saving the model after epoch {}".format(learn.epoch))
        save_model(learn, model_path)

        # Plot loss
        plot_loss(learn, model_path)

        # Plot input, prediction and true image if asked
        # if inspection is True:
        #     test_ds = load_data(data_path, "test", fourier=False)
        #     img_test, img_true = get_images(test_ds, 5, norm_path)
        #     pred = eval_model(img_test, learn.model)
        #     out_path = Path(model_path).parent
        #     plot_results(
        #         img_test,
        #         reshape_2d(pred),
        #         reshape_2d(img_true),
        #         out_path,
        #         save=True,
            # )
    else:
        print("Stopping after epoch {}".format(learn.epoch))
    sys.exit(1)
