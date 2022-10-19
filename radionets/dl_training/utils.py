import sys
import click
from pathlib import Path
from radionets.dl_framework.data import load_data, DataBunch, get_dls
import radionets.dl_framework.architecture as architecture
from radionets.dl_framework.inspection import plot_loss
from radionets.dl_framework.model import save_model
from radionets.evaluation.train_inspection import create_inspection_plots


def create_databunch(data_path, fourier, source_list, batch_size):
    # Load data sets
    train_ds = load_data(data_path, "train", source_list=source_list, fourier=fourier)
    valid_ds = load_data(data_path, "valid", source_list=source_list, fourier=fourier)

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

    train_conf["quiet"] = config["mode"]["quiet"]
    train_conf["gpu"] = config["mode"]["gpu"]

    train_conf["comet_ml"] = config["logging"]["comet_ml"]
    train_conf["plot_n_epochs"] = config["logging"]["plot_n_epochs"]
    train_conf["project_name"] = config["logging"]["project_name"]
    train_conf["scale"] = config["logging"]["scale"]

    train_conf["bs"] = config["hypers"]["batch_size"]
    train_conf["lr"] = config["hypers"]["lr"]

    train_conf["fourier"] = config["general"]["fourier"]
    train_conf["amp_phase"] = config["general"]["amp_phase"]
    train_conf["arch_name"] = config["general"]["arch_name"]
    train_conf["loss_func"] = config["general"]["loss_func"]
    train_conf["num_epochs"] = config["general"]["num_epochs"]
    train_conf["inspection"] = config["general"]["inspection"]
    train_conf["separate"] = False
    train_conf["format"] = config["general"]["output_format"]
    train_conf["switch_loss"] = config["general"]["switch_loss"]
    train_conf["when_switch"] = config["general"]["when_switch"]

    train_conf["param_scheduling"] = config["param_scheduling"]["use"]
    train_conf["lr_start"] = config["param_scheduling"]["lr_start"]
    train_conf["lr_max"] = config["param_scheduling"]["lr_max"]
    train_conf["lr_stop"] = config["param_scheduling"]["lr_stop"]
    train_conf["lr_ratio"] = config["param_scheduling"]["lr_ratio"]

    train_conf["source_list"] = config["general"]["source_list"]
    return train_conf


def check_outpath(model_path, train_conf):
    path = Path(model_path)
    exists = path.exists()
    if exists:
        if train_conf["quiet"]:
            click.echo("Overwriting existing model file!")
            path.unlink()
        else:
            if click.confirm(
                "Do you really want to overwrite existing model file?", abort=True
            ):
                click.echo("Overwriting existing model file!")
                path.unlink()


def define_arch(arch_name, img_size):
    if (
        "filter_deep" in arch_name
        or "resnet" in arch_name
        or "Uncertainty" in arch_name
    ):
        arch = getattr(architecture, arch_name)(img_size)
    else:
        arch = getattr(architecture, arch_name)()
    return arch


def pop_interrupt(learn, train_conf):
    if click.confirm("KeyboardInterrupt, do you want to save the model?", abort=False):
        model_path = train_conf["model_path"]
        # save model
        print(f"Saving the model after epoch {learn.epoch}")
        save_model(learn, model_path)

        # plot loss
        plot_loss(learn, model_path)

        # Plot input, prediction and true image if asked
        if train_conf["inspection"]:
            create_inspection_plots(learn, train_conf)
    else:
        print(f"Stopping after epoch {learn.epoch}")
    sys.exit(1)


def end_training(learn, train_conf):
    # Save model
    save_model(learn, Path(train_conf["model_path"]))

    # Plot loss
    plot_loss(learn, Path(train_conf["model_path"]))
