import click
from pathlib import Path
from dl_framework.data import load_data, DataBunch, get_dls
import dl_framework.architectures as architecture


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

    train_conf["bs"] = config["hypers"]["batch_size"]

    train_conf["fourier"] = config["general"]["fourier"]
    train_conf["arch_name"] = config["general"]["arch_name"]
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
