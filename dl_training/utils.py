import click
from pathlib import Path
from dl_framework.data import load_data, DataBunch, get_dls


def create_databunch(data_path, fourier, batch_size):
    # Load data sets
    train_ds = load_data(data_path, "train", fourier=fourier)
    valid_ds = load_data(data_path, "valid", fourier=fourier)

    # Create databunch with defined batchsize
    bs = batch_size
    data = DataBunch(*get_dls(train_ds, valid_ds, bs))
    return data


def read_config(config):
    sim_conf = {}
    sim_conf["data_path"] = config["paths"]["data_path"]

    sim_conf["bs"] = config["hypers"]["batch_size"]

    sim_conf["fourier"] = config["general"]["fourier"]
    sim_conf["arch_name"] = config["general"]["arch_name"]
    return sim_conf


def check_outpath(model_path):
    path = Path(model_path)
    exists = path.exists()
    print(exists)
    if exists:
        if click.confirm(
            "Do you really want to overwrite existing model?", abort=False
        ):
            click.echo("Overwriting existing model!")
            path.unlink()
        else:
            continue
    return None
