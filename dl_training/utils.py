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

    sim_conf["fourier"] = config["data"]["fourier"]
    return sim_conf


def check_outpath():
    return None
