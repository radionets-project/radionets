import click
from simulations.uv_simulations import sample_freqs
from pointsources.utils import write_h5, get_h5_data
from tqdm import tqdm
import numpy as np


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("antenna_config_path", type=click.Path(exists=True, dir_okay=True))
@click.option("-train", type=bool)
@click.option("-specific_mask", type=bool)
@click.option("-lon", type=float, required=False)
@click.option("-lat", type=float, required=False)
@click.option("-steps", type=float, required=False)
def main(
    data_path,
    out_path,
    antenna_config_path,
    specific_mask=False,
    lon=None,
    lat=None,
    steps=None,
    train=True,
):
    if train is True:
        x, y = get_h5_data(data_path, columns=["x_train", "y_train"])
    else:
        x, y = get_h5_data(data_path, columns=["x_valid", "y_valid"])

    if specific_mask is True:
        x_samp = [
            sample_freqs(i.reshape(1, 64, 64), antenna_config_path, 64, lon, lat, steps)
            for i in tqdm(x)
        ]
    else:
        x_samp = [
            sample_freqs(i.reshape(1, 64, 64), antenna_config_path, 64) for i in tqdm(x)
        ]
    y_samp = y

    x_samp = np.array(
        [np.abs(np.fft.ifft2(np.fft.ifftshift(x_samp[i]))) for i in range(len(x_samp))]
    )
    x_samp = x_samp.reshape(-1, 4096)

    if train is True:
        write_h5(out_path, x_samp, y_samp, "x_train", "y_train")
    else:
        write_h5(out_path, x_samp, y_samp, "x_valid", "y_valid")


if __name__ == "__main__":
    main()
