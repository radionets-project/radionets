import click
import re
from tqdm import tqdm
import numpy as np
from mnist_cnn.scripts.utils import adjust_outpath
from simulations.uv_simulations import sample_freqs
from dl_framework.data import get_bundles, open_fft_pair, save_fft_pair, split_real_imag


@click.command()
@click.argument("data_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("antenna_config_path", type=click.Path(exists=True, dir_okay=True))
@click.option("-fourier", type=bool, default=False)
@click.option("-real_imag", type=bool, default=True, required=False)
@click.option("-specific_mask", type=bool)
@click.option("-lon", type=float, required=False)
@click.option("-lat", type=float, required=False)
@click.option("-steps", type=int, required=False)
def main(
    data_path,
    out_path,
    antenna_config_path,
    specific_mask=False,
    lon=None,
    lat=None,
    steps=None,
    fourier=False,
    real_imag=True,
):
    """
    Use mnist_fft and create a sampled data set for train, valid and test.
    Specific or random masks can be used.
    Input data can be split into 2 channel: real and imaginary part.
    To train in Fourier space, target image can be saved in frequency space.

    Parameters
    ----------
    data_path: str
        path to data directory
    out_path: str
        path to save sampled data
    antenna_config_path: str
        path to antenna config file

    Options
    -------
    fourier: bool
        if True: save target in frequency space, default is False
    real_imag: bool
        if True: split input data into a real and imag channel, default is True
    specific_mask: bool
        if True: use specific mask for all images
    lon: float
        start longitude for specific mask
    lat: float
        start latitude for specific mask
    steps: int
        number of steps for specific mask
    """
    modes = ["train", "valid", "test"]

    for mode in modes:
        print(f"Sampling {mode} data set.")
        bundles = get_bundles(data_path)
        bundles = [
            path
            for path in bundles
            if re.findall("fft_bundle_{}".format(mode), path.name)
        ]

        for path in tqdm(bundles):
            freq, img = open_fft_pair(path)
            size = freq.shape[-1]
            if real_imag:
                real, imag = split_real_imag(freq)
                freq = np.stack((real, imag), axis=1)
            if specific_mask is True:
                freq_samp = sample_freqs(
                    freq, antenna_config_path, size, lon, lat, steps
                )
            else:
                freq_samp = sample_freqs(
                    freq, antenna_config_path, size=size, specific_mask=False
                )

            out = adjust_outpath(out_path, "/fft_bundle_samp_" + mode)
            if fourier:
                save_fft_pair(out, freq_samp, freq)
            else:
                save_fft_pair(out, freq_samp, img)


if __name__ == "__main__":
    main()
