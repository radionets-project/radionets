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
@click.option("-fourier", type=bool)
@click.option("-real_imag", type=bool, default=False, required=False)
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
    fourier=False,
    real_imag=False,
):
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
            bundle = np.asarray(open_fft_pair(path))
            freq, img = bundle[0], bundle[1]
            size = bundle.shape[-1]
            if real_imag:
                real, imag = split_real_imag(freq)
                freq = np.stack((real, imag), axis=1)
            copy = freq.copy()

            if specific_mask is True:
                freq_samp = np.array(
                    [
                        sample_freqs(image, antenna_config_path, size, lon, lat, steps)
                        for image in copy
                    ]
                )
            else:
                freq_samp = np.array(
                    [
                        sample_freqs(image, antenna_config_path, size=size)
                        for image in copy
                    ]
                )

            out = adjust_outpath(out_path, "/fft_bundle_samp_" + mode)
            if fourier:
                save_fft_pair(out, freq_samp, freq)
            else:
                save_fft_pair(out, freq_samp, img.real)


if __name__ == "__main__":
    main()
