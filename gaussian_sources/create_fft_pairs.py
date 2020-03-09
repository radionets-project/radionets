import click
import numpy as np
from tqdm import tqdm
from dl_framework.data import (
    open_bundle,
    save_fft_pair,
    get_bundles,
    split_amp_phase,
)
from simulations.uv_simulations import sample_freqs
from simulations.gaussian_simulations import add_noise
import re


@click.command()
@click.argument("in_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("out_path", type=click.Path(exists=False, dir_okay=True))
@click.argument("antenna_config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-mode", type=str, required=True)
@click.option("-amp_phase", type=bool, required=True)
@click.option("-fourier", type=bool, required=True)
@click.option("-samp", type=bool, required=False)
@click.option("-size", type=int, required=True)
@click.option("-specific_mask", type=bool)
@click.option("-lon", type=float, required=False)
@click.option("-lat", type=float, required=False)
@click.option("-steps", type=float, required=False)
@click.option("-noise", type=bool)
@click.option("-preview", type=bool)
def main(
    in_path,
    out_path,
    antenna_config_path,
    mode,
    amp_phase,
    fourier,
    size,
    samp=True,
    specific_mask=False,
    lon=None,
    lat=None,
    steps=None,
    noise=False,
    preview=False,
):
    """
    get list of bundles
    get len of all all bundles
    split bundles into train and valid (factor 0.2?)
    for every bundle
        calculate fft -> create fft pairs
        save to new h5 file
    tagg train and valid in filename
    """
    bundles = get_bundles(in_path)
    bundles = [
        path
        for path in bundles
        if re.findall("gaussian_sources_{}".format(mode), path.name)
    ]

    for path in tqdm(bundles):
        bundle = open_bundle(path)
        images = bundle.copy()
        if noise is True:
            images = add_noise(images, preview=preview)
        bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in images])
        if amp_phase:
            amp, phase = split_amp_phase(bundle_fft)
            amp = np.log10(amp + 1)
            bundle_fft = np.stack((amp, phase), axis=1)
        copy = bundle_fft.copy()
        if samp is True:
            if specific_mask is True:
                bundle_samp = np.array(
                    [
                        sample_freqs(
                            img, antenna_config_path, size, lon, lat, steps, test=True,
                        )
                        for img in copy
                    ]
                )
            else:
                bundle_samp = np.array(
                    [sample_freqs(img, antenna_config_path, size=size) for img in copy]
                )
        out = out_path + path.name.split("_")[-1]
        if fourier:
            save_fft_pair(out, bundle_samp, bundle_fft)
        else:
            save_fft_pair(out, bundle_samp, images)


if __name__ == "__main__":
    main()
