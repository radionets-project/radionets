from pathlib import Path
import click
import os
import sys
import re
import numpy as np
from scipy import interpolate
from radionets.dl_framework.data import (
    get_bundles,
    split_amp_phase,
    split_real_imag,
)


def check_outpath(outpath, data_format, quiet=False):
    """
    Check if outpath exists. Check for existing fft_files and sampled-files.
    Ask to overwrite or reuse existing files.

    Parameters
    ----------
    path : str
        path to out directory

    Returns
    -------
    sim_fft : bool
        flag to enable/disable fft routine
    sim_sampled : bool
        flag to enable/disable sampling routine
    """
    path = Path(outpath)
    exists = path.exists()
    if exists is True:
        fft = {p for p in path.rglob("*fft*." + str(data_format)) if p.is_file()}
        samp = {p for p in path.rglob("*samp*." + str(data_format)) if p.is_file()}
        if fft:
            click.echo("Found existing fft_files!")
            if quiet:
                click.echo("Overwriting old fft_files!")
                [p.unlink() for p in fft]
                [p.unlink() for p in samp]
                sim_fft = True
                sim_sampled = True
                return sim_fft, sim_sampled
            elif click.confirm(
                "Do you really want to overwrite the files?", abort=False
            ):
                click.echo("Overwriting old fft_files!")
                [p.unlink() for p in fft]
                [p.unlink() for p in samp]
                sim_fft = True
                sim_sampled = True
                return sim_fft, sim_sampled
            else:
                click.echo("Using old fft_files!")
                sim_fft = False
        else:
            sim_fft = True
        if samp:
            click.echo("Found existing samp_files!")
            if quiet:
                click.echo("Overwriting old samp_files!")
                [p.unlink() for p in samp]
                sim_sampled = True
            elif click.confirm(
                "Do you really want to overwrite the files?", abort=False
            ):
                click.echo("Overwriting old samp_files!")
                [p.unlink() for p in samp]
                sim_sampled = True
            else:
                click.echo("No new images sampled!")
                sim_sampled = False
                sys.exit()
        else:
            sim_sampled = True
    else:
        Path(path).mkdir(parents=True, exist_ok=False)
        sim_fft = True
        sim_sampled = True
    return sim_fft, sim_sampled


def read_config(config):
    sim_conf = {}
    sim_conf["data_path"] = config["paths"]["data_path"]
    sim_conf["data_format"] = config["paths"]["data_format"]
    if config["gaussians"]["simulate"]:
        click.echo("Create fft_images from gaussian data set! \n")

        sim_conf["type"] = "gaussians"
        sim_conf["num_components"] = config["gaussians"]["num_components"]
        click.echo("Adding extended gaussian sources.")

    if config["point_sources"]["simulate"]:
        click.echo("Create fft_images from point source data set! \n")

        sim_conf["type"] = "point_sources"
        sim_conf["add_extended"] = config["point_sources"]["add_extended"]
        click.echo("Adding point sources.")

    sim_conf["bundles_train"] = config["image_options"]["bundles_train"]
    sim_conf["bundles_valid"] = config["image_options"]["bundles_valid"]
    sim_conf["bundles_test"] = config["image_options"]["bundles_test"]
    sim_conf["bundle_size"] = config["image_options"]["bundle_size"]
    sim_conf["img_size"] = config["image_options"]["img_size"]
    sim_conf["noise"] = config["image_options"]["noise"]
    sim_conf["noise_level"] = config["image_options"]["noise_level"]
    sim_conf["white_noise"] = config["image_options"]["white_noise"]
    sim_conf["mean_real"] = config["image_options"]["mean_real"]
    sim_conf["std_real"] = config["image_options"]["std_real"]
    sim_conf["mean_imag"] = config["image_options"]["mean_imag"]
    sim_conf["std_imag"] = config["image_options"]["std_imag"]

    sim_conf["amp_phase"] = config["sampling_options"]["amp_phase"]
    sim_conf["real_imag"] = config["sampling_options"]["real_imag"]
    sim_conf["source_list"] = config["sampling_options"]["source_list"]
    sim_conf["antenna_config"] = config["sampling_options"]["antenna_config"]
    sim_conf["specific_mask"] = config["sampling_options"]["specific_mask"]
    sim_conf["lon"] = config["sampling_options"]["lon"]
    sim_conf["lat"] = config["sampling_options"]["lat"]
    sim_conf["steps"] = config["sampling_options"]["steps"]
    sim_conf["fourier"] = config["sampling_options"]["fourier"]
    sim_conf["compressed"] = config["sampling_options"]["compressed"]
    sim_conf["keep_fft_files"] = config["sampling_options"]["keep_fft_files"]
    sim_conf["interpolation"] = config["sampling_options"]["interpolation"]
    sim_conf["multi_channel"] = config["sampling_options"]["multi_channel"]
    sim_conf["bandwidths"] = config["sampling_options"]["bandwidths"]
    return sim_conf


def adjust_outpath(path, option, form="h5"):
    """
    Add number to out path when filename already exists.

    Parameters
    ----------
    path: str
        path to save directory
    option: str
        additional keyword to add to path

    Returns
    -------
    out: str
        adjusted path
    """
    counter = 0
    filename = str(path) + (option + "{}." + form)
    while os.path.isfile(filename.format(counter)):
        counter += 1
    out = filename.format(counter)
    return out


def get_fft_bundle_paths(data_path, ftype, mode):
    bundles = get_bundles(data_path)
    bundle_paths = [
        path for path in bundles if re.findall(f"{ftype}_{mode}", path.name)
    ]
    return bundle_paths


def prepare_fft_images(fft_images, amp_phase, real_imag):
    if amp_phase:
        amp, phase = split_amp_phase(fft_images)
        amp = (np.log10(amp + 1e-10) / 10) + 1

        # Test new masking for 511 Pixel pictures
        if amp.shape[1] == 511:
            mask = amp > 0.1
            phase[~mask] = 0
        fft_scaled = np.stack((amp, phase), axis=1)
    else:
        real, imag = split_real_imag(fft_images)
        fft_scaled = np.stack((real, imag), axis=1)
    return fft_scaled


def get_noise(image, scale, mean=0, std=1):
    """
    Calculate random noise values for all image pixels.

    Parameters
    ----------
    image: 2darray
        2d image
    scale: float
        scaling factor to increase noise
    mean: float
        mean of noise values
    std: float
        standard deviation of noise values

    Returns
    -------
    out: ndarray
        array with noise values in image shape
    """
    return np.random.normal(mean, std, size=image.shape) * scale


def add_noise(bundle, noise_level):
    """
    Used for adding noise and plotting the original and noised picture,
    if asked. Using 0.05 * max(image) as scaling factor.

    Parameters
    ----------
    bundle: path
        path to hdf5 bundle file
    noise_level: int
        noise level in percent

    Returns
    -------
    bundle_noised hdf5_file
        bundle with noised images
    """
    bundle_noised = np.array(
        [img + get_noise(img, (img.max() * noise_level / 100)) for img in bundle]
    )
    return bundle_noised


def interpol(img):
    """Interpolates fft sampled amplitude and phase data.
    Parameters
    ----------
    img : array
        array with shape 2,width,heigth
        input image array with amplitude and phase on axis 0
    Returns
    -------
    array
        array with shape 2,width,heigth
        interpolated image array with amplitude and phase on axis 0
    """
    grid_x, grid_y = np.mgrid[0 : len(img[0, 0]) : 1, 0 : len(img[0, 0]) : 1]

    idx_amp = np.nonzero(img[0])
    amp = interpolate.griddata(
        (idx_amp[0], idx_amp[1]), img[0][idx_amp], (grid_x, grid_y), method="nearest"
    )

    img[1][img[1] < 0] = 0
    idx_phase = np.nonzero(img[1])
    phase = interpolate.griddata(
        (idx_phase[0], idx_phase[1]),
        img[1][idx_phase],
        (grid_x, grid_y),
        method="nearest",
    )

    mask = np.ones((len(img[0, 0]), len(img[0, 0])))
    mask[1::2, 1::2] = 0
    mask[::2, ::2] = 0
    for i in range(len(img[0, 0])):
        mask[i, len(img[0, 0]) - 1 - i :] = 1 - mask[i, len(img[0, 0]) - 1 - i :]

    phase_fl = -np.flip(phase, [0, 1])
    phase = phase * mask + phase_fl * (1 - mask)

    return np.array([amp, phase])


def add_white_noise(images, mean_real=25, std_real=1.25, mean_imag=7, std_imag=0.35):
    img_size = images.shape[2]
    noise_real = np.random.normal(
        mean_real, std_real, size=(images.shape[0], img_size, img_size)
    )
    noise_imag = np.random.normal(
        mean_imag, std_imag, size=(images.shape[0], img_size, img_size)
    )
    images.real += noise_real
    images.imag += noise_imag
    return images
