from radionets.simulations.gaussians import simulate_gaussian_sources
from radionets.simulations.sampling import sample_frequencies
from radionets.simulations.point_sources import create_point_source_img
import click
from pathlib import Path


def create_fft_images(sim_conf):
    """
    Create fft source images and save them to h5 files.

    Parameters
    ----------
    sim_conf : dict
        dict holding simulation parameters
    """
    if sim_conf["type"] == "gaussians":
        for opt in ["train", "valid", "test"]:
            simulate_gaussian_sources(
                data_path=sim_conf["data_path"],
                option=opt,
                num_bundles=sim_conf["bundles_" + str(opt)],
                bundle_size=sim_conf["bundle_size"],
                img_size=sim_conf["img_size"],
                num_comp_ext=sim_conf["num_components"],
                noise=sim_conf["noise"],
                noise_level=sim_conf["noise_level"],
                source_list=sim_conf["source_list"],
                white_noise=sim_conf["white_noise"],
                mean_real=sim_conf["mean_real"],
                std_real=sim_conf["std_real"],
                mean_imag=sim_conf["mean_imag"],
                std_imag=sim_conf["std_imag"],
            )

    if sim_conf["type"] == "point_sources":
        for opt in ["train", "valid", "test"]:
            create_point_source_img(
                img_size=sim_conf["img_size"],
                bundle_size=sim_conf["bundle_size"],
                num_bundles=sim_conf["bundles_" + str(opt)],
                path=sim_conf["data_path"],
                option=opt,
                extended=sim_conf["add_extended"],
            )


def sample_fft_images(sim_conf):
    """
    check for fft files
    keep fft_files?
    """
    sample_frequencies(
        data_path=sim_conf["data_path"],
        amp_phase=sim_conf["amp_phase"],
        real_imag=sim_conf["real_imag"],
        specific_mask=sim_conf["specific_mask"],
        antenna_config=sim_conf["antenna_config"],
        lon=sim_conf["lon"],
        lat=sim_conf["lat"],
        steps=sim_conf["steps"],
        fourier=sim_conf["fourier"],
        compressed=sim_conf["compressed"],
        interpolation=sim_conf["interpolation"],
        multi_channel=sim_conf["multi_channel"],
        bandwidths=sim_conf["bandwidths"],
        source_type=sim_conf["type"],
    )
    if sim_conf["keep_fft_files"] is not True:
        if click.confirm("Do you really want to delete the fft_files?", abort=False):
            fft = {
                p
                for p in Path(sim_conf["data_path"]).rglob(
                    "*fft*." + str(sim_conf["data_format"])
                )
                if p.is_file()
            }
            [p.unlink() for p in fft]
