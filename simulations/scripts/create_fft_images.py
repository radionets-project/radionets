from simulations.scripts.mnist import mnist_fft
from simulations.scripts.gaussians import simulate_gaussian_sources


def create_fft_images(sim_conf):
    """
    Create fft source images and save them to h5 files.

    Parameters
    ----------
    sim_conf : dict
        dict holding simulation parameters
    """
    if sim_conf["type"] == "mnist":
        mnist_fft(
            data_path=sim_conf["resource"],
            out_path=sim_conf["out_path"],
            size=sim_conf["img_size"],
            bundle_size=sim_conf["bundle_size"],
            noise=sim_conf["noise"],
        )
    if sim_conf["type"] == "gaussians":
        simulate_gaussian_sources(
            out_path=sim_conf["out_path"],
            num_bundles=sim_conf["num_bundles"],
            bundle_size=sim_conf["bundle_size"],
            img_size=sim_conf["img_size"],
            num_comp_ext=sim_conf["num_components"],
            num_pointlike=sim_conf["num_pointlike_gaussians"],
            num_pointsources=sim_conf["num_pointsources"],
            noise=sim_conf["noise"],
        )
