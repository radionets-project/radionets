import click
import toml
from create_fft_images import create_fft_images
from simulations.scripts.utils import check_outpath


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Generate monte carlo simulation data sets to train and test neural networks for
    reconstruction of radio interferometric data.

    CONFIGURATION_PATH: Path to the config toml file
    """
    config = toml.load(configuration_path)

    # check out path and look for existing files
    out_path = config["paths"]["out_path"]
    sim_fft, sim_sampled = check_outpath(
        out_path, data_format=config["paths"]["data_format"]
    )

    # declare source options
    sim_conf = {}
    sim_conf['out_path'] = out_path
    if config["mnist"]["simulate"]:
        click.echo("Create fft_images from mnist data set!")

        sim_conf["type"] = "mnist"
        sim_conf["resource"] = config["mnist"]["resource"]
    # else:
    #     click.echo("Create fft_images from simulated data set!")

    #     points = config["pointsources"]["simulate"]
    #     if points:
    #         num_sources_points = config["pointsources"]["num_sources"]

    #     pointg = config["pointlike_gaussians"]["simulate"]
    #     if pointg:
    #         num_sources_pointg = config["pointlike_gaussians"]["num_sources"]

    #     extendedg = config["extended_gaussians"]["simulate"]
    #     if extendedg:
    #         num_components = config["pointlike_gaussians"]["num_components"]

    sim_conf["num_bundles"] = config["image_options"]["num_bundles"]
    sim_conf["bundle_size"] = config["image_options"]["bundle_size"]
    sim_conf["img_size"] = config["image_options"]["img_size"]
    sim_conf["noise"] = config["image_options"]["noise"]
    print(sim_conf)

    # start simulations
    if sim_fft is True:
        click.echo("Starting simulation of fft_files!")
        create_fft_images(sim_conf)

    if sim_sampled is True:
        click.echo("Start sampling fft_files!")

    # save simulations


"""
block 1
check if out_path exists -> create directory
check if out_path is empty -> overwrite old simulations?
keep fft_files?

block 2
configure simulation options
mnist (need resource) / gaussian sources / pointsources

block 3
simulate source images
fft
sample

block 4
save simulated images in bundles to out_path
"""


if __name__ == "__main__":
    main()
