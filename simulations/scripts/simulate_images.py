import click
import toml
from create_fft_images import create_fft_images
from simulations.scripts.utils import check_outpath, read_config


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Generate monte carlo simulation data sets to train and test neural networks for
    reconstruction of radio interferometric data.

    configuration_path: Path to the config toml file
    """
    config = toml.load(configuration_path)

    # check out path and look for existing files
    out_path = config["paths"]["out_path"]
    sim_fft, sim_sampled = check_outpath(
        out_path, data_format=config["paths"]["data_format"]
    )

    # declare source options
    sim_conf = read_config(config)

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
