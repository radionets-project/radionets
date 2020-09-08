import click
import toml
from dl_training.utils import read_config, check_outpath, create_databunch


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Start DNN training with options specified in configuration file.

    configuration_path: Path to the config toml file
    """
    config = toml.load(configuration_path)
    train_conf = read_config(config)

    click.echo("\n Simulation config:")
    print(train_conf, "\n")

    # check out path and look for existing files
    # check_outpath(train_conf)

    # create databunch
    data = create_databunch(
        data_path=train_conf["data_path"],
        fourier=train_conf["fourier"],
        batch_size=train_conf["bs"],
    )

    print(data)

"""
1) load data

"""


if __name__ == "__main__":
    main()
