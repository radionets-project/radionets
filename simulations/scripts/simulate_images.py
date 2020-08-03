import click
import toml


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
def main(configuration_path):
    """
    Generate monte carlo simulation data sets to train and test neural networks for
    reconstruction of radio interferometric data.

    CONFIGURATION_PATH: Path to the config toml file
    """
    config = toml.load(configuration_path)
    click.echo(config['paths']['out_path'])


if __name__ == '__main__':
    main()
