import rich_click as click

from radionets.io import PlottingConfig
from radionets.plotting import Hist


@click.command()
@click.argument(
    "config_path", default=None, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-i",
    "--input_paths",
    default=None,
    type=click.Path(exists=True, dir_okay=True),
    multiple=True,
)
@click.option(
    "-o",
    "--output_path",
    default=None,
    type=click.Path(exists=True, dir_okay=True),
)
def main(config_path, input_paths, output_path):
    """Plotting pipeline for radionets evaluation data."""
    if not config_path:
        if not input_paths and not output_path:
            raise RuntimeError(
                "Please specify the input paths and the output path "
                "if no config path is provided!"
            )
        config = PlottingConfig(
            paths=dict(data_path=input_paths, save_path=output_path)
        )
    else:
        config = PlottingConfig.from_toml(config_path)

    hist = Hist(config)
    hist.plot()
