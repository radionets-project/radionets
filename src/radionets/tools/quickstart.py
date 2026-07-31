import sysconfig
from pathlib import Path

import rich_click as click
import toml
from rich.pretty import pretty_repr

from radionets import __version__
from radionets.core.logging import _setup_logger


@click.command()
@click.argument(
    "config_path",
    type=click.Path(dir_okay=True),
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["train", "eval", "inference", "plot"]),
    default="train",
    help="""What config file to create at config_path.
        Valid are {train, eval, inference, plot}. Default: train""",
)
@click.option(
    "-y",
    "--yes",
    "overwrite",
    type=bool,
    is_flag=True,
    help="Overwrite file if it already exists.",
)
def main(
    config_path: str | Path,
    mode: str = "train",
    overwrite: bool = False,
) -> None:
    """Quickstart CLI tool for radionets. Creates
    a copy of the default train or eval configuration
    file at the specified path.

    Parameters
    ----------
    config_path : str or Path
        Path to write the config to.
    mode : str, optional
        Determines the type of config. One of 'train',
        'eval', 'inference' or 'plot' are valid. Default: 'train'
    overwrite : bool, optional
        If ``True``, overwrites the config file if it already
        exists. Default: ``False``

    Notes
    -----
    If a directory is given, this tool will create
    a file called 'radionets_default_{train,eval,inference,plot}_config.toml'
    inside that directory.

    Raises
    ------
    ValueError
        If mode is not one of 'train', 'eval', 'inference' or 'plot'.
    """
    if mode not in ["train", "eval", "inference", "plot"]:
        raise ValueError(
            "Unknown mode: Expected one of {train, eval, inference, plot}."
        )

    log = _setup_logger(namespace=__name__, tracebacks_suppress=[click])

    msg = f"This is the radionets [blue]v{__version__}[/] quickstart tool"
    log.info(msg, extra={"markup": True, "highlighter": None})
    log.info((len(msg) - len("[blue][/]")) * "=")

    if isinstance(config_path, str):
        config_path = Path(config_path)

    root = sysconfig.get_path("data", sysconfig.get_default_scheme())

    default_config_path = Path(
        root + f"/share/configs/radionets_default_{mode}_config.toml"
    )

    log.info(f"Loading default radionets {mode} configuration...")
    with open(default_config_path) as f:
        default_config = toml.load(f)

    log.info(pretty_repr(default_config))

    if config_path.is_dir():
        config_path /= f"radionets_default_{mode}_config.toml"

    # write_file is used below; the following if statement acts as
    # a switch, toggling write_file to False if the user does not
    # wish to overwrite
    write_file = True
    if config_path.is_file() and not overwrite:
        log.info("")
        write_file = click.confirm(
            f"{config_path} already exists! Overwrite?", default=False
        )

    if write_file:
        with open(config_path, "w") as f:
            toml.dump(default_config, f)

        log.info(
            f"Configuration file was successfully written to {config_path.absolute()}",
        )
    else:
        log.warning("No output file was written!")


if __name__ == "__main__":
    main()
