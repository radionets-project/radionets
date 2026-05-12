"""
raionets - Deep Learning-based imaging in radio interferometry

Licensed under a MIT style license - see LICENSE
"""

import rich_click as click
from matplotlib import colormaps
from rich.traceback import install
from torch.serialization import add_safe_globals

from radionets.core.logging import _setup_logger
from radionets.plotting._puor import PuOr, PuOr_r

from .version import __version__

__all__ = ["__version__"]

colormaps.register(cmap=PuOr)
colormaps.register(cmap=PuOr_r)

install(show_locals=False)

_setup_logger(namespace="radionets")


click.rich_click.MAX_WIDTH = 100
click.rich_click.STYLE_COMMANDS_TABLE_COLUMN_WIDTH_RATIO = (1, 7)
click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = (
    "Try running the '--help' flag for more information."
)
click.rich_click.ERRORS_EPILOGUE = "To find out more, visit [link=https://pyvisgen.readthedocs.io]https://pyvisgen.readthedocs.io[/link]"
