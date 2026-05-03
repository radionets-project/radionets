"""
raionets - Deep Learning-based imaging in radio interferometry

Licensed under a MIT style license - see LICENSE
"""

from fastcore.foundation import L
from matplotlib import colormaps
from rich.traceback import install
from torch.serialization import add_safe_globals

from radionets.core.logging import _setup_logger
from radionets.plotting._puor import PuOr, PuOr_r

from .version import __version__

__all__ = ["__version__"]

colormaps.register(cmap=PuOr)
colormaps.register(cmap=PuOr_r)

add_safe_globals([L])

install(show_locals=False)

_setup_logger(namespace="radionets")
