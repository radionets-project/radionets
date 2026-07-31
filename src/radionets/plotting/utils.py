from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from numpy.typing import ArrayLike


def set_cbar(
    fig: Figure,
    ax: Axes,
    image: AxesImage,
    title: str,
    phase: bool = False,
    unc: bool = False,
) -> None:
    """Create nice colorbars with bigger label size
    for every axis in a subplot. Also use ticks for the phase.

    Parameters
    ----------
    fig : :class:`~matplotlib.figure.Figure`
        Current figure object.
    ax : :class:`~matplotlib.axes.Axes`
        Current axis object.
    image : :class:`~matplotlib.image.AxesImage`
        Plotted image.
    title : str
        Title of subplot.
    phase : bool, optional
        If ``True``, sets colorbar to units of π. Default: False
    unc : bool, optional
        If ``True``, sets colorbar label to uncertainty.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(title)

    if phase:
        cbar = fig.colorbar(image, cax=cax, orientation="vertical", label="Phase / rad")
        cbar.set_ticks(
            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            labels=[r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"],
        )
    elif unc:
        cbar = fig.colorbar(
            image,
            cax=cax,
            orientation="vertical",
            label=r"$\sigma$ / $\mathrm{Jy \cdot px^{-1}}$",
        )
    else:
        cbar = fig.colorbar(
            image,
            cax=cax,
            orientation="vertical",
            label=r"$\mathrm{Flux \ density / Jy \cdot px^{-1}}$",
        )

    return cbar


def get_vmin_vmax(image: ArrayLike):
    """Check whether the absolute of the maximum or the minimum is bigger.
    If the minimum is bigger, return value with negative sign. Otherwise return
    maximum.

    Parameters
    ----------
    image : array_like
        Input image.
    Returns
    -------
    float
        Negative minimum value or maximum value otherwise.
    """
    a = -image.min() if np.abs(image.min()) > np.abs(image.max()) else image.max()
    return a
