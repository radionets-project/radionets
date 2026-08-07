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
    unit=r"$\mathrm{Jy \cdot px^{-1}}$",
    orientation="vertical",
    location="right",
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
    cax = divider.append_axes(location, size="5%", pad=0.05)
    ax.set_title(title)

    if phase:
        cbar = fig.colorbar(
            image,
            cax=cax,
            orientation=orientation,
            label="Phase / rad",
            location=location,
        )
        cbar.set_ticks(
            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            labels=[r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"],
        )
    elif unc:
        cbar = fig.colorbar(
            image,
            cax=cax,
            orientation=orientation,
            label=rf"$\sigma$ \:/\: {unit}",
            location=location,
        )
    else:
        cbar = fig.colorbar(
            image,
            cax=cax,
            orientation=orientation,
            label=rf"$\mathrm{{Flux Density}} \:/\: {unit}$",
            location=location,
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
