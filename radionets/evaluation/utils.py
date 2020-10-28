import numpy as np


def reshape_2d(array):
    """
    Reshape 1d arrays into 2d ones.

    Parameters
    ----------
    array: 1d array
        input array

    Returns
    -------
    array: 2d array
        reshaped array
    """
    shape = [int(np.sqrt(array.shape[-1]))] * 2
    return array.reshape(-1, *shape)


def make_axes_nice(fig, ax, im, title, phase=False):
    """Create nice colorbars with bigger label size for every axis in a subplot.
    Also use ticks for the phase.
    Parameters
    ----------
    fig : figure object
        current figure
    ax : axis object
        current axis
    im : ndarray
        plotted image
    title : str
        title of subplot
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(title, fontsize=20)

    if phase:
        cbar = fig.colorbar(
            im, cax=cax, orientation="vertical", ticks=[-np.pi, 0, np.pi]
        )
    else:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")

    cbar.set_label("Intensity / a.u.", size=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.get_offset_text().set_fontsize(20)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    if phase:
        # set ticks for colorbar
        cbar.ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])


def reshape_split(img):
    """
    reshapes and splits the the given image based on the image shape.
    If the image is based on two channels, it reshapes with shape
    (1, 2, img_size, img_size), otherwise with shape (img_size, img_size).
    Afterwards, the array is splitted in real and imaginary part if given.
    Parameters
    ----------
    img : ndarray
        image
    Returns
    ----------
    img_reshaped : ndarry
        contains the reshaped image in a numpy array
    img_real, img_imag: ndarrays
        contain the real and the imaginary part
    -------
    """
    if img.shape[0] == 1:
        img_size = int(np.sqrt(img.shape[0]))
        img_reshaped = img.reshape(img_size, img_size)

        return img_reshaped

    else:
        img_size = int(np.sqrt(img.shape[0] / 2))
        img_reshaped = img.reshape(1, 2, img_size, img_size)
        img_real = img_reshaped[0, 0, :]
        img_imag = img_reshaped[0, 1, :]

        return img_real, img_imag


def check_vmin_vmax(inp):
    """
    Check wether the absolute of the maxmimum or the minimum is bigger.
    If the minimum is bigger, return value with minus. Otherwise return
    maximum.
    Parameters
    ----------
    inp : float
        input image
    Returns
    -------
    float
        negative minimal or maximal value
    """
    if np.abs(inp.min()) > np.abs(inp.max()):
        a = -inp.min()
    else:
        a = inp.max()
    return a
