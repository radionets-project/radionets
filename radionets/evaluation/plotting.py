import numpy as np
import matplotlib.pyplot as plt
from radionets.dl_framework.inspection import reshape_2d, make_axes_nice
from matplotlib.colors import LogNorm
from math import pi


def plot_target(h5_dataset, log=False):
    index = np.random.randint(len(h5_dataset) - 1)
    plt.figure(figsize=(5.78, 3.57))
    target = reshape_2d(h5_dataset[index][1]).squeeze(0)
    if log:
        plt.imshow(target, norm=LogNorm())
    else:
        plt.imshow(target)
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")
    plt.colorbar(label="Intensity / a.u.")


def plot_inp_tar(h5_dataset, fourier=False, amp_phase=False):
    index = np.random.randint(len(h5_dataset) - 1)

    if fourier is False:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.45, 3.57))

        inp1 = h5_dataset[index][0][0]
        lim1 = check_vmin_vmax(inp1)
        im1 = ax1.imshow(inp1, cmap="RdBu", vmin=-lim1, vmax=lim1)
        make_axes_nice(fig, ax1, im1, "Input: real part")

        inp2 = h5_dataset[index][0][1]
        lim2 = check_vmin_vmax(inp2)
        im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-lim2, vmax=lim2)
        make_axes_nice(fig, ax2, im2, "Input: imaginary part")

        tar = reshape_2d(h5_dataset[index][1]).squeeze(0)
        im3 = ax3.imshow(tar, cmap="inferno")
        make_axes_nice(fig, ax3, im3, "Target: source image")

    if fourier is True:
        if amp_phase is False:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.45, 8.92))

            inp1 = h5_dataset[index][0][0]
            lim1 = check_vmin_vmax(inp1)
            im1 = ax1.imshow(inp1, cmap="RdBu", vmin=-lim1, vmax=lim1)
            make_axes_nice(fig, ax1, im1, "Input: real part")

            inp2 = h5_dataset[index][0][1]
            lim2 = check_vmin_vmax(inp2)
            im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-lim2, vmax=lim2)
            make_axes_nice(fig, ax2, im2, "Input: imaginary part")

            tar1 = h5_dataset[index][1][0]
            lim_t1 = check_vmin_vmax(tar1)
            im3 = ax3.imshow(tar1, cmap="RdBu", vmin=-lim_t1, vmax=lim_t1)
            make_axes_nice(fig, ax3, im3, "Target: real part")

            tar2 = h5_dataset[index][1][1]
            lim_t2 = check_vmin_vmax(tar2)
            im4 = ax4.imshow(tar2, cmap="RdBu", vmin=-lim_t2, vmax=lim_t2)
            make_axes_nice(fig, ax4, im4, "Target: imaginary part")

        if amp_phase is True:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14.45, 8.92))

            inp1 = h5_dataset[index][0][0]
            im1 = ax1.imshow(inp1, cmap="inferno")
            make_axes_nice(fig, ax1, im1, "Input: amplitude")

            inp2 = h5_dataset[index][0][1]
            lim2 = check_vmin_vmax(inp2)
            im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-pi, vmax=pi)
            make_axes_nice(fig, ax2, im2, "Input: phase")

            tar1 = h5_dataset[index][1][0]
            im3 = ax3.imshow(tar1, cmap="inferno")
            make_axes_nice(fig, ax3, im3, "Target: amplitude")

            tar2 = h5_dataset[index][1][1]
            im4 = ax4.imshow(tar2, cmap="RdBu", vmin=-pi, vmax=pi)
            make_axes_nice(fig, ax4, im4, "Target: phase")

    fig.tight_layout()


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
