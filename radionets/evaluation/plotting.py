import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from radionets.simulations.utils import adjust_outpath
from tqdm import tqdm
from radionets.evaluation.utils import (
    reshape_2d,
    make_axes_nice,
    check_vmin_vmax,
    pad_unsqueeze,
    round_n_digits,
)
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.dynamic_range import calc_dr, get_boxsize
from radionets.evaluation.blob_detection import calc_blobs
from radionets.evaluation.contour import compute_area_difference
from pytorch_msssim import ms_ssim
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib as mpl

# make nice Latex friendly plots
# mpl.use("pgf")
# mpl.rcParams.update(
#     {
#         "font.size": 12,
#         "font.family": "sans-serif",
#         "text.usetex": True,
#         "pgf.rcfonts": False,
#         "pgf.texsystem": "lualatex",
#     }
# )


def create_OrBu():
    top = cm.get_cmap("Blues_r", 128)
    bottom = cm.get_cmap("Oranges", 128)
    white = np.array([256 / 256, 256 / 256, 256 / 256, 1])
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    newcolors[128, :] = white
    newcmp = ListedColormap(newcolors, name="OrangeBlue")
    return newcmp


OrBu = create_OrBu()


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


def plot_results(inp, pred, truth, model_path, save=False, plot_format="png"):
    """
    Plot input images, prediction and true image.
    Parameters
    ----------
    inp: n 2d arrays with 2 channel
        input images
    pred: n 2d arrays
        predicted images
    truth:n 2d arrays
        true images
    """
    for i in tqdm(range(len(inp))):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

        real = inp[i][0]
        im1 = ax1.imshow(real, cmap="RdBu", vmin=-real.max(), vmax=real.max())
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax1.set_title(r"Real Input")
        fig.colorbar(im1, cax=cax, orientation="vertical")

        imag = inp[i][1]
        im2 = ax2.imshow(imag, cmap="RdBu", vmin=-imag.max(), vmax=imag.max())
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax2.set_title(r"Imag Input")
        fig.colorbar(im2, cax=cax, orientation="vertical")

        pre = pred[i]
        im3 = ax3.imshow(pre, cmap="RdBu", vmin=-pre.max(), vmax=pre.max())
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax3.set_title(r"Prediction")
        fig.colorbar(im3, cax=cax, orientation="vertical")

        true = truth[i]
        im4 = ax4.imshow(true, cmap="RdBu", vmin=-pre.max(), vmax=pre.max())
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax4.set_title(r"Truth")
        fig.colorbar(im4, cax=cax, orientation="vertical")

        plt.tight_layout()

        if save:
            out = model_path / "predictions/"
            out.mkdir(parents=True, exist_ok=True)

            out_path = adjust_outpath(out, "/prediction", form=plot_format)
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def visualize_with_fourier(
    i, img_input, img_pred, img_truth, amp_phase, out_path, plot_format="png"
):
    """
    Visualizing, if the target variables are displayed in fourier space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (2*img_size^2)
    img_truth: current true image as a numpy array with shape (2*img_size^2)
    out_path: str which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    inp_real, inp_imag = img_input[0], img_input[1]
    real_pred, imag_pred = img_pred[0], img_pred[1]
    real_truth, imag_truth = img_truth[0], img_truth[1]

    if amp_phase:
        inp_real = 10 ** (10 * inp_real - 10) - 1e-10
        real_pred = 10 ** (10 * real_pred - 10) - 1e-10
        real_truth = 10 ** (10 * real_truth - 10) - 1e-10

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, figsize=(16, 10), sharex=True, sharey=True
    )

    if amp_phase:
        im1 = ax1.imshow(inp_real, cmap="inferno")
        make_axes_nice(fig, ax1, im1, r"Amplitude Input")

        im2 = ax2.imshow(real_pred, cmap="inferno")
        make_axes_nice(fig, ax2, im2, r"Amplitude Prediction")

        im3 = ax3.imshow(real_truth, cmap="inferno")
        make_axes_nice(fig, ax3, im3, r"Amplitude Truth")

        a = check_vmin_vmax(inp_imag)
        im4 = ax4.imshow(inp_imag, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax4, im4, r"Phase Input", phase=True)

        a = check_vmin_vmax(imag_truth)
        im5 = ax5.imshow(imag_pred, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax5, im5, r"Phase Prediction", phase=True)

        a = check_vmin_vmax(imag_truth)
        im6 = ax6.imshow(imag_truth, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax6, im6, r"Phase Truth", phase=True)
    else:
        a = check_vmin_vmax(inp_real)
        im1 = ax1.imshow(inp_real, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax1, im1, r"Real Input")

        a = check_vmin_vmax(real_truth)
        im2 = ax2.imshow(real_pred, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax2, im2, r"Real Prediction")

        a = check_vmin_vmax(real_truth)
        im3 = ax3.imshow(real_truth, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax3, im3, r"Real Truth")

        a = check_vmin_vmax(inp_imag)
        im4 = ax4.imshow(inp_imag, cmap="RdBu", vmin=-a, vmax=a)
        make_axes_nice(fig, ax4, im4, r"Imaginary Input")

        a = check_vmin_vmax(imag_truth)
        im5 = ax5.imshow(imag_pred, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax5, im5, r"Imaginary Prediction")

        a = check_vmin_vmax(imag_truth)
        im6 = ax6.imshow(imag_truth, cmap="RdBu", vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax6, im6, r"Imaginary Truth")

    ax1.set_ylabel(r"Pixels", fontsize=20)
    ax4.set_ylabel(r"Pixels", fontsize=20)
    ax4.set_xlabel(r"Pixels", fontsize=20)
    ax5.set_xlabel(r"Pixels", fontsize=20)
    ax6.set_xlabel(r"Pixels", fontsize=20)
    ax1.tick_params(axis="both", labelsize=20)
    ax2.tick_params(axis="both", labelsize=20)
    ax3.tick_params(axis="both", labelsize=20)
    ax4.tick_params(axis="both", labelsize=20)
    ax5.tick_params(axis="both", labelsize=20)
    ax6.tick_params(axis="both", labelsize=20)
    plt.tight_layout(pad=1.5)

    outpath = str(out_path) + f"/prediction_{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    return real_pred, imag_pred, real_truth, imag_truth


def visualize_with_fourier_diff(
    i, img_pred, img_truth, amp_phase, out_path, plot_format="png"
):
    """
    Visualizing, if the target variables are displayed in fourier space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (2*img_size^2)
    img_truth: current true image as a numpy array with shape (2*img_size^2)
    out_path: str which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    real_pred, imag_pred = img_pred[0], img_pred[1]
    real_truth, imag_truth = img_truth[0], img_truth[1]

    if amp_phase:
        real_pred = 10 ** (10 * real_pred - 10) - 1e-10
        real_truth = 10 ** (10 * real_truth - 10) - 1e-10

    # plotting
    # plt.style.use('./paper_large_3_2.rc')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, figsize=(16, 10), sharex=True, sharey=True
    )

    if amp_phase:
        im1 = ax1.imshow(real_pred, cmap="inferno", norm=LogNorm())
        make_axes_nice(fig, ax1, im1, r"Amplitude Prediction")

        im2 = ax2.imshow(real_truth, cmap="inferno", norm=LogNorm())
        make_axes_nice(fig, ax2, im2, r"Amplitude Truth")

        a = check_vmin_vmax(real_pred - real_truth)
        im3 = ax3.imshow(real_pred - real_truth, cmap=OrBu, vmin=-a, vmax=a)
        make_axes_nice(fig, ax3, im3, r"Amplitude Difference")

        a = check_vmin_vmax(imag_truth)
        im4 = ax4.imshow(imag_pred, cmap=OrBu, vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax4, im4, r"Phase Prediction", phase=True)

        a = check_vmin_vmax(imag_truth)
        im5 = ax5.imshow(imag_truth, cmap=OrBu, vmin=-np.pi, vmax=np.pi)
        make_axes_nice(fig, ax5, im5, r"Phase Truth", phase=True)

        a = check_vmin_vmax(imag_pred - imag_truth)
        im6 = ax6.imshow(
            imag_pred - imag_truth, cmap=OrBu, vmin=-2 * np.pi, vmax=2 * np.pi
        )
        make_axes_nice(fig, ax6, im6, r"Phase Difference", phase_diff=True)

    ax1.set_ylabel(r"Pixels")
    ax4.set_ylabel(r"Pixels")
    ax4.set_xlabel(r"Pixels")
    ax5.set_xlabel(r"Pixels")
    ax6.set_xlabel(r"Pixels")
    # ax1.tick_params(axis="both", labelsize=20)
    # ax2.tick_params(axis="both", labelsize=20)
    # ax3.tick_params(axis="both", labelsize=20)
    # ax4.tick_params(axis="both", labelsize=20)
    # ax5.tick_params(axis="both", labelsize=20)
    # ax6.tick_params(axis="both", labelsize=20)
    plt.tight_layout(pad=1)

    outpath = str(out_path) + f"/prediction_{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    return real_pred, imag_pred, real_truth, imag_truth


def visualize_source_reconstruction(
    ifft_pred,
    ifft_truth,
    out_path,
    i,
    dr=False,
    blobs=False,
    msssim=False,
    plot_format="png",
):
    m_truth, n_truth, alpha_truth = calc_jet_angle(ifft_truth)
    m_pred, n_pred, alpha_pred = calc_jet_angle(ifft_pred)
    x_space = torch.arange(0, 511, 1)

    # plt.style.use("./paper_large_3.rc")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.plot(
        x_space,
        m_pred * x_space + n_pred,
        "w-",
        alpha=0.5,
        label=fr"$\alpha = {np.round(alpha_pred[0], 3)}\,$deg",
    )
    im1 = ax1.imshow(ifft_pred, vmax=ifft_truth.max(), cmap="inferno")
    ax2.plot(
        x_space,
        m_truth * x_space + n_truth,
        "w-",
        alpha=0.5,
        label=fr"$\alpha = {np.round(alpha_truth[0], 3)}\,$deg",
    )
    im2 = ax2.imshow(ifft_truth, cmap="inferno")

    a = check_vmin_vmax(ifft_pred - ifft_truth)
    im3 = ax3.imshow(ifft_pred - ifft_truth, cmap=OrBu, vmin=-a, vmax=a)

    make_axes_nice(fig, ax1, im1, r"")
    make_axes_nice(fig, ax2, im2, r"")
    make_axes_nice(fig, ax3, im3, r"")

    ax1.set_ylabel(r"Pixels")
    ax1.set_xlabel(r"Pixels")
    ax2.set_xlabel(r"Pixels")
    ax3.set_xlabel(r"Pixels")

    # ax1.tick_params(axis="both", labelsize=20)
    # ax2.tick_params(axis="both", labelsize=20)
    # ax3.tick_params(axis="both", labelsize=20)

    if blobs:
        blobs_pred, blobs_truth = calc_blobs(ifft_pred, ifft_truth)
        plot_blobs(blobs_pred, ax1)
        plot_blobs(blobs_truth, ax2)

    if dr:
        dr_truth, dr_pred, num_boxes, corners = calc_dr(
            ifft_truth[None, ...], ifft_pred[None, ...]
        )
        ax1.plot([], [], " ", label=f"DR: {int(dr_pred[0])}")
        ax2.plot([], [], " ", label=f"DR: {int(dr_truth[0])}")

        plot_box(ax1, num_boxes, corners[0])
        plot_box(ax2, num_boxes, corners[0])

    if msssim:
        ifft_truth = pad_unsqueeze(torch.tensor(ifft_truth).unsqueeze(0))
        ifft_pred = pad_unsqueeze(torch.tensor(ifft_pred).unsqueeze(0))
        val = ms_ssim(ifft_pred, ifft_truth, data_range=ifft_truth.max())

        ax1.plot([], [], " ", label=f"ms ssim: {round_n_digits(val)}")

    outpath = str(out_path) + f"/fft_pred_{i}.{plot_format}"

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    fig.tight_layout(pad=1)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    return np.abs(ifft_pred), np.abs(ifft_truth)


def plot_contour(ifft_pred, ifft_truth, out_path, i, plot_format="png"):
    labels = [r"10%", r"30%", r"50%", r"80%"]
    colors = ("r", "tomato", "mistyrose", "black")
    levels = [
        ifft_truth.max() * 0.1,
        ifft_truth.max() * 0.3,
        ifft_truth.max() * 0.5,
        ifft_truth.max() * 0.8,
    ]

    # plt.style.use('./paper_large.rc')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    im1 = ax1.imshow(ifft_pred, vmax=ifft_truth.max())
    CS1 = ax1.contour(ifft_pred, levels=levels, colors=colors)
    make_axes_nice(fig, ax1, im1, "Prediction")

    im2 = ax2.imshow(ifft_truth)
    CS2 = ax2.contour(ifft_truth, levels=levels, colors=colors)
    diff = np.round(compute_area_difference(CS1, CS2), 2)
    make_axes_nice(fig, ax2, im2, "Truth, ratio: {}".format(diff))
    outpath = str(out_path) + f"/contour_{diff}_{i}.{plot_format}"

    # Assign labels for the levels and save them for the legend
    for i in range(len(labels)):
        CS1.collections[i].set_label(labels[i])
        CS2.collections[i].set_label(labels[i])

    # plotting legend
    ax1.legend(loc="best")
    ax2.legend(loc="best")

    ax1.set_ylabel(r"Pixels")
    ax1.set_xlabel(r"Pixels")
    ax2.set_xlabel(r"Pixels")

    plt.tight_layout(pad=0.75)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.05)


def histogram_jet_angles(alpha_truth, alpha_pred, out_path, plot_format="png"):
    dif = (alpha_pred - alpha_truth).numpy()

    mean = np.round(np.mean(dif), 3)
    std = np.round(np.std(dif, ddof=1), 3)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.hist(
        dif,
        51,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("Offset / deg")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    extra_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax1.legend([extra_1, extra_2], ("Mean: {}".format(mean), "Std: {}".format(std)))

    ax2.hist(
        dif[(dif > -10) & (dif < 10)],
        25,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax2.set_xticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    ax2.set_xlabel("Offset / deg")
    ax2.set_ylabel("Number of sources")

    fig.tight_layout()

    outpath = str(out_path) + f"/jet_offsets.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_dynamic_ranges(dr_truth, dr_pred, out_path, plot_format="png"):
    # dif = dr_pred - dr_truth

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    ax1.set_title("True Images")
    ax1.hist(
        dr_truth,
        51,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("Dynamic range")
    ax1.set_ylabel("Number of sources")

    ax2.set_title("Predictions")
    ax2.hist(
        dr_pred,
        25,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax2.set_xlabel("Dynamic range")
    ax2.set_ylabel("Number of sources")

    # plotting differences does not make much sense at the moment
    # ax3.set_title("Differences")
    # ax3.hist(
    #     dif,
    #     25,
    #     color="darkorange",
    #     linewidth=3,
    #     histtype="step",
    #     alpha=0.75,
    # )
    # ax3.set_xlabel("Dynamic range")
    # ax3.set_ylabel("Number of sources")

    fig.tight_layout()

    outpath = str(out_path) + f"/dynamic_ranges.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def plot_box(ax, num_boxes, corners):
    size = get_boxsize(num_boxes)
    img_size = 63
    if corners[2]:
        ax.axvspan(
            xmin=0,
            xmax=size,
            ymin=(img_size - size) / img_size,
            ymax=0.99,
            color="red",
            fill=False,
        )
    if corners[3]:
        ax.axvspan(
            xmin=img_size - size,
            xmax=img_size - 1,
            ymin=(img_size - size) / img_size,
            ymax=0.99,
            color="red",
            fill=False,
        )
    if corners[0]:
        ax.axvspan(
            xmin=0,
            xmax=size,
            ymin=0.01,
            ymax=(size) / img_size,
            color="red",
            fill=False,
        )
    if corners[1]:
        ax.axvspan(
            xmin=img_size - size,
            xmax=img_size - 1,
            ymin=0.01,
            ymax=(size) / img_size,
            color="red",
            fill=False,
        )


def plot_blobs(blobs_log, ax):
    """Plot the blobs created in sklearn.blob_log
    Parameters
    ----------
    blobs_log : ndarray
        return values of blob_log
    ax : axis object
        plotting axis
    """
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax.add_patch(c)


def histogram_ms_ssim(msssim, out_path, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        msssim.numpy(),
        51,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("ms ssim")
    ax1.set_ylabel("Number of sources")

    fig.tight_layout()

    outpath = str(out_path) + f"/ms_ssim.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_mean_diff(vals, out_path, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals.numpy(),
        51,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("Mean flux deviation / %")
    ax1.set_ylabel("Number of sources")

    fig.tight_layout()

    outpath = str(out_path) + f"/mean_diff.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_area(vals, out_path, plot_format="png"):
    vals = vals.numpy()
    mean = np.round(np.mean(vals), 3)
    std = np.round(np.std(vals, ddof=1), 3)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals,
        51,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("ratio of areas")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    extra_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax1.legend([extra_1, extra_2], ("Mean: {}".format(mean), "Std: {}".format(std)))

    fig.tight_layout()

    outpath = str(out_path) + f"/hist_area.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)
