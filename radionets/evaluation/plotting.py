from math import pi

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_msssim import ms_ssim
from radionets.evaluation.blob_detection import calc_blobs
from radionets.evaluation.contour import compute_area_ratio
from radionets.evaluation.dynamic_range import calc_dr, get_boxsize
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.utils import (
    check_vmin_vmax,
    make_axes_nice,
    pad_unsqueeze,
    reshape_2d,
)
from radionets.simulations.utils import adjust_outpath
from tqdm import tqdm

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
    def colorFader(
        c1, c2, mix=0
    ):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    c1 = "#3B0963"  # lila
    c2 = "#F88410"  # orange
    c3 = "#ebe6ef"
    c4 = "#fef3e7"
    n = 256

    fader_lila = [colorFader(c1, c3, x / n) for x in range(n + 1)]
    fader_orange = [colorFader(c4, c2, x / n) for x in range(n + 1)]
    cmap = fader_lila + ["white"] + fader_orange
    newcmp = ListedColormap(cmap, name="OrangeBlue")
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

    ax1.set_ylabel(r"Pixels")
    ax4.set_ylabel(r"Pixels")
    ax4.set_xlabel(r"Pixels")
    ax5.set_xlabel(r"Pixels")
    ax6.set_xlabel(r"Pixels")
    plt.tight_layout(pad=1.5)

    outpath = str(out_path) + f"/prediction_{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)


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
    plt.tight_layout(pad=1)

    outpath = str(out_path) + f"/prediction_{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    plt.close("all")


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
    x_space = torch.arange(0, 63, 1)

    # plt.style.use("./paper_large_3.rc")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10), sharey=True)

    # Plot prediction
    ax1.plot(x_space, m_pred * x_space + n_pred, "w--", alpha=0.5)
    ax1.axvline(32, 0, 1, linestyle="--", color="white", alpha=0.5)

    # create angle visualization
    theta1 = min(0, -alpha_pred.numpy()[0])
    theta2 = max(0, -alpha_pred.numpy()[0])
    ax1.add_patch(Arc([32, 32], 50, 50, 90, theta1, theta2, color="white"))

    im1 = ax1.imshow(ifft_pred, vmax=ifft_truth.max(), cmap="inferno")

    # Plot truth
    ax2.plot(x_space, m_truth * x_space + n_truth, "w--", alpha=0.5)
    ax2.axvline(32, 0, 1, linestyle="--", color="white", alpha=0.5)

    # create angle visualization
    theta1 = min(0, -alpha_truth.numpy()[0])
    theta2 = max(0, -alpha_truth.numpy()[0])
    ax2.add_patch(Arc([32, 32], 50, 50, 90, theta1, theta2, color="white"))

    im2 = ax2.imshow(ifft_truth, cmap="inferno")

    a = check_vmin_vmax(ifft_pred - ifft_truth)
    im3 = ax3.imshow(ifft_pred - ifft_truth, cmap=OrBu, vmin=-a, vmax=a)

    make_axes_nice(fig, ax1, im1, r"FFT Prediction")
    make_axes_nice(fig, ax2, im2, r"FFT Truth")
    make_axes_nice(fig, ax3, im3, r"FFT Diff")

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

        ax1.plot([], [], " ", label=f"ms ssim: {val:.2f}")

    outpath = str(out_path) + f"/fft_pred_{i}.{plot_format}"

    line = Line2D(
        [], [], linestyle="-", color="w", label=rf"$\alpha = {alpha_pred[0]:.2f}\,$deg"
    )
    line_truth = Line2D(
        [], [], linestyle="-", color="w", label=rf"$\alpha = {alpha_truth[0]:.2f}\,$deg"
    )

    ax1.legend(loc="best", handles=[line])
    ax2.legend(loc="best", handles=[line_truth])
    fig.tight_layout(pad=1)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    plt.close("all")
    return np.abs(ifft_pred), np.abs(ifft_truth)


def visualize_uncertainty(
    i, img_pred, img_truth, img_unc, amp_phase, out_path, plot_format="png"
):
    pred_amp, pred_phase = img_pred[0], img_pred[1]
    true_amp, true_phase = img_truth[0], img_truth[1]
    unc_amp, unc_phase = img_unc[0], img_unc[1]

    # amplitude
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharey=True, sharex=True, figsize=(12, 10)
    )

    im1 = ax1.imshow(true_amp)

    im2 = ax2.imshow(pred_amp)

    im3 = ax3.imshow(unc_amp)

    a = check_vmin_vmax(true_amp - pred_amp)
    im4 = ax4.imshow(true_amp - pred_amp, cmap=OrBu, vmin=-a, vmax=a)

    make_axes_nice(fig, ax1, im1, r"Simulation")
    make_axes_nice(fig, ax2, im2, r"Predicted $\mu$")
    make_axes_nice(fig, ax3, im3, r"Predicted $\sigma^2$", unc=True)
    make_axes_nice(fig, ax4, im4, r"Difference")

    ax1.set_ylabel(r"pixels")
    ax3.set_ylabel(r"pixels")
    ax3.set_xlabel(r"pixels")
    ax4.set_xlabel(r"pixels")

    fig.tight_layout(pad=1)
    outpath = str(out_path) + f"/unc_amp{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)

    # phase
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharey=True, sharex=True, figsize=(12, 10)
    )

    im1 = ax1.imshow(true_phase)

    im2 = ax2.imshow(pred_phase)

    im3 = ax3.imshow(unc_phase)

    a = check_vmin_vmax(true_phase - pred_phase)
    im4 = ax4.imshow(true_phase - pred_phase, cmap=OrBu, vmin=-a, vmax=a)

    make_axes_nice(fig, ax1, im1, r"Simulation")
    make_axes_nice(fig, ax2, im2, r"Predicted $\mu$")
    make_axes_nice(fig, ax3, im3, r"Predicted $\sigma^2$", unc=True)
    make_axes_nice(fig, ax4, im4, r"Difference")

    ax1.set_ylabel(r"pixels")
    ax3.set_ylabel(r"pixels")
    ax3.set_xlabel(r"pixels")
    ax4.set_xlabel(r"pixels")

    fig.tight_layout(pad=1)
    outpath = str(out_path) + f"/unc_phase{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    plt.close("all")


def visualize_sampled_unc(i, mean, std, ifft_truth, out_path, plot_format):
    # plt.style.use('../paper_large_3.rc')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 10), sharey=True, sharex=True
    )

    im1 = ax1.imshow(ifft_truth)
    im2 = ax2.imshow(mean)
    im3 = ax3.imshow(std)
    a = check_vmin_vmax(mean - ifft_truth)
    im4 = ax4.imshow(mean - ifft_truth, cmap=OrBu, vmin=-a, vmax=a)

    make_axes_nice(fig, ax1, im1, r"Simulation")
    make_axes_nice(fig, ax2, im2, r"Prediction")
    make_axes_nice(fig, ax3, im3, r"Uncertainty")
    make_axes_nice(fig, ax4, im4, r"Difference")

    ax1.set_ylabel(r"pixels")
    ax3.set_xlabel(r"pixels")
    ax3.set_ylabel(r"pixels")
    ax4.set_xlabel(r"pixels")
    fig.tight_layout(pad=1.5)
    outpath = str(out_path) + f"/unc_samp{i}.{plot_format}"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
    plt.close("all")


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
    diff = np.round(compute_area_ratio(CS1, CS2), 2)
    make_axes_nice(fig, ax2, im2, f"Truth, ratio: {diff}")
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
    plt.close("all")


def histogram_jet_angles(dif, out_path, plot_format="png"):
    mean = np.mean(dif)
    std = np.std(dif, ddof=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.hist(dif, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Offset / deg")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    extra_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

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
    ax1.hist(dr_truth, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Dynamic range")
    ax1.set_ylabel("Number of sources")

    ax2.set_title("Predictions")
    ax2.hist(dr_pred, 25, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
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
    img_size = 64
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
    vals = vals.numpy()
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(vals, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Mean flux deviation / %")
    ax1.set_ylabel("Number of sources")
    extra_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    extra_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

    fig.tight_layout()

    outpath = str(out_path) + f"/mean_diff.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_area(vals, out_path, plot_format="png"):
    vals = vals.numpy()
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    bins = np.arange(0, vals.max() + 0.1, 0.1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals, bins=bins, color="darkorange", linewidth=3, histtype="step", alpha=0.75
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("ratio of areas")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    extra_2 = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor="none", linewidth=0)
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

    fig.tight_layout()

    outpath = str(out_path) + f"/hist_area.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def hist_point(vals, mask, out_path, plot_format="png"):
    binwidth = 5
    min_all = vals.min()
    bins = np.arange(min_all, 100 + binwidth, binwidth)

    mean_point = np.mean(vals[mask])
    std_point = np.std(vals[mask], ddof=1)
    mean_extent = np.mean(vals[~mask])
    std_extent = np.std(vals[~mask], ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals[mask],
        bins=bins,
        color="darkorange",
        linewidth=2,
        histtype="step",
        alpha=0.75,
    )
    ax1.hist(
        vals[~mask],
        bins=bins,
        color="#1f77b4",
        linewidth=2,
        histtype="step",
        alpha=0.75,
    )
    ax1.axvline(0, linestyle="dotted", color="red")
    ax1.set_ylabel("Number of sources")
    ax1.set_xlabel("Mean specific intensity deviation")

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=2
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="#1f77b4", linewidth=2
    )
    ax1.legend(
        [extra_1, extra_2],
        [
            rf"Point: $({mean_point:.2f}\pm{std_point:.2f})\,\%$",
            rf"Extended: $({mean_extent:.2f}\pm{std_extent:.2f})\,\%$",
        ],
    )
    outpath = str(out_path) + f"/hist_point.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def plot_length_point(length, vals, mask, out_path, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.plot(
        length[mask],
        vals[mask],
        ".",
        markersize=1,
        color="darkorange",
        label="Point sources",
    )
    ax1.plot(
        length[~mask],
        vals[~mask],
        ".",
        markersize=1,
        color="#1f77b4",
        label="Extended sources",
    )
    ax1.set_ylabel("Mean specific intensity deviation")
    ax1.set_xlabel("Linear extent / pixels")
    plt.grid()
    plt.legend(loc="best", markerscale=10)

    outpath = str(out_path) + "/extend_point.png"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def plot_jet_results(inp, pred, truth, path, save=False, plot_format="pdf"):
    """
    Plot input images, prediction, true and diff image of the overall prediction.
    (Not component wise)
    Parameters
    ----------
    inp: n 4d arrays with 1 channel
        input images
    pred: n 4d arrays with multiple channels
        predicted images
    truth:n 4d arrays with multiple channels
        true images
    """
    if truth.shape[1] > 2:
        truth = torch.sum(truth[:, 0:-1], axis=1)
        pred = torch.sum(pred[:, 0:-1], axis=1)
    elif truth.shape[1] == 2:
        truth = truth[:, 0:-1].squeeze()
        pred = pred[:, 0:-1].squeeze()
    else:
        truth = truth.squeeze()
        pred = pred.squeeze()

    for i in tqdm(range(len(inp))):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(4, 7))

        im1 = ax[0].imshow(inp[i, 0], cmap=plt.cm.inferno)
        ax[0].set_xlabel(r"Pixels")
        ax[0].set_ylabel(r"Pixels")
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im1, cax=cax, orientation="vertical")
        cbar.set_label(r"Specific Intensity / a.u.")

        diff = pred[i] - truth[i]
        im2 = ax[1].imshow(diff, cmap=plt.cm.inferno)
        ax[1].set_xlabel(r"Pixels")
        ax[1].set_ylabel(r"Pixels")
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im2, cax=cax, orientation="vertical")
        cbar.set_label(r"Specific Intensity / a.u.")

        plt.tight_layout()

        if save:
            Path(path).mkdir(parents=True, exist_ok=True)
            outpath = str(path) + f"/prediction_{i}.{plot_format}"
            fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
        plt.close()


def plot_jet_components_results(inp, pred, truth, path, save=False, plot_format="pdf"):
    """
    Plot input images, prediction and true image.
    Parameters
    ----------
    inp: n 4d arrays with 1 channel
        input images
    pred: n 4d arrays with multiple channels
        predicted images
    truth: n 4d arrays with multiple channels
        true images
    """
    X, Y = np.meshgrid(np.arange(inp.shape[-1]), np.arange(inp.shape[-1]))
    for i in tqdm(range(len(inp))):
        c = truth.shape[1] - 1  # -1 because last one is the background
        for j in range(c):
            truth_max = torch.max(truth[i, j])
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 7))
            if not truth_max == 0:
                pred_max = torch.max(pred[i, j])
                axs[0, 0].contour(
                    X, Y, truth[i, j], levels=[truth_max * 0.32], colors="white"
                )
                axs[0, 1].contour(
                    X, Y, truth[i, j], levels=[truth_max * 0.32], colors="white"
                )
                axs[1, 0].contour(
                    X, Y, truth[i, j], levels=[truth_max * 0.32], colors="white"
                )
                axs[1, 0].contour(
                    X,
                    Y,
                    pred[i, j],
                    levels=[pred_max * 0.32],
                    colors="cyan",
                    linestyles="dashed",
                )

            im1 = axs[0, 0].imshow(inp[i, 0], cmap=plt.cm.inferno)
            axs[0, 0].set_xlabel(r"Pixels")
            axs[0, 0].set_ylabel(r"Pixels")
            divider = make_axes_locatable(axs[0, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation="vertical")
            cbar.set_label(r"Specific Intensity / a.u.")

            im2 = axs[0, 1].imshow(truth[i, j], cmap=plt.cm.inferno)
            axs[0, 1].set_xlabel(r"Pixels")
            axs[0, 1].set_ylabel(r"Pixels")
            divider = make_axes_locatable(axs[0, 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im2, cax=cax, orientation="vertical")
            cbar.set_label(r"Specific Intensity / a.u.")

            im1 = axs[1, 0].imshow(pred[i, j], cmap=plt.cm.inferno)
            axs[1, 0].set_xlabel(r"Pixels")
            axs[1, 0].set_ylabel(r"Pixels")
            divider = make_axes_locatable(axs[1, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation="vertical")
            cbar.set_label(r"Specific Intensity / a.u.")

            im4 = axs[1, 1].imshow(pred[i, j] - truth[i, j], cmap=plt.cm.inferno)
            divider = make_axes_locatable(axs[1, 1])
            axs[1, 1].set_xlabel(r"Pixels")
            axs[1, 1].set_ylabel(r"Pixels")
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im4, cax=cax, orientation="vertical")
            cbar.set_label(r"Specific Intensity / a.u.")

            plt.tight_layout(w_pad=2)

            if save:
                Path(path).mkdir(parents=True, exist_ok=True)
                outpath = str(path) + f"/prediction_{i}_comp_{j}.{plot_format}"
                fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
            plt.close()


def plot_fitgaussian(
    data, fit_list, params_list, iteration, path, save=False, plot_format="pdf"
):
    """
    Plotting the sky image with the fitted gaussian distributian and the related
    parameters.
    Parameters
    ----------
    data: 2d array
        skymap, usually the prediction of the NN
    fit: 2d array
        gaussian fit around the maxima
    params: list
        parameters related to the gaussian: height, x, y, width_x, width_y, theta
    """
    fig, axs = plt.subplots(
        1,
        len(params_list),
        sharex=True,
        sharey=True,
        figsize=(4 * len(params_list), 3.5),
    )
    for i, (fit, params) in enumerate(zip(fit_list, params_list)):
        im = axs[i].imshow(data, cmap=plt.cm.inferno)
        axs[i].set_xlabel(r"Pixels")
        axs[i].set_ylabel(r"Pixels")
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(r"Specific Intensity / a.u.")
        axs[i].contour(fit, cmap=plt.cm.gray_r)
        data -= fit
        (height, x, y, width_x, width_y, theta) = params.parameters
        plt.text(
            0.95,
            0.02,
            """
        height : %.2f
        x : %.1f
        y : %.1f
        width_x : %.1f
        width_y : %.1f
        theta : %.2f"""
            % (height, x, y, width_x, width_y, theta),
            fontsize=8,
            horizontalalignment="right",
            c="w",
            verticalalignment="bottom",
            transform=axs[i].transAxes,
        )

    plt.tight_layout()

    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        outpath = str(path) + f"/eval_iterativ_gaussian_{iteration}.{plot_format}"
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def hist_jet_gaussian_distance(dist, path, save=False, plot_format="pdf"):
    """
    Plotting the distances between predicted and true component of several images.
    Parameters
    ----------
    dist: 2d array
        array of shape (n, 2), where n is the number of distances
    """
    ran = [0, 50]

    plt.figure()
    plt.hist(
        dist[dist[:, 0] == 0][:, 1], bins=20, range=ran, alpha=0.7, label="Component 0"
    )
    plt.hist(
        dist[dist[:, 0] == 1][:, 1], bins=20, range=ran, alpha=0.7, label="Component 1"
    )
    plt.hist(
        dist[dist[:, 0] == 2][:, 1], bins=20, range=ran, alpha=0.7, label="Component 2"
    )
    plt.hist(
        dist[dist[:, 0] == 3][:, 1], bins=20, range=ran, alpha=0.7, label="Component 3"
    )
    plt.hist(
        dist[dist[:, 0] == 4][:, 1], bins=20, range=ran, alpha=0.7, label="Component 4"
    )
    plt.hist(
        dist[dist[:, 0] == 5][:, 1], bins=20, range=ran, alpha=0.7, label="Component 5"
    )
    plt.hist(
        dist[dist[:, 0] == 6][:, 1], bins=20, range=ran, alpha=0.7, label="Component 6"
    )
    plt.hist(
        dist[dist[:, 0] == 7][:, 1], bins=20, range=ran, alpha=0.7, label="Component 7"
    )
    plt.hist(
        dist[dist[:, 0] == 8][:, 1], bins=20, range=ran, alpha=0.7, label="Component 8"
    )
    plt.hist(
        dist[dist[:, 0] == 9][:, 1], bins=20, range=ran, alpha=0.7, label="Component 9"
    )
    plt.hist(
        dist[dist[:, 0] == 10][:, 1],
        bins=20,
        range=ran,
        alpha=0.7,
        label="Component 10",
    )
    plt.xlabel("Distance")
    plt.ylabel("Counts")
    plt.legend()

    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        outpath = str(path) + f"/hist_jet_gaussian_distance.{plot_format}"
        plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_data(x, path, rows=1, cols=1, save=False, plot_format="pdf"):
    """
    Plotting image of the dataset
    ----------
    x: array
        array of shape (n, 1, size, size), n must be at least rows * cols
    rows: int
        number of rows in the plot
    cols: int
        number of cols in the plot
    """
    fig, ax = plt.subplots(
        rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows)
    )
    for i in range(rows):
        for j in range(cols):
            img = ax[i, j].imshow(x[i * cols + j, 0], cmap=plt.cm.inferno)
            ax[i, j].set_xlabel(r"Pixels")
            ax[i, j].set_ylabel(r"Pixels")
            divider = make_axes_locatable(ax[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(img, cax=cax, orientation="vertical")
            cbar.set_label(r"Specific Intensity / a.u.")

    plt.tight_layout()

    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        outpath = str(path) + f"/simulation_examples.{plot_format}"
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def histogram_gan_sources(
    ratio, num_zero, above_zero, below_zero, num_images, out_path, plot_format="png"
):
    fig, ax1 = plt.subplots(1)
    bins = np.arange(0, ratio.max() + 0.1, 0.1)
    ax1.hist(
        ratio,
        bins=bins,
        histtype="step",
        label=f"mean: {ratio.mean():.2f}, max: {ratio.max():.2f}",
    )
    ax1.set_xlabel(r"Maximum difference to maximum true flux ratio")
    ax1.set_ylabel(r"Number of sources")
    ax1.legend(loc="best")

    fig.tight_layout()

    outpath = str(out_path) + f"/ratio.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)

    plt.clf()

    bins = np.arange(0, 102, 2)
    num_zero = num_zero.reshape(4, num_images)
    for i, label in enumerate(["1e-4", "1e-3", "1e-2", "1e-1"]):
        plt.hist(num_zero[i], bins=bins, histtype="step", label=label)
    plt.xlabel(r"Proportion of pixels close to 0 / %")
    plt.ylabel(r"Number of sources")
    plt.legend(loc="upper center")

    plt.tight_layout()

    outpath = str(out_path) + f"/num_zeros.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)

    plt.clf()

    bins = np.arange(0, 102, 2)
    plt.hist(
        above_zero,
        bins=bins,
        histtype="step",
        label=f"Above, mean: {above_zero.mean():.2f}%, max: {above_zero.max():.2f}%",
    )
    plt.hist(
        below_zero,
        bins=bins,
        histtype="step",
        label=f"Below, mean: {below_zero.mean():.2f}%, max: {below_zero.max():.2f}%",
    )
    plt.xlabel(r"Proportion of pixels below or above 0%")
    plt.ylabel(r"Number of sources")
    plt.legend(loc="upper center")
    plt.tight_layout()

    outpath = str(out_path) + f"/above_below.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)
