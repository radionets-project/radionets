from math import pi
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_msssim import ms_ssim
from tqdm import tqdm

from radionets.dl_framework.utils import build_target_yolo
from radionets.evaluation.blob_detection import calc_blobs
from radionets.evaluation.contour import compute_area_ratio
from radionets.evaluation.dynamic_range import calc_dr, get_boxsize
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.utils import (
    check_vmin_vmax,
    get_strides,
    make_axes_nice,
    objectness_mapping,
    reshape_2d,
    yolo_apply_nms,
)
from radionets.simulations.utils import adjust_outpath

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


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


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
        im1 = ax1.imshow(inp_real, cmap="RdBu")
        make_axes_nice(fig, ax1, im1, r"Real Input")

        im2 = ax2.imshow(real_pred, cmap="RdBu")
        make_axes_nice(fig, ax2, im2, r"Real Prediction")

        im3 = ax3.imshow(real_truth, cmap="RdBu")
        make_axes_nice(fig, ax3, im3, r"Real Truth")

        a = check_vmin_vmax(inp_imag)
        im4 = ax4.imshow(inp_imag, cmap="RdBu")
        make_axes_nice(fig, ax4, im4, r"Imaginary Input")

        a = check_vmin_vmax(imag_truth)
        im5 = ax5.imshow(imag_pred, cmap="RdBu")
        make_axes_nice(fig, ax5, im5, r"Imaginary Prediction")

        a = check_vmin_vmax(imag_truth)
        im6 = ax6.imshow(imag_truth, cmap="RdBu")
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

    # plotting
    # plt.style.use('./paper_large_3_2.rc')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        2, 3, figsize=(16, 10), sharex=True, sharey=True
    )

    if amp_phase:
        im1 = ax1.imshow(real_pred, cmap="inferno")
        make_axes_nice(fig, ax1, im1, r"Amplitude Prediction")

        im2 = ax2.imshow(real_truth, cmap="inferno")
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

    else:
        im1 = ax1.imshow(real_pred, cmap="inferno")
        make_axes_nice(fig, ax1, im1, r"Real Prediction")

        im2 = ax2.imshow(real_truth, cmap="inferno")
        make_axes_nice(fig, ax2, im2, "Real Truth")

        a = check_vmin_vmax(real_pred - real_truth)
        im3 = ax3.imshow(real_pred - real_truth, cmap=OrBu, vmin=-a, vmax=a)
        make_axes_nice(fig, ax3, im3, r"Real Difference")

        im4 = ax4.imshow(imag_pred, cmap=OrBu)
        make_axes_nice(fig, ax4, im4, r"Imaginary Prediction")

        im5 = ax5.imshow(imag_truth, cmap=OrBu)
        make_axes_nice(fig, ax5, im5, r"Imaginary Truth")

        im6 = ax6.imshow(imag_pred - imag_truth, cmap=OrBu)
        make_axes_nice(fig, ax6, im6, r"Imaginary Difference")

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
    msssim=False,
    plot_format="png",
):
    # plt.style.use("./paper_large_3.rc")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 10), sharey=True)

    # Plot prediction
    im1 = ax1.imshow(ifft_pred, vmax=ifft_truth.max(), cmap="inferno")

    # Plot truth
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

    if dr:
        dr_truth, dr_pred, num_boxes, corners = calc_dr(
            ifft_truth[None, ...], ifft_pred[None, ...]
        )
        ax1.plot([], [], " ", label=f"DR: {int(dr_pred[0])}")
        ax2.plot([], [], " ", label=f"DR: {int(dr_truth[0])}")

        plot_box(ax1, num_boxes, corners[0])
        plot_box(ax2, num_boxes, corners[0])

    if msssim:
        val = ms_ssim(
            torch.tensor(ifft_pred).unsqueeze(0).unsqueeze(0),
            torch.tensor(ifft_truth).unsqueeze(0).unsqueeze(0),
            data_range=1,
            win_size=7,
            size_average=False,
        )
        val = val.numpy()[0]
        ax1.plot([], [], " ", label=f"MS-SSIM: {val:.2f}")
        ax1.legend(loc="best")

    outpath = str(out_path) + f"/fft_pred_{i}.{plot_format}"

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

    im1 = ax1.imshow(true_phase, cmap=OrBu)

    im2 = ax2.imshow(pred_phase, cmap=OrBu)

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

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
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


def plot_box2(ax, box, angle: float = 0, label: str = "True", c="g", alpha: float = 1):
    """Plots a box

    Parameters
    ----------
    ax: axis
        matplotlib axis object
    box: 1d-array
        center x, center y, width, height
    angle: float
        rotation angle of the box
    label: str
        label of the box
    c: color
        color for matplotlib
    alpha: float
        transparency of box
    """
    x, y, w, h = box
    x0 = x - w / 2
    y0 = y - h / 2

    # Create a Rectangle patch
    rect = Rectangle(
        (x0, y0),
        w,
        h,
        angle=angle,
        rotation_point="center",
        linewidth=1,
        edgecolor=c,
        facecolor="none",
        label=label,
        alpha=alpha,
    )
    # Add the patch to the Axes
    ax.add_patch(rect)


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
    mean = np.mean(msssim)
    std = np.std(msssim, ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        msssim,
        80,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("ms ssim")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle(
        (0, 0), 0.1, 0.1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    ax1.legend(
        [extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"), loc="upper center"
    )
    fig.tight_layout()

    outpath = str(out_path) + f"/ms_ssim.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_sum_intensity(ratios_sum, out_path, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    mean = np.mean(ratios_sum)
    std = np.std(ratios_sum, ddof=1)
    bins = np.arange(0.05, ratios_sum.max() + 0.05, 0.1)
    bins = np.insert(bins, 0, 0)
    ax1.hist(
        ratios_sum,
        bins=bins,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("Ratio of integrated intensity")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))
    ax1.set_xlim(-0.5, 5)

    fig.tight_layout()

    outpath = str(out_path) + f"/intensity_sum.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_peak_intensity(ratios_peak, out_path, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    mean = np.mean(ratios_peak)
    std = np.std(ratios_peak, ddof=1)
    bins = np.arange(0.05, ratios_peak.max() + 0.05, 0.1)
    bins = np.insert(bins, 0, 0)
    ax1.hist(
        ratios_peak,
        bins=bins,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("Ratio of intensity peak flux")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))
    ax1.set_xlim(-0.5, 5)

    fig.tight_layout()

    outpath = str(out_path) + f"/intensity_peak.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_mean_diff(vals, out_path, plot_format="png"):
    vals = vals.numpy()
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(vals, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Mean flux deviation / %")
    ax1.set_ylabel("Number of sources")
    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

    fig.tight_layout()

    outpath = str(out_path) + f"/mean_diff.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_area(vals, out_path, plot_format="png"):
    vals = vals.numpy()
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    bins = np.arange(0.05, np.round(vals.max()) + 0.05, 0.1)
    bins = np.insert(bins, 0, 0)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals, bins=bins, color="darkorange", linewidth=3, histtype="step", alpha=0.75
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("ratio of areas")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
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


def plot_yolo_obj_true(ax, y, pred: list, strides, idx: int = 0, anchor_idx: int = 0):
    """Plotting true objectness on axis

    Parameters
    ----------
    ax: matplotlib axis object
        axis to be plotted on
    y: 3d-array
        true data from simulation [bs, comp, attr]
    pred: list
        list of feature maps, each of shape (bs, 1, my, mx, 6)
    strides
        strides used in model
    idx: int
        index of image to be plotted
    anchor_idx: int
        indes of the anchor
    """
    obj_true = []
    for i in range(len(pred)):
        obj_true.append(build_target_yolo(y, shape=pred[i].shape, stride=strides[i]))

    out = objectness_mapping(obj_true, calc="sum", scaling=None)
    img = ax.imshow(out[idx, anchor_idx])

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    return img


def plot_yolo_obj_pred(ax, pred: list, idx: int = 0, anchor_idx: int = 0):
    """Plotting predicted objectness on axis by taking the maximum for each pixel.
    Feature maps with lower size are upsampled.

    Parameters
    ----------
    ax: matplotlib axis object
        axis to be plotted on
    pred: list
        list of feature maps, each of shape (bs, 1, my, mx, 6)
    idx: int
        index of image to be plotted
    anchor_idx: int
        indes of the anchor
    """
    out = objectness_mapping(pred, calc="sum")
    img = ax.imshow(out[idx, anchor_idx], vmin=0, vmax=1)

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    return img


def plot_yolo_box(
    ax,
    x,
    y,
    pred: list = [],
    idx: int = 0,
    true_boxes: bool = True,
    pred_boxes: bool = True,
    pred_label: bool = True,
):
    """Evaluation plot for YOLO boxes

    Parameters
    ----------
    ax: matplotlib axis
        using a predefined axis
    x: 4d-array
        input image (bs, 1, ny, nx)
    y: 3d-array
        true data from simulation (bs, components, paramters)
    pred: list
        list of feature maps, each of shape (bs, a, my, mx, 6)
    idx: int
        index of image to be plotted
    true_boxes: bool
        decide if true boxes are plotted
    pred_boxes: bool
        decide if predicted boxes are plotted
    pred_label: bool
        decide if label of predicted boxes are plotted
    """
    # Plot input image
    img = ax.imshow(x[idx, 0], cmap="inferno")

    # Plot true boxes
    if true_boxes:
        if torch.is_tensor(y):
            y = y.clone()
        else:
            y = y.copy()
        y[..., 3:5] *= 2
        y[..., 5] *= 180 / np.pi
        for i in range(y.shape[1]):
            if y[idx, i, 0] > 0.01:
                plot_box2(ax, y[idx, i, 1:5], angle=y[idx, i, 5], alpha=0.7)

    # Plot predicted boxes
    if pred_boxes and pred:
        outputs = yolo_apply_nms(pred, x)
        outputs = outputs[idx].detach().cpu().numpy()

        for i in range(outputs.shape[0]):
            plot_box2(
                ax,
                outputs[i, :4],
                angle=outputs[i, 5] * 180,
                label="Pred",
                c="w",
                alpha=0.7,
            )
            if pred_label:
                text = np.round(outputs[i, 4], 2)
                ax.text(outputs[i, 0], outputs[i, 1], text, color="lightblue")

    if true_boxes or (pred_boxes and pred_label and pred):
        legend_without_duplicate_labels(ax)

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    return img


def plot_yolo_eval(
    x,
    y,
    pred: list,
    idx: int = 0,
    anchor_idx: int = 0,
    out_path: str = "",
    plot_format: str = "pdf",
):
    """Default evaluation plot for YOLO

    Parameters
    ----------
    x: 4d-array
        input image (bs, 1, ny, nx)
    y: 3d-array
        true data from simulation (bs, components, paramters)
    pred: list
        list of feature maps, each of shape (bs, a, my, mx, 6)
    idx: int
        index of image to be plotted
    anchor_idx: int
        indes of the anchor
    out_path: str
        path in file directory to save output
    plot_format: str
        format of the plot
    """
    strides = get_strides(x, pred)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(3.6 * 2, 3 * 2))

    im1 = ax1.imshow(x[idx, 0], cmap="inferno")
    im2 = plot_yolo_obj_true(ax2, y, pred, strides, idx=idx, anchor_idx=anchor_idx)
    im3 = plot_yolo_box(
        ax3, x, y, pred, idx=idx, true_boxes=1, pred_boxes=1, pred_label=0
    )
    im4 = plot_yolo_obj_pred(ax4, pred, idx=idx, anchor_idx=anchor_idx)

    make_axes_nice(fig, ax1, im1)
    make_axes_nice(fig, ax2, im2, objectness=True)
    make_axes_nice(fig, ax3, im3)
    make_axes_nice(fig, ax4, im4, objectness=True)

    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Pixel")

    plt.tight_layout()
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/yolo_eval_{idx}.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_counterjet_eval(
    x,
    pred,
    y=None,
    data_name: str = "",
    out_path: str = "",
    plot_format: str = "pdf",
):
    """Evaluation plot for counterjet method

    Parameters
    ----------
    x: 4d-array
        input image (bs, 1, ny, nx)
    pred:
        1d-array
    y: 3d-array
        true data from simulation (bs, components, paramters)
    data_name: str
        name to save plots for different data types
    out_path: str
        path in file directory to save output
    plot_format: str
        format of the plot
    """
    idx_min = np.argmin(pred)
    idx_max = np.argmax(pred)
    idx_mid = (np.abs(pred - 0.5)).argmin()

    fig, axs = plt.subplots(3, 1, figsize=(3.6, 3 * 3), sharex=True)
    for i, (ax, idx) in enumerate(zip(axs, [idx_min, idx_mid, idx_max])):
        if y is None:
            s = f"Pred: {np.round(pred[idx], 2):.2}"
        else:
            s = f"True: {int(y[idx])}, Pred: {np.round(pred[idx], 2):.2}"
        im = ax.imshow(x[idx, 0], cmap="inferno")
        ax.text(
            0.10,
            0.90,
            s,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.7,
            ),
        )
        if i == 2:
            ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        make_axes_nice(fig, ax, im)

    plt.tight_layout()
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/counterjet_eval_{data_name}.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_yolo_mojave(
    x,
    pred: list,
    idx: int = 0,
    objectness: bool = True,
    out_path: str = "",
    name: str = "",
    date: str = "",
    plot_format: str = "pdf",
):
    """Default evaluation plot for MOJAVE data in YOLO

    Parameters
    ----------
    x: 4d-array
        input image (bs, 1, ny, nx)
    pred: list
        list of feature maps, each of shape (bs, a, my, mx, 6)
    idx: int
        index of image to be plotted
    objectness: bool
        Plot objectness of prediction
    out_path: str
        path in file directory to save output
    name: str
        name of source
    date: str
        date of measurement
    plot_format: str
        format of the plot
    """
    if objectness:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.6 * 2, 3))
        im1 = plot_yolo_box(
            ax1, x, None, pred, idx=idx, true_boxes=False, pred_label=False
        )
        im2 = plot_yolo_obj_pred(ax2, pred, idx=idx)
        make_axes_nice(fig, ax2, im2, objectness=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(3.6, 3))
        im1 = plot_yolo_box(
            ax1, x, None, pred, idx=idx, true_boxes=False, pred_label=False
        )

    make_axes_nice(fig, ax1, im1)

    legend_without_duplicate_labels(ax1)

    fig.tight_layout(pad=0.05)
    if out_path and name and date:
        out_path = str(out_path) + f"/{name}"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/mojave_eval_{date}.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_yolo_post_clustering(
    x,
    df,
    idx: int = 0,
    out_path: str = "",
    name: str = "",
    date: str = "",
    plot_format: str = "pdf",
):
    """Default evaluation plot for MOJAVE data in YOLO

    Parameters
    ----------
    x: 3d-array
        input image (bs, ny, nx)
    df: pandas.DataFrame
        dataframe with reconstructed properties
    idx: int
        index of image to be plotted
    out_path: str
        path in file directory to save output
    name: str
        name of source
    date: str
        date of measurement
    plot_format: str
        format of the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3))
    im = ax.imshow(x[idx], cmap="inferno")

    boxes = df[df["idx_img"] == idx][["x", "y", "sx", "sy"]].to_numpy()
    rot = df[df["idx_img"] == idx]["rotation"].to_numpy()
    for i in range(boxes.shape[0]):
        plot_box2(ax, boxes[i], angle=rot[i], label="Pred box", c="w", alpha=0.7)

    legend_without_duplicate_labels(ax)
    make_axes_nice(fig, ax, im)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    fig.tight_layout(pad=0.05)
    if out_path and name and date:
        out_path = str(out_path) + f"/{name}"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/mojave_post_eval_{date}.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_yolo_clustering(
    df, out_path: str = "", name: str = "", plot_format: str = "pdf"
):
    """Plot all predicted positions used for clustering.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe with reconstructed properties
    out_path: str
        path in file directory to save output
    name: str
        name of source
    plot_format: str
        format of the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=((3.6, 3)))
    for i in sorted(df["idx_comp"].unique()):
        ax.scatter(
            df[df["idx_comp"] == i]["x_mas"],
            df[df["idx_comp"] == i]["y_mas"],
            s=10,
            label=f"{i}",
        )
    legend_without_duplicate_labels(ax)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("Relative RA / mas")
    plt.ylabel("Relative DEC / mas")
    plt.grid()

    fig.tight_layout(pad=0.05)
    if out_path and name:
        out_path = str(out_path) + "/" + name
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/cluster_positions.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_yolo_velocity(
    df, quiet: bool = True, out_path: str = "", name: str = "", plot_format: str = "pdf"
):
    """Plot all predicted distances and the linear fit for the velocity

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe with reconstructed properties
    quiet: bool
        suppress print, enables legend
    out_path: str
        path in file directory to save output
    name: str
        name of source
    plot_format: str
        format of the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=((3.6, 3)))
    ax.grid()
    for i in sorted(df["idx_comp"].unique()):
        x = df[df["idx_comp"] == i]["date"]
        y = df[df["idx_comp"] == i]["distance"]
        m = df[df["idx_comp"] == i]["fit_param_m"]
        b = df[df["idx_comp"] == i]["fit_param_b"]

        v = np.round(df[df["idx_comp"] == i]["v"].values[0], 2)
        v_unc = np.round(df[df["idx_comp"] == i]["v_unc"].values[0], 2)

        ax.plot(x, m * x.astype(int) / 1e9 + b, "-", c=f"C{i}", linewidth=0.7)
        if not quiet:
            print(f"v{i} = {v} ± {v_unc} c, d{i} = {np.round(y.mean(), 2)} mas")
            ax.scatter(x, y, s=10, c=f"C{i}")
        else:
            # ax.plot(x, y, "o", c=f"C{i}", label=f"$v_{i} = {v} \pm {v_unc}$c")
            ax.scatter(
                x,
                y,
                s=10,
                c=f"C{i}",
                label=f"$v_{i} = {v} ± {v_unc} c_0$",
                # label=rf"$v_{i} = \SI{{{v}({v_unc})}}{{\clight}}$",
            )

    ax.set_xlabel("")
    ax.set_ylabel("Distance / mas")
    if quiet:
        ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels(which="major"):
        label.set(rotation=30, horizontalalignment="right")

    fig.tight_layout()
    if out_path and name:
        out_path = str(out_path) + "/" + name
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/velocity.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight")


def plot_hist_counterjet(
    x1,
    x2=None,
    threshold: float = None,
    data_name: str = "",
    out_path: str = "",
    plot_format: str = "pdf",
):
    """Plot destribution of counterjet in a histogram

    Parameters
    ----------
    x1: array
        prediction
    x2: array
        true value
    threshold: float
        threshold for detecting a counterjet
    data_name: str
        name to save plots for different data types
    out_path: str
        path in file directory to save output
    plot_format: str
        format of the plot
    """
    if data_name == "MOJAVE":
        bins = 20
    else:
        bins = 50

    fig, ax = plt.subplots(1, 1, figsize=((3, 2.5)))

    ax.hist(x1, bins=bins, range=(0, 1), alpha=0.7, histtype="step", label="Prediction")
    if x2 is not None:
        ax.hist(
            x2,
            bins=bins,
            range=(0, 1),
            alpha=0.7,
            histtype="step",
            linestyle="dashed",
            label="True",
        )
    if threshold is not None:
        ax.axvline(threshold, c="red", linewidth=0.5)
        ax.axvline(1 - threshold, c="red", linewidth=0.5)

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels(
        [0.0, "\nNo counterjet", 0.2, "", 0.4, "", 0.6, "", 0.8, "\nCounterjet", 1.0]
    )
    ax.set_xlabel("Model score")
    ax.set_ylabel("Counts")

    ax.set_yscale("log")
    ax.legend(loc="upper center")

    fig.tight_layout(pad=0.05)
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/hist_counterjet_{data_name}.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_hist_velocity(v1, v2, out_path: str = "", plot_format: str = "pdf"):
    """Plot destribution of velocities in a histogram

    Parameters
    ----------
    v: array
        velocity difference as a 1d array
    out_path: str
        path in file directory to save output
    plot_format: str
        format of the plot
    """
    r_min = np.min(np.concatenate((v1, v2)))
    r_max = np.max(np.concatenate((v1, v2)))
    bins = 10 ** (np.linspace(np.log10(r_min), np.log10(r_max), 30))
    # bins = 30

    fig, ax = plt.subplots(1, 1, figsize=((3.6, 3)))
    ax.hist(
        v1,
        bins=bins,
        range=(r_min, r_max),
        histtype="step",
        label="Prediction",
    )
    ax.hist(
        v2,
        bins=bins,
        range=(r_min, r_max),
        histtype="step",
        label="Lister et al.",
    )

    ax.set_xlabel("Velocity / c")
    ax.set_ylabel("Counts")
    ax.set_xscale("log")
    # ax.grid()
    ax.legend()

    fig.tight_layout(pad=0.05)
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/hist_velocity.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_hist_velocity_unc(
    v1_unc, v2_unc=None, out_path: str = "", plot_format: str = "pdf"
):
    """Plot destribution of velocities in a histogram

    Parameters
    ----------
    v1_unc: array
        relative velocity uncertainty as a 1d array
    v2_unc: array
        relative velocity uncertainty as a 1d array
    out_path: str
        path in file directory to save output
    plot_format: str
        format of the plot
    """
    if v2_unc is not None:
        r_min = np.min(np.concatenate((v1_unc, v2_unc)))
        r_max = np.max(np.concatenate((v1_unc, v2_unc)))
    else:
        r_min = np.min(v1_unc)
        r_max = np.max(v1_unc)

    bins = 10 ** (np.linspace(np.log10(r_min), np.log10(r_max), 30))

    fig, ax = plt.subplots(1, 1, figsize=((4.5, 4)))
    ax.hist(v1_unc, bins=bins, range=(r_min, r_max), alpha=0.7, label="Predicted")
    if v2_unc is not None:
        ax.hist(
            v2_unc, bins=bins, range=(r_min, r_max), alpha=0.7, label="Lister et al."
        )

    ax.set_xlabel("Relative velocity uncertainty")
    ax.set_ylabel("Counts")
    ax.set_xscale("log")
    # ax.grid()
    ax.legend()

    fig.tight_layout(pad=0.05)
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/hist_velocity_unc.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_beam_props(
    bmaj: list,
    bmin: list,
    bpa: list,
    freq: list,
    out_path: str = "",
    plot_format: str = "pdf",
):
    """Plot histogram of beam properties in MOJAVE dataset to find value for
    evaluating simulated sources with PyBDSF.

    Parameters
    ----------
    bmaj: list or array
        clean beam major axis diameter (degrees)
    bmin: list or array
        clean beam minor axis diameter (degrees)
    bpa: list or array
        clean beam position angle (degrees)
    freq: list or array
        frequency value (Hz)
    out_path: str
        path in file directory to save output
    plot_format: str
        format of the plot
    """
    fig, ax = plt.subplots(1, 4, figsize=((18, 4)))
    ax[0].hist(bmaj, bins=20, label=f"Mean: {np.array(bmaj).mean():.3}")
    ax[1].hist(bmin, bins=20, label=f"Mean: {np.array(bmin).mean():.3}")
    ax[2].hist(bpa, bins=20, label=f"Mean: {np.array(bpa).mean():.3}")
    ax[3].hist(freq, bins=20, label=f"Mean: {np.array(freq).mean():.3}")

    ax[0].set_xlabel("Clean beam major axis diameter / °")
    ax[0].set_ylabel("Counts")
    ax[1].set_xlabel("Clean beam minor axis diameter / °")
    ax[1].set_ylabel("Counts")
    ax[2].set_xlabel("Clean beam position angle / °")
    ax[2].set_ylabel("Counts")
    ax[3].set_xlabel("Frequency / Hz")
    ax[3].set_ylabel("Counts")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    fig.tight_layout()
    if out_path:
        Path(out_path).mkdir(parents=True, exist_ok=True)
        out_path = str(out_path) + f"/beam_props.{plot_format}"
        plt.savefig(out_path, bbox_inches="tight")


def plot_data(x, path, rows=1, cols=1, save=False, plot_format="pdf"):
    """
    Plotting image of the dataset

    Parameters
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


def plot_loss(
    model_path: str,
    out_path: str,
    model_name: str = "",
    metric_name: str = "",
    log_loss: bool = True,
    save: bool = False,
    plot_format: str = "pdf",
):
    """Plotting loss curve of trained model and metric, if available

    Parameters
    ----------
    model_path: str
        path in file directory to model
    out_path: str
        path in file directory to save output
    metric_name: str
        name of used metric
    save: bool
        image is saved if set to true
    plot_format: str
        format of the plot
    """
    checkpoint = torch.load(model_path, map_location="cuda:0")

    fig, ax1 = plt.subplots(figsize=(6.4 * 0.7, 4.8 * 0.7))
    ax2 = ax1.twinx()
    lns1 = ax1.plot(checkpoint["train_loss"], label="Training loss")

    valid_loss = np.array(checkpoint["valid_loss"])
    if valid_loss.shape[1] == 2:
        lns2 = ax1.plot(
            np.array(checkpoint["valid_loss"])[:, 0], label="Validation loss"
        )
        lns3 = ax2.plot(
            np.array(checkpoint["valid_loss"])[:, 1], "g", label=metric_name
        )
        lns = lns1 + lns2 + lns3
        ax1.legend(handles=lns, loc="center right")
        if model_name:
            out_path = str(out_path) + f"/loss_metric_{model_name}.{plot_format}"
        else:
            out_path = str(out_path) + f"/loss_metric.{plot_format}"
    else:
        lns2 = ax1.plot(
            np.array(checkpoint["valid_loss"])[:, 0], label="Validation loss"
        )
        lns = lns1 + lns2
        ax1.legend(handles=lns, loc="upper right")
        if model_name:
            out_path = str(out_path) + f"/loss_{model_name}.{plot_format}"
        else:
            out_path = str(out_path) + f"/loss.{plot_format}"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    if log_loss:
        ax1.set_yscale("log")
    ax2.set_ylabel(metric_name)

    plt.tight_layout()
    if save:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


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
