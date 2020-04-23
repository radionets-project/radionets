import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
from dl_framework.data import do_normalisation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import blob_log
from math import sqrt


def open_csv(path, mode):
    """
    opens a .csv file as a pandas dataframe which contains either the input,
    the predicted or the true images plus the used indices in the index row.
    The resulting dataframe is converted to a numpy array and returned along
    with the images.

    Parameters
    ----------
    path : path object from click
        path which leads to the csv file.
    mode : string
        Either input, predictions or truth.

    Returns
    ----------
    imgs : ndarry
        contains the images in a numpy array
    indices : 1-dim. array
        contains the used indices as an array
    -------
    """
    img_path = str(path) + "{}.csv".format(mode)
    img_df = pd.read_csv(img_path, index_col=0)
    indices = img_df.index.to_numpy()
    imgs = img_df.to_numpy()
    return imgs, indices


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
    if np.sqrt(img.shape[0]) % 1 == 0.0:
        img_size = int(np.sqrt(img.shape[0]))
        img_reshaped = img.reshape(img_size, img_size)

        return img_reshaped

    else:
        img_size = int(np.sqrt(img.shape[0] / 2))
        img_reshaped = img.reshape(1, 2, img_size, img_size)
        img_real = img_reshaped[0, 0, :]
        img_imag = img_reshaped[0, 1, :]

        return img_real, img_imag


def get_eval_img(valid_ds, model, norm_path):
    model.cuda()
    rand = np.random.randint(0, len(valid_ds))
    img = valid_ds[rand][0].cuda().unsqueeze(0).unsqueeze(0)
    # norm = pd.read_csv(norm_path)
    # img = do_normalisation(img, norm)
    h = int(img.shape[-1])
    #     img = img.view(-1, h, h).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()
    return img, pred, h, rand


def evaluate_model(valid_ds, model, norm_path, nrows=3):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=5,
        figsize=(18, 6 * nrows),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]},
    )

    for i in range(nrows):
        img, pred, h, rand = get_eval_img(valid_ds, model, norm_path)
        axes[i][0].set_title("x")
        axes[i][0].imshow(
            img[0].view(h, h).cpu(),
            # norm=LogNorm(),
        )
        axes[i][1].set_title("uncertainty")
        axes[i][1].imshow(
            pred[0, 1].view(h, h).cpu(), norm=LogNorm(),
        )
        axes[i][2].set_title("y_pred")
        im = axes[i][2].imshow(
            pred[0, 0].view(h, h),
            # norm=LogNorm(),
            vmax=valid_ds[rand][1].max(),
            vmin=1e-5,
        )
        axes[i][3].set_title("y_true")
        axes[i][3].imshow(
            valid_ds[rand][1].view(h, h),
            # norm=LogNorm(),
        )
        fig.colorbar(im, cax=axes[i][4])
    plt.tight_layout()


def plot_loss(learn, model_path, log=True):
    # to prevent the localhost error from happening
    # first change the backende and second turn off
    # the interactive mode
    matplotlib.use("Agg")
    plt.ioff()
    name_model = model_path.split("/")[-1].split(".")[0]
    save_path = model_path.split('.model')[0]
    print('\nPlotting Loss for: {}\n'.format(name_model))
    learn.recorder.plot_loss()
    plt.title(r"{}".format(name_model))
    plt.savefig('{}_loss.pdf'.format(save_path), bbox_inches='tight', pad_inches=0.01)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_lr_loss(learn, arch_name, skip_last):
    # to prevent the localhost error from happening
    # first change the backende and second turn off
    # the interactive mode
    matplotlib.use("Agg")
    plt.ioff()
    print("\nPlotting Lr vs Loss for architecture: {}\n".format(arch_name))
    learn.recorder_lr_find.plot(skip_last, save=True)
    # plt.yscale('log')
    plt.savefig("./models/lr_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def visualize_without_fourier(i, img_input, img_pred, img_truth, out_path):
    """
    Visualizing, if the target variables are displayed in spatial space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (img_size^2)
    img_truth: current true image as a numpy array with shape (img_size^2)
    out_path: string which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    inp_real, inp_imag = reshape_split(img_input)
    img_pred = reshape_split(img_pred)
    img_truth = reshape_split(img_truth)

    # plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    im1 = ax1.imshow(inp_real, cmap='RdBu', vmin=-inp_real.max(),
                     vmax=inp_real.max())
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax1.set_title(r'Real Input')
    fig.colorbar(im1, cax=cax, orientation='vertical')

    im2 = ax2.imshow(inp_imag, cmap='RdBu', vmin=-inp_imag.max(),
                     vmax=inp_imag.max())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax2.set_title(r'Imaginary Input')
    fig.colorbar(im2, cax=cax, orientation='vertical')

    im3 = ax3.imshow(img_pred)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax3.set_title(r'Prediction')
    fig.colorbar(im3, cax=cax, orientation='vertical')

    im4 = ax4.imshow(img_truth)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax4.set_title(r'Truth')
    fig.colorbar(im4, cax=cax, orientation='vertical')

    outpath = str(out_path) + "prediction_{}.png".format(i)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def visualize_with_fourier(i, img_input, img_pred, img_truth, amp_phase, out_path):
    """
    Visualizing, if the target variables are displayed in fourier space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (2*img_size^2)
    img_truth: current true image as a numpy array with shape (2*img_size^2)
    out_path: string which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    inp_real, inp_imag = reshape_split(img_input)
    real_pred, imag_pred = reshape_split(img_pred)
    real_truth, imag_truth = reshape_split(img_truth)

    if amp_phase:
        inp_real = 10**(10*inp_real-10) - 1e-10
        real_pred = 10**(10*real_pred-10) - 1e-10
        real_truth = 10**(10*real_truth-10) - 1e-10

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))

    im1 = ax1.imshow(inp_real, cmap='RdBu')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax1.set_title(r'Real Input')
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    im2 = ax2.imshow(real_pred, cmap='RdBu', vmin=real_truth.min(), vmax=real_truth.max())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax2.set_title(r'Real Prediction')
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    im3 = ax3.imshow(real_truth, cmap='RdBu')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax3.set_title(r'Real Truth')
    cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    im4 = ax4.imshow(inp_imag, cmap='RdBu')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax4.set_title(r'Imaginary Input')
    cbar = fig.colorbar(im4, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    im5 = ax5.imshow(imag_pred, cmap='RdBu', vmin=imag_truth.min(), vmax=imag_truth.max())
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax5.set_title(r'Imaginary Prediction')
    cbar = fig.colorbar(im5, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    im6 = ax6.imshow(imag_truth, cmap='RdBu')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax6.set_title(r'Imaginary Truth')
    cbar = fig.colorbar(im6, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    outpath = str(out_path) + "prediction_{}.png".format(i)
    fig.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
    return real_pred, imag_pred, real_truth, imag_truth


def visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, amp_phase, out_path):
    """
    function for visualizing the output of a inverse fourier transform. For now, it is
    necessary to take the absolute of the result of the inverse fourier transform,
    because the output is complex.
    i: current index of the loop, just used for saving
    real_pred: real part of the prediction computed in visualize with fourier
    imag_pred: imaginary part of the prediction computed in visualize with fourier
    real_truth: real part of the truth computed in visualize with fourier
    imag_truth: imaginary part of the truth computed in visualize with fourier
    """
    # create (complex) input for inverse fourier transformation for prediction
    if amp_phase:
        a = real_pred * np.cos(imag_pred)
        b = real_pred * np.sin(imag_pred)
        compl_pred = a + b * 1j

        a = real_truth * np.cos(imag_truth)
        b = real_truth * np.sin(imag_truth)
        compl_truth = a + b * 1j
    else:
        compl_pred = real_pred + imag_pred * 1j
        compl_truth = real_truth + imag_truth * 1j

    # inverse fourier transformation for prediction
    ifft_pred = np.fft.ifft2(compl_pred)

    # inverse fourier transform for truth
    ifft_truth = np.fft.ifft2(compl_truth)

    # plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    im1 = ax1.imshow(np.abs(ifft_pred))
    im2 = ax2.imshow(np.abs(ifft_truth))
    ax1.set_title(r'FFT Prediction')
    ax2.set_title(r'FFT Truth')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    outpath = str(out_path) + "fft_pred_{}.png".format(i)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
    return ifft_pred, ifft_truth


def plot_difference(i, img_pred, img_truth, fourier, out_path):
    plt.rcParams.update({"figure.max_open_warning": 0})
    if fourier:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 12))

        abs_pred, abs_truth = np.abs(img_pred), np.abs(img_truth)

        rms1 = np.sqrt((abs_pred[:10, :10]**2).mean())
        rms2 = np.sqrt((abs_pred[:10, -10:]**2).mean())
        rms3 = np.sqrt((abs_pred[-10:, :10]**2).mean())
        rms4 = np.sqrt((abs_pred[-10:, -10:]**2).mean())
        rms = np.sqrt((rms1**2 + rms2**2 + rms3**2 + rms4**2)/4)
        dynamic_range = abs_pred.max()/rms

        im1 = ax1.imshow(abs_pred)
        ax1.axvspan(0, 9, ymin=0.844, ymax=0.999, color='red', fill=False, label='Off')
        ax1.axvspan(0, 9, ymax=0.156, ymin=0.01, color='red', fill=False)
        ax1.axvspan(54, 63, ymin=0.844, ymax=0.999, color='red', fill=False)
        ax1.axvspan(54, 63, ymax=0.156, ymin=0.01, color='red', fill=False)

        im2 = ax2.imshow(abs_truth)
        ax2.axvspan(0, 9, ymin=0.844, ymax=0.999, color='red', fill=False, label='Off')
        ax2.axvspan(0, 9, ymax=0.156, ymin=0.01, color='red', fill=False)
        ax2.axvspan(54, 63, ymin=0.844, ymax=0.999, color='red', fill=False)
        ax2.axvspan(54, 63, ymax=0.156, ymin=0.01, color='red', fill=False)

        im3 = ax3.imshow(np.abs(img_pred - img_truth))

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax1.set_title(r'Prediction')
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax2.set_title(r'Truth')
        cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax3.set_title(r'DR: {}'.format(dynamic_range))
        cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        ax1.legend(loc="best")
        ax2.legend(loc="best")
        outpath = str(out_path) + "diff/difference_{}.png".format(i)
        plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)

        plt.clf()

    else:
        img_pred = reshape_split(img_pred)
        img_truth = reshape_split(img_truth)

        rms1 = np.sqrt((img_pred[:10, :10]**2).mean())
        rms2 = np.sqrt((img_pred[:10, -10:]**2).mean())
        rms3 = np.sqrt((img_pred[-10:, :10]**2).mean())
        rms4 = np.sqrt((img_pred[-10:, -10:]**2).mean())
        rms = np.sqrt((rms1**2 + rms2**2 + rms3**2 + rms4**2)/4)
        dynamic_range = img_pred.max()/rms

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 12))

        im1 = ax1.imshow(img_pred)
        ax1.axvspan(0, 9, ymin=0.844, ymax=0.999, color='red', fill=False, label='Off')
        ax1.axvspan(0, 9, ymax=0.156, ymin=0.01, color='red', fill=False)
        ax1.axvspan(54, 63, ymin=0.844, ymax=0.999, color='red', fill=False)
        ax1.axvspan(54, 63, ymax=0.156, ymin=0.01, color='red', fill=False)

        im2 = ax2.imshow(img_truth)
        ax2.axvspan(0, 9, ymin=0.844, ymax=0.999, color='red', fill=False, label='Off')
        ax2.axvspan(0, 9, ymax=0.156, ymin=0.01, color='red', fill=False)
        ax2.axvspan(54, 63, ymin=0.844, ymax=0.999, color='red', fill=False)
        ax2.axvspan(54, 63, ymax=0.156, ymin=0.01, color='red', fill=False)

        im3 = ax3.imshow(np.abs(img_pred - img_truth))

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax1.set_title(r'Prediction')
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax2.set_title(r'Truth')
        cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax3.set_title(r'DR: {}'.format(dynamic_range))
        cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()

        ax1.legend(loc="best")
        ax2.legend(loc="best")
        outpath = str(out_path) + "diff/difference_{}.png".format(i)
        plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)

        plt.clf()
    hist_difference(i, img_pred, img_truth, out_path)
    return dynamic_range


def hist_difference(i, img_pred, img_truth, out_path):
    x = np.abs(img_pred - img_truth).reshape(-1)
    plt.hist(x, label="Max Distance: {}".format(np.round(x.max(), 4)))
    plt.xlabel(r'Difference / a.u.')
    plt.ylabel(r"Number of pixels")
    plt.legend(loc="best")
    outpath = str(out_path) + "diff/hist_difference_{}.pdf".format(i)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)


def save_indices_and_data(indices, dr, outpath):
    df = pd.DataFrame(data=dr, index=indices)
    df.to_csv(outpath, index=True)


def plot_blobs(blobs_log, ax):
    for blob in blobs_log:
        y, x, r = blob
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        ax.add_patch(c)


def blob_detection(i, img_pred, img_truth, fourier, out_path):
    plt.rcParams.update({"figure.max_open_warning": 0})
    if fourier:
        img_pred = np.abs(img_pred)
        img_truth = np.abs(img_truth)
    else:
        img_pred = img_pred.reshape(64, 64)
        img_truth = img_truth.reshape(64, 64)
    tresh = img_truth.max()*0.1
    kwargs = {"min_sigma":1, "max_sigma":10, "num_sigma":100, "threshold": tresh, "overlap":0.9}
    blobs_log = blob_log(img_pred, **kwargs)
    blobs_log_truth = blob_log(img_truth, **kwargs)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blobs_log_truth[:, 2] = blobs_log_truth[:, 2] * sqrt(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    im1 = ax1.imshow(img_pred)
    im2 = ax2.imshow(img_truth)

    plot_blobs(blobs_log, ax1)
    plot_blobs(blobs_log_truth, ax2)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax1.set_title(r'Prediction')
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax2.set_title(r"Truth")
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    outpath = str(out_path) + "blob/blob_detection_{}.png".format(i)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
