import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import pandas as pd
from skimage.feature import blob_log
from math import sqrt


# make nice Latex friendly plots
mpl.use("pgf")
mpl.rcParams.update(
    {
        "font.size": 12,
        "font.family": "sans-serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "lualatex",
    }
)


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
    mode : str
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
    mpl.use("Agg")
    plt.ioff()
    name_model = model_path.split("/")[-1].split(".")[0]
    save_path = model_path.split(".model")[0]
    print("\nPlotting Loss for: {}\n".format(name_model))
    learn.recorder.plot_loss()
    plt.title(r"{}".format(name_model))
    plt.savefig("{}_loss.pdf".format(save_path), bbox_inches="tight", pad_inches=0.01)
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr_loss(learn, arch_name, skip_last):
    # to prevent the localhost error from happening
    # first change the backende and second turn off
    # the interactive mode
    mpl.use("Agg")
    plt.ioff()
    print("\nPlotting Lr vs Loss for architecture: {}\n".format(arch_name))
    learn.recorder_lr_find.plot(skip_last, save=True)
    # plt.yscale('log')
    plt.savefig("./models/lr_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    mpl.rcParams.update(mpl.rcParamsDefault)


def visualize_without_fourier(i, img_input, img_pred, img_truth, out_path):
    """
    Visualizing, if the target variables are displayed in spatial space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (img_size^2)
    img_truth: current true image as a numpy array with shape (img_size^2)
    out_path: str which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    inp_real, inp_imag = reshape_split(img_input)
    img_pred = reshape_split(img_pred)
    img_truth = reshape_split(img_truth)

    # plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    im1 = ax1.imshow(inp_real, cmap="RdBu", vmin=-inp_real.max(), vmax=inp_real.max())
    make_axes_nice(fig, ax1, im1, r"Real Input")

    im2 = ax2.imshow(inp_imag, cmap="RdBu", vmin=-inp_imag.max(), vmax=inp_imag.max())
    make_axes_nice(fig, ax2, im2, r"Imaginary Input")

    im3 = ax3.imshow(img_pred)
    make_axes_nice(fig, ax3, im3, r"Prediction")

    im4 = ax4.imshow(img_truth)
    make_axes_nice(fig, ax4, im4, r"Truth")

    outpath = str(out_path) + "prediction_{}.png".format(i)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def visualize_with_fourier(i, img_input, img_pred, img_truth, amp_phase, out_path):
    """
    Visualizing, if the target variables are displayed in fourier space.
    i: Current index given form the loop
    img_input: current input image as a numpy array in shape (2*img_size^2)
    img_pred: current prediction image as a numpy array with shape (2*img_size^2)
    img_truth: current true image as a numpy array with shape (2*img_size^2)
    out_path: str which contains the output path
    """
    # reshaping and splitting in real and imaginary part if necessary
    inp_real, inp_imag = reshape_split(img_input)
    real_pred, imag_pred = reshape_split(img_pred)
    real_truth, imag_truth = reshape_split(img_truth)

    if amp_phase:
        inp_real = 10 ** (10 * inp_real - 10) - 1e-10
        real_pred = 10 ** (10 * real_pred - 10) - 1e-10
        real_truth = 10 ** (10 * real_truth - 10) - 1e-10

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))

    im1 = ax1.imshow(inp_real, cmap="RdBu")
    make_axes_nice(fig, ax1, im1, r"Amplitude Input")

    im2 = ax2.imshow(
        real_pred, cmap="RdBu", vmin=real_truth.min(), vmax=real_truth.max()
    )
    make_axes_nice(fig, ax2, im2, r"Amplitude Prediction")

    im3 = ax3.imshow(real_truth, cmap="RdBu")
    make_axes_nice(fig, ax3, im3, r"Amplitude Truth")

    im4 = ax4.imshow(inp_imag, cmap="RdBu")
    make_axes_nice(fig, ax4, im4, r"Phase Input")

    im5 = ax5.imshow(
        imag_pred, cmap="RdBu", vmin=imag_truth.min(), vmax=imag_truth.max()
    )
    make_axes_nice(fig, ax5, im5, r"Phase Prediction")

    im6 = ax6.imshow(imag_truth, cmap="RdBu")
    make_axes_nice(fig, ax6, im6, r"Phase Truth")

    outpath = str(out_path) + "prediction_{}.png".format(i)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
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

    make_axes_nice(fig, ax1, im1, r"FFT Prediction")
    make_axes_nice(fig, ax2, im2, r"FFT Truth")

    outpath = str(out_path) + "fft_pred_{}.png".format(i)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)
    return np.abs(ifft_pred), np.abs(ifft_truth)


def plot_difference(i, img_pred, img_truth, sensitivity, out_path):
    """Create a subplot with the prediction, the truth and the difference.
    Also computes the dynamic range for the prediction and the truth. Last,
    calculates the quotient between these two values. If one RMS value in one corner of
    the true image exceeds a given sensitivity, this region is excluded from the
    computation and the other regions get larger by a third. If two opposite corners
    exceed the sensitivity, the other two regions double in size.


    Parameters
    ----------
    i : int
        number of picture
    img_pred : ndarray
        image of prediction
    img_truth : ndarray
        image of truth
    sensitivity : float
        upper limit for RMS value
    out_path : str
        Output path as defined in makerc

    Returns
    -------
    float
        Quotient between the dynamic range sof truth and prediction
    """
    plt.rcParams.update({"figure.max_open_warning": 0})

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 12))

    img_size = img_pred.shape[0]
    if img_size == 63:
        # num_tree for 3 off regions, num_two for 2 off regions and so on
        num_three = 13
        num_two = 20
        num_four = 10
    elif img_size == 127:
        num_three = 60
        num_two = 50
        num_four = 40
    elif img_size == 511:
        # work in progress, these are dummy values for compilating
        num_three = 60
        num_two = 50
        num_four = 40
    num = [num_three, num_two, num_four]
    dr_truth, mode = compute_dr(i, img_truth, sensitivity, num)
    dr_pred = compute_dr_pred(img_pred, mode, num)
    dynamic_range = dr_pred / dr_truth

    im1 = ax1.imshow(img_pred)
    plot_off_regions(ax1, mode, img_size, num)

    im2 = ax2.imshow(img_truth)
    plot_off_regions(ax2, mode, img_size, num)

    im3 = ax3.imshow(np.abs(img_pred - img_truth))

    make_axes_nice(fig, ax1, im1, r"Prediction: {}".format(np.round(dr_pred, 4)))
    make_axes_nice(fig, ax2, im2, r"Truth: {}".format(np.round(dr_truth, 4)))
    make_axes_nice(fig, ax3, im3, r"DR: {}".format(dynamic_range))

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    outpath = str(out_path) + "diff/difference_{}.png".format(i)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)

    plt.clf()

    hist_difference(i, img_pred, img_truth, out_path)
    return dynamic_range


def plot_off_regions(ax, mode, img_size, num):
    """Plot the off regions for the computation of the dynamic range.
    Adjust the plotted off regions according to the mode.

    Parameters
    ----------
    ax : axis object
        current axis
    mode : str
        current mode
    img_size : int
        current image size
    num : list
        side length of quadratic off region
    """
    num_three, num_two, num_four = num[0], num[1], num[2]
    if mode == "rms1":
        ax.axvspan(
            0,
            num_three - 1,
            ymax=num_three / img_size,
            ymin=0.01,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            img_size - num_three,
            img_size - 1,
            ymin=1 - num_three / img_size,
            ymax=0.99,
            color="red",
            fill=False,
        )
        ax.axvspan(
            img_size - num_three,
            img_size - 1,
            ymax=num_three / img_size,
            ymin=0.01,
            color="red",
            fill=False,
        )
    elif mode == "rms2":
        ax.axvspan(
            0,
            num_three - 1,
            ymax=num_three / img_size,
            ymin=0.01,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            0,
            num_three - 1,
            ymax=1 - num_three / img_size,
            ymin=0.99,
            color="red",
            fill=False,
        )
        ax.axvspan(
            img_size - num_three,
            img_size - 1,
            ymax=num_three / img_size,
            ymin=0.01,
            color="red",
            fill=False,
        )
    elif mode == "rms3":
        ax.axvspan(
            0,
            num_three - 1,
            ymax=1 - num_three / img_size,
            ymin=0.99,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            img_size - num_three,
            img_size - 1,
            ymax=1 - num_three / img_size,
            ymin=0.99,
            color="red",
            fill=False,
        )
        ax.axvspan(
            img_size - num_three,
            img_size - 1,
            ymax=num_three / img_size,
            ymin=0.01,
            color="red",
            fill=False,
        )
    elif mode == "rms4":
        ax.axvspan(
            0,
            num_three - 1,
            ymax=num_three / img_size,
            ymin=0.01,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            0,
            num_three - 1,
            ymax=1 - num_three / img_size,
            ymin=0.99,
            color="red",
            fill=False,
        )
        ax.axvspan(
            img_size - num_three,
            img_size - 1,
            ymax=1 - num_three / img_size,
            ymin=0.99,
            color="red",
            fill=False,
        )
    elif mode == "rms1+4":
        ax.axvspan(
            0,
            num_two - 1,
            ymax=num_two / img_size,
            ymin=0.01,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            img_size - num_two,
            img_size - 1,
            ymax=1 - num_two / img_size,
            ymin=0.99,
            color="red",
            fill=False,
        )
    elif mode == "rms2+3":
        ax.axvspan(
            0,
            num_two - 1,
            ymax=1 - num_two / img_size,
            ymin=0.99,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            img_size - num_two,
            img_size - 1,
            ymax=num_two / img_size,
            ymin=0.01,
            color="red",
            fill=False,
        )
    elif mode is None:
        ax.axvspan(
            0,
            num_four - 1,
            ymin=1 - num_four / img_size,
            ymax=0.99,
            color="red",
            fill=False,
            label="Off",
        )
        ax.axvspan(
            0,
            num_four - 1,
            ymax=num_four / img_size,
            ymin=0.01,
            color="red",
            fill=False,
        )
        ax.axvspan(
            img_size - num_four,
            img_size - 1,
            ymin=1 - num_four / img_size,
            ymax=0.99,
            color="red",
            fill=False,
        )
        ax.axvspan(
            img_size - num_four,
            img_size - 1,
            ymax=num_four / img_size,
            ymin=0.01,
            color="red",
            fill=False,
        )


def compute_dr(i, img, sensitivity, num):
    """Compute the dynamic range for the true image. Also checks if one region
    exceeds the sensitivity. If so, mode is set accordingly for the following
    computations, including the predicted image.

    Parameters
    ----------
    i : int
        current index
    img : ndarray
        current image
    sensitivity : float
        upper limit for RMS value
    num : list
        side length of quadratic off region

    Returns
    -------
    dynamic_range
        computed dynamic range for the true image
    mode
        mode for following computations
    """
    num_four = num[2]
    # upper left
    rms1 = compute_rms(img, "rms1", num_four)
    # upper right
    rms2 = compute_rms(img, "rms2", num_four)
    # down left
    rms3 = compute_rms(img, "rms3", num_four)
    # down right
    rms4 = compute_rms(img, "rms4", num_four)

    if rms1 > sensitivity:
        if rms4 > sensitivity:
            print("Image {}: RMS exceeds upper left and down right.".format(i))
            mode = "rms1+4"
            rms = rms_comp(img, mode, num)
        else:
            print("Image {}: RMS exceeds upper left".format(i))
            mode = "rms1"
            rms = rms_comp(img, mode, num)
    elif rms2 > sensitivity:
        if rms3 > sensitivity:
            print("Image {}: RMS exceeds upper right and down left.".format(i))
            mode = "rms2+3"
            rms = rms_comp(img, mode, num)
        else:
            print("Image {}: RMS exceeds upper right".format(i))
            mode = "rms2"
            rms = rms_comp(img, mode, num)
    elif rms3 > sensitivity:
        print("Image {}: RMS exceeds down left".format(i))
        mode = "rms3"
        rms = rms_comp(img, mode, num)
    elif rms4 > sensitivity:
        print("Image {}: RMS exceeds down right".format(i))
        mode = "rms4"
        rms = rms_comp(img, mode, num)
    else:
        mode = None
        rms = rms_comp(img, mode, num)

    dynamic_range = img.max() / rms
    return dynamic_range, mode


def compute_dr_pred(img, mode, num):
    """Compute the dynamic range on the predicted image according to the mode set
    by compute_dr.

    Parameters
    ----------
    img : ndarray
        current image
    mode : str
        current mode
    num : list
        side length of quadratic off region

    Returns
    -------
    dynamic_range
        dynamic range for the predicted image
    """
    if mode == "rms1":
        rms = rms_comp(img, "rms1", num)

    elif mode == "rms2":
        rms = rms_comp(img, "rms2", num)

    elif mode == "rms3":
        rms = rms_comp(img, "rms3", num)

    elif mode == "rms4":
        rms = rms_comp(img, "rms4", num)

    elif mode == "rms1+4":
        rms = rms_comp(img, "rms1+4", num)

    elif mode == "rms2+3":
        rms = rms_comp(img, "rms2+3", num)

    elif mode is None:
        rms = rms_comp(img, None, num)

    dynamic_range = img.max() / rms
    return dynamic_range


def rms_comp(img, mode, num):
    """Calls compute_rms according to the mode for the remaining corners
    with the right size.

    Parameters
    ----------
    img : ndarray
        current image
    mode : str
        current mode
    num : list
        side length of quadratic off region

    Returns
    -------
    rms
        combined RMS value for the given image
    """
    num_three, num_two, num_four = num[0], num[1], num[2]
    rms = []
    if mode == "rms1":
        rms.extend([compute_rms(img, "rms2", num_three)])
        rms.extend([compute_rms(img, "rms3", num_three)])
        rms.extend([compute_rms(img, "rms4", num_three)])
        rms = combine_rms(rms)

    elif mode == "rms2":
        rms.extend([compute_rms(img, "rms1", num_three)])
        rms.extend([compute_rms(img, "rms3", num_three)])
        rms.extend([compute_rms(img, "rms4", num_three)])
        rms = combine_rms(rms)

    elif mode == "rms3":
        rms.extend([compute_rms(img, "rms1", num_three)])
        rms.extend([compute_rms(img, "rms2", num_three)])
        rms.extend([compute_rms(img, "rms4", num_three)])
        rms = combine_rms(rms)

    elif mode == "rms4":
        rms.extend([compute_rms(img, "rms1", num_three)])
        rms.extend([compute_rms(img, "rms2", num_three)])
        rms.extend([compute_rms(img, "rms3", num_three)])
        rms = combine_rms(rms)

    elif mode == "rms1+4":
        rms.extend([compute_rms(img, "rms2", num_two)])
        rms.extend([compute_rms(img, "rms3", num_two)])
        rms = combine_rms(rms)

    elif mode == "rms2+3":
        rms.extend([compute_rms(img, "rms1", num_two)])
        rms.extend([compute_rms(img, "rms4", num_two)])
        rms = combine_rms(rms)

    elif mode is None:
        rms.extend([compute_rms(img, "rms1", num_four)])
        rms.extend([compute_rms(img, "rms2", num_four)])
        rms.extend([compute_rms(img, "rms3", num_four)])
        rms.extend([compute_rms(img, "rms4", num_four)])
        rms = combine_rms(rms)
    assert rms.dtype == np.float64
    return rms


def compute_rms(img, corner, num):
    """Compute the RMS value for the given corner with the size determined by num.

    Parameters
    ----------
    img : int
        current index
    corner : str
        corner of image
    num : int
        size of the corner

    Returns
    -------
    rms
        RMS value of the given corner
    """
    if corner == "rms1":
        rms = np.sqrt((img[:num, :num] ** 2).mean())
    elif corner == "rms2":
        rms = np.sqrt((img[:num, -num:] ** 2).mean())
    elif corner == "rms3":
        rms = np.sqrt((img[-num:, :num] ** 2).mean())
    elif corner == "rms4":
        rms = np.sqrt((img[-num:, -num:] ** 2).mean())
    return rms


def combine_rms(rms):
    """Combine the RMS values of the image according to the number of values.

    Parameters
    ----------
    rms : list
        list of RMS values

    Returns
    -------
    rms
        combined RMS value for the image
    """
    if len(rms) == 2:
        rms = np.sqrt((rms[0] ** 2 + rms[1] ** 2) / 2)
    elif len(rms) == 3:
        rms = np.sqrt((rms[0] ** 2 + rms[1] ** 2 + rms[2] ** 2) / 3)
    elif len(rms) == 4:
        rms = np.sqrt((rms[0] ** 2 + rms[1] ** 2 + rms[2] ** 2 + rms[3] ** 2) / 4)
    return rms


def hist_difference(i, img_pred, img_truth, out_path):
    """Histogram the difference between the prediction and the truth

    Parameters
    ----------
    i : int
        number of picture
    img_pred : ndarray
        image of prediction
    img_truth : ndarray
        image of truth
    out_path : str
        Output path as defined in the makerc
    """
    x = np.abs(img_pred - img_truth).reshape(-1)
    plt.hist(x, label="Max Distance: {}".format(np.round(x.max(), 4)))
    plt.xlabel(r"Difference / a.u.")
    plt.ylabel(r"Number of pixels")
    plt.legend(loc="best")
    outpath = str(out_path) + "diff/hist_difference_{}.pdf".format(i)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)


def save_indices_and_data(indices, dr, outpath):
    """function for saving data in a csv file with indices

    Parameters
    ----------
    indices : list
        list of indices
    dr : ndarray
        data
    outpath : str
        Output path as defined in the makerc
    """
    df = pd.DataFrame(data=dr, index=indices)
    df.to_csv(outpath, index=True)


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


def blob_detection(i, img_pred, img_truth, out_path):
    """Blob detection for the prediction and the truth with the same kwargs.

    Parameters
    ----------
    i : int
        number of picture
    img_pred : ndarray
        image of prediction
    img_truth : ndarray
        image of truth
    out_path : str
        Output path as defined in makerc
    """
    plt.rcParams.update({"figure.max_open_warning": 0})

    tresh = img_truth.max() * 0.1
    kwargs = {
        "min_sigma": 1,
        "max_sigma": 10,
        "num_sigma": 100,
        "threshold": tresh,
        "overlap": 0.9,
    }
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

    make_axes_nice(fig, ax1, im1, r"Prediction")
    make_axes_nice(fig, ax2, im2, r"Truth")

    outpath = str(out_path) + "blob/blob_detection_{}.png".format(i)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)


def make_axes_nice(fig, ax, im, title):
    """Create nice colorbars for every axis in a subplot

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
    ax.set_title(title)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
