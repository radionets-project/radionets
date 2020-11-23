import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from radionets.dl_framework.data import do_normalisation, load_data
import radionets.dl_framework.architecture as architecture
from radionets.dl_framework.model import load_pre_model
from radionets.simulations.utils import adjust_outpath
from pathlib import Path

# from gaussian_sources.inspection import visualize_with_fourier


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


def load_pretrained_model(arch_name, model_path):
    """
    Load model architecture and pretrained weigths.

    Parameters
    ----------
    arch_name: str
        name of the architecture (architectures are in dl_framework.architectures)
    model_path: str
        path to pretrained model

    Returns
    -------
    arch: architecture object
        architecture with pretrained weigths
    """
    arch = getattr(architecture, arch_name)(63)
    load_pre_model(arch, model_path, visualize=True)
    return arch


def get_images(test_ds, num_images, norm_path=None):
    """
    Get n random test and truth images.

    Parameters
    ----------
    test_ds: h5_dataset
        data set with test images
    num_images: int
        number of test images
    norm_path: str
        path to normalization factors, if None: no normalization is applied

    Returns
    -------
    img_test: n 2d arrays
        test images
    img_true: n 2d arrays
        truth images
    """
    rand = torch.randint(0, len(test_ds), size=(num_images,))
    img_test = test_ds[rand][0]
    norm = "none"
    if norm_path != "none":
        norm = pd.read_csv(norm_path)
    img_test = do_normalisation(img_test, norm)
    img_true = test_ds[rand][1]
    # print(img_true.shape)
    if num_images == 1:
        img_test = img_test.unsqueeze(0)
        img_true = img_true.unsqueeze(0)
    return img_test, img_true


def eval_model(img, model):
    """
    Put model into eval mode and evaluate test images.

    Parameters
    ----------
    img: str
        test image
    model: architecture object
        architecture with pretrained weigths

    Returns
    -------
    pred: n 1d arrays
        predicted images
    """
    if len(img.shape) == (3):
        img = img.unsqueeze(0)
    model.eval()
    model.cuda()
    with torch.no_grad():
        pred = model(img.float().cuda())
    return pred.cpu()


def fft_pred(pred, truth, amp_phase=True):
    """
    Transform predicted image and true image to local domain.

    Parameters
    ----------
    pred: 4D array [1, channel, height, width]
        prediction from eval_model
    truth: 3D array [channel, height, width]
        true image
    amp_phase: Bool
        trained on Amp/Phase or Re/Im

    Returns
    -------
    ifft_pred, ifft_true: two 2D arrays [height, width]
        predicted and true image in local domain
    """
    a = pred[:, 0, :, :]
    b = pred[:, 1, :, :]

    a_true = truth[:, 0, :, :]
    b_true = truth[:, 1, :, :]

    if amp_phase:
        amp_pred_rescaled = (10 ** (10 * a) - 1) / 10 ** 10
        phase_pred = b

        amp_true_rescaled = (10 ** (10 * a_true) - 1) / 10 ** 10
        phase_true = b_true

        compl_pred = amp_pred_rescaled * np.exp(1j * phase_pred)
        compl_true = amp_true_rescaled * np.exp(1j * phase_true)
    else:
        compl_pred = a + 1j * b
        compl_true = a_true + 1j * b_true

    ifft_pred = np.fft.ifft2(compl_pred)
    ifft_true = np.fft.ifft2(compl_true)

    return np.absolute(ifft_pred), np.absolute(ifft_true)


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


def plot_loss(learn, model_path):
    """
    Plot train and valid loss of model.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    model_path: str
        path to trained model
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    save_path = model_path.with_suffix("")
    print(f"\nPlotting Loss for: {model_path.stem}\n")
    learn.recorder.plot_loss()
    plt.title(r"{}".format(str(model_path.stem).replace("_", " ")))
    plt.yscale("log")
    plt.savefig(f"{save_path}_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr(learn, model_path):
    """
    Plot learning rate of model.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    model_path: str
        path to trained model
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    save_path = model_path.with_suffix("")
    print(f"\nPlotting Learning rate for: {model_path.stem}\n")
    plt.plot(learn.recorder.lrs)
    plt.savefig(f"{save_path}_lr.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_lr_loss(learn, arch_name, out_path, skip_last):
    """
    Plot loss of learning rate finder.

    Parameters
    ----------
    learn: learner-object
        learner containing data and model
    arch_path: str
        name of the architecture
    out_path: str
        path to save loss plot
    skip_last: int
        skip n last points
    """
    # to prevent the localhost error from happening first change the backende and
    # second turn off the interactive mode
    mpl.use("Agg")
    plt.ioff()
    print(f"\nPlotting Lr vs Loss for architecture: {arch_name}\n")
    learn.recorder.plot_lr_find()
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / "lr_loss.pdf", bbox_inches="tight", pad_inches=0.01)
    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_results(inp, pred, truth, model_path, save=False):
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

            out_path = adjust_outpath(out, "/prediction", form="pdf")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def create_inspection_plots(learn, train_conf):
    test_ds = load_data(train_conf["data_path"], "test", fourier=train_conf["fourier"])
    img_test, img_true = get_images(test_ds, 5, train_conf["norm_path"])
    pred = eval_model(img_test.cuda(), learn.model)
    model_path = train_conf["model_path"]
    out_path = Path(model_path).parent
    if train_conf["fourier"]:
        for i in range(len(img_test)):
            visualize_with_fourier(
                i, img_test[i], pred[i], img_true[i], amp_phase=True, out_path=out_path
            )
    else:
        plot_results(
            img_test.cpu(),
            reshape_2d(pred.cpu()),
            reshape_2d(img_true),
            out_path,
            save=True,
        )


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
    cbar.set_label("Intensity / a.u.")
    # cbar.formatter.set_powerlimits((0, 0))
    # cbar.update_ticks()

def psnr(pred, truth):
    L = [np.amax(t)**2 for t in truth]
    psnr = [10*np.log10(L[i]/(np.mean((truth[i]-pred[i])**2))) for i in range(len(pred))]
    return np.mean(psnr)

def ssim(pred,true):
    mean_pred = [np.mean(p) for p in pred]
    std_pred = [np.std(p) for p in pred]
    mean_true = [np.mean(t) for t in true]
    std_true = [np.std(t) for t in true]
    cov = [1/(len(pred[i])**2-1)*np.sum((pred[i]-mean_pred[i])*(true[i]-mean_true[i])) for i in range(len(pred))]
    c1 = [(0.01*np.amax(t))**2 for t in true]
    c2 = [(0.03*np.amax(t))**2 for t in true]
    c3 = [c/2 for c in c2]
    l = [(2*mean_pred[i]*mean_true[i]+c1[i])/(mean_pred[i]**2+mean_true[i]**2+c1[i]) for i in range(len(pred))]
    c = [(2*std_pred[i]*std_true[i]+c2[i])/(std_pred[i]**2+std_true[i]**2+c2[i]) for i in range(len(pred))]
    s = [(cov[i]+c3[i])/(std_pred[i]*std_true[i]+c3[i]) for i in range(len(pred))]
    return np.mean([l[i]*c[i]*s[i] for i in range(len(l))])