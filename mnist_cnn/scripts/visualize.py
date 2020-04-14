import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dl_framework.inspection import reshape_2d
from mnist_cnn.scripts.utils import adjust_outpath
from tqdm import tqdm


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
            out = model_path/"predictions/"
            if os.path.exists(out) is False:
                os.mkdir(out)

            out_path = adjust_outpath(out, "prediction", form="pdf")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)


def plot_dataset(data, num_images, save=False):
    """
    Plot input and target images of a data set.

    Parameters
    ----------
    data: h5_dataset
        data set to be visualized
    num_images: int
        number of plotted images
    """
    for i in tqdm(range(num_images)):
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10, 8))

        inp1 = data[i][0][0]
        im1 = ax1.imshow(inp1, cmap="RdBu", vmin=-inp1.max(), vmax=inp1.max())
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax1.set_title(r"Real Input")
        fig.colorbar(im1, cax=cax, orientation="vertical")

        inp2 = data[i][0][1]
        im2 = ax2.imshow(inp2, cmap="RdBu", vmin=-inp2.max(), vmax=inp2.max())
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax2.set_title(r"Imag Input")
        fig.colorbar(im2, cax=cax, orientation="vertical")

        target = reshape_2d(data[i][1])[0]
        im3 = ax3.imshow(target, cmap="RdBu", vmin=-target.max(), vmax=target.max())
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax3.set_title(r"Target")
        fig.colorbar(im3, cax=cax, orientation="vertical")

        plt.tight_layout()

        if save:
            out = "./examples/"
            if os.path.exists(out) is False:
                os.mkdir(out)

            out_path = adjust_outpath(out, "dataset_example", form="pdf")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
