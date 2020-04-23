import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    # inp = inp[:, 0].unsqueeze(1)
    # pred = pred[:, 0]
    # truth = truth[:, 0]  # .unsqueeze(1)
    # print(inp.shape)
    # print(pred.shape)
    # print(truth.shape)
    for i in tqdm(range(len(inp))):
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 10))

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

        pre_amp = pred[i][0]
        im3 = ax3.imshow(pre_amp, cmap="RdBu", vmin=-pre_amp.max(), vmax=pre_amp.max())
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax3.set_title(r"Prediction Amp")
        fig.colorbar(im3, cax=cax, orientation="vertical")

        # pre_phase = pred[i][1]
        im4 = ax4.imshow(imag, cmap="RdBu", vmin=-imag.max(), vmax=imag.max())
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax4.set_title(r"Prediction Phase")
        fig.colorbar(im4, cax=cax, orientation="vertical")

        true_amp = truth[i][0]
        im5 = ax5.imshow(true_amp, cmap="RdBu", vmin=-true_amp.max(), vmax=true_amp.max())
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax5.set_title(r"Truth Amp")
        fig.colorbar(im5, cax=cax, orientation="vertical")

        true_phase = truth[i][1]
        im6 = ax6.imshow(true_phase, cmap="RdBu", vmin=-true_phase.max(), vmax=true_phase.max())
        divider = make_axes_locatable(ax6)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax6.set_title(r"Truth Phase")
        fig.colorbar(im6, cax=cax, orientation="vertical")

        plt.tight_layout()

        if save:
            out = model_path/"predictions/"
            if os.path.exists(out) is False:
                os.mkdir(out)

            out_path = adjust_outpath(out, "/prediction", form="pdf")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)