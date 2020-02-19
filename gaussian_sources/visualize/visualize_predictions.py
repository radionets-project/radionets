import click
import matplotlib
import pandas as pd
import numpy as np
from mnist_cnn.visualize.utils import eval_model
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dl_framework.architectures as architecture
from dl_framework.model import load_pre_model
from dl_framework.data import do_normalisation, get_bundles, h5_dataset
from tqdm import tqdm
import re


@click.command()
@click.argument('arch', type=str)
@click.argument('pretrained_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('data_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('norm_path', type=click.Path(exists=False, dir_okay=True))
@click.argument('out_path', type=click.Path(exists=False, dir_okay=True))
@click.option('-log', type=bool, required=False)
@click.option('-index', type=int, required=False)
@click.option('-num', type=int, required=False)
def main(arch, pretrained_path, data_path, norm_path,
         out_path, index=None, log=False, num=None):
    # to prevent the localhost error from happening
    # first change the backende and second turn off
    # the interactive mode
    matplotlib.use("Agg")
    plt.ioff()
    bundle_paths = get_bundles(data_path)
    test = [
        path for path in bundle_paths
        if re.findall('fft_samp_test', path.name)
        ]
    test_ds = h5_dataset(test)

    if index is None:
        indices = np.random.randint(0, len(test_ds), size=num)
        img = [test_ds[i][0] for i in indices]
    else:
        img = test_ds[index][0]

    if log is True:
        img = torch.log(img)

    # get arch
    arch = getattr(architecture, arch)()

    # load pretrained model
    load_pre_model(arch, pretrained_path, visualize=True)

    if index is None:
        print('\nPlotting {} pictures.\n'.format(num))
        for i in tqdm(range(len(indices))):
            index = indices[i]
            img_reshaped = img[i].view(1, 2, 64, 64)
            norm = pd.read_csv(norm_path)
            img_normed = do_normalisation(img_reshaped, norm)

            # predict image
            prediction = eval_model(img_normed, arch)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

            inp = img_normed.numpy()

            inp_real = inp[0, 0, :]
            im1 = ax1.imshow(inp_real, cmap='RdBu', vmin=-inp_real.max(),
                             vmax=inp_real.max())
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax1.set_title(r'Real Input')
            fig.colorbar(im1, cax=cax, orientation='vertical')

            inp_imag = inp[0, 1, :]
            im2 = ax2.imshow(inp_imag, cmap='RdBu', vmin=-inp_imag.max(),
                             vmax=inp_imag.max())
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax2.set_title(r'Imaginary Input')
            fig.colorbar(im2, cax=cax, orientation='vertical')

            pred_img = prediction.reshape(64, 64).numpy()
            im3 = ax3.imshow(pred_img)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax3.set_title(r'Prediction')
            im4 = ax4.imshow(test_ds[index][1].reshape(64, 64))
            fig.colorbar(im4, cax=cax, orientation='vertical')

            # im4 = ax4.imshow(y_valid[index].reshape(64, 64))
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            ax4.set_title(r'Truth')
            fig.colorbar(im4, cax=cax, orientation='vertical')

            outpath = str(out_path).split('.')[0] + '_{}.{}'.format(i, str(out_path).split('.')[-1])
            plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
            plt.clf()
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    else:
        print('\nPlotting a single index.\n')
        img_reshaped = img.view(1, 2, 64, 64)
        norm = pd.read_csv(norm_path)
        img_normed = do_normalisation(img_reshaped, norm)

        # predict image
        prediction = eval_model(img_normed, arch)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        inp = img_normed.numpy()

        inp_real = inp[0, 0, :]
        im1 = ax1.imshow(inp_real, cmap='RdBu', vmin=-inp_real.max(),
                         vmax=inp_real.max())
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax1.set_title(r'Real Input')

        fig.colorbar(im1, cax=cax, orientation='vertical')
        inp_imag = inp[0, 1, :]
        im2 = ax2.imshow(inp_imag, cmap='RdBu', vmin=-inp_imag.max(),
                         vmax=inp_imag.max())
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax2.set_title(r'Imaginary Input')
        fig.colorbar(im2, cax=cax, orientation='vertical')

        pred_img = prediction.reshape(64, 64).numpy()
        im3 = ax3.imshow(pred_img)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax3.set_title(r'Prediction')
        im4 = ax4.imshow(test_ds[index][1].reshape(64, 64))
        fig.colorbar(im4, cax=cax, orientation='vertical')

        # im4 = ax4.imshow(y_valid[index].reshape(64, 64))
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax4.set_title(r'Truth')
        fig.colorbar(im4, cax=cax, orientation='vertical')
        plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0.01)
        plt.clf()
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)


if __name__ == '__main__':
    main()
