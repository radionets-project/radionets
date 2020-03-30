import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
from dl_framework.data import do_normalisation
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    rand = np.random.randint(0, len(valid_ds))
    img = valid_ds[rand][0].cuda()
    norm = pd.read_csv(norm_path)
    img = do_normalisation(img, norm)
    h = int(np.sqrt(img.shape[1]))
    img = img.view(-1, h, h).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()
    return img, pred, h, rand


def evaluate_model(valid_ds, model, norm_path, nrows=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 6*nrows),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

    for i in range(nrows):
        img, pred, h, rand = get_eval_img(valid_ds, model, norm_path)
        axes[i][0].set_title('x')
        axes[i][0].imshow(img[:, 0].view(h, h).cpu(), cmap='RdGy_r',
                          vmax=img.max(), vmin=-img.max())
        axes[i][1].set_title('y_pred')
        im = axes[i][1].imshow(pred.view(h, h),
                               # norm=LogNorm(vmin=1e-6),
                               # vmin=valid_ds[rand][1].min(),
                               # vmax=valid_ds[rand][1].max()
                               )
        axes[i][2].set_title('y_true')
        axes[i][2].imshow(valid_ds[rand][1].view(h, h),
                          vmin=valid_ds[rand][1].min(),
                          vmax=valid_ds[rand][1].max())
        fig.colorbar(im, cax=axes[i][3])
    plt.tight_layout()


def plot_loss(learn, model_path):
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
    print('\nPlotting Lr vs Loss for architecture: {}\n'.format(arch_name))
    learn.recorder_lr_find.plot(skip_last, save=True)
    # plt.yscale('log')
    plt.savefig('./models/lr_loss.pdf', bbox_inches='tight', pad_inches=0.01)
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
    im4 = ax4.imshow(img_truth)
    fig.colorbar(im4, cax=cax, orientation='vertical')

    # im4 = ax4.imshow(y_valid[index].reshape(64, 64))
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax4.set_title(r'Truth')
    fig.colorbar(im4, cax=cax, orientation='vertical')

    outpath = str(out_path) + "prediction_{}.png".format(i)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def visualize_with_fourier(i, img_input, img_pred, img_truth, out_path):
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

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))

    im1 = ax1.imshow(inp_real, cmap='RdBu')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax1.set_title(r'Real Input')
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    im2 = ax2.imshow(real_pred, cmap='RdBu')
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

    im5 = ax5.imshow(imag_pred, cmap='RdBu')
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


def visualize_fft(i, real_pred, imag_pred, real_truth, imag_truth, out_path):
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
    a = real_pred * np.cos(imag_pred)
    b = real_pred * np.sin(imag_pred)
    # compl_pred = real_pred + imag_pred * 1j
    compl_pred = a + b * 1j

    # inverse fourier transformation
    ifft_pred = np.fft.ifft2((compl_pred))

    # create (complex) input for inverse fourier transformation for prediction
    a = real_truth * np.cos(imag_truth)
    b = real_truth * np.sin(imag_truth)
    # compl_truth = real_truth + imag_truth * 1j
    compl_truth = a + b * 1j

    # inverse fourier transform
    ifft_truth = np.fft.ifft2(compl_truth)

    # plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    im1 = ax1.imshow(np.abs(ifft_pred), cmap='RdBu')
    im2 = ax2.imshow(np.abs(ifft_truth), cmap='RdBu')
    ax1.set_title(r'FFT Prediction')
    ax2.set_title(r'FFT Truth')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    outpath = str(out_path) + "fft_pred_{}.png".format(i)
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
