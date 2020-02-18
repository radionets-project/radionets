import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
from dl_framework.data import do_normalisation
from mnist_cnn.visualize.utils import eval_model
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
                               #norm=LogNorm(vmin=1e-6),
                               #vmin=valid_ds[rand][1].min(),
                               #vmax=valid_ds[rand][1].max()
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


def visualize_without_fourier(i, index, img, img_y, arch, out_path):
    print(index)
    img_reshaped = img[i].view(1, 2, 64, 64)

    # predict image
    prediction = eval_model(img_reshaped, arch)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    inp = img_reshaped.numpy()

    inp_real = inp[0, 0, :]
    inp_imag = inp[0, 1, :]

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

    pred_img = prediction.reshape(64, 64).numpy()
    im3 = ax3.imshow(pred_img)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax3.set_title(r'Prediction')
    im4 = ax4.imshow(img_y[index].reshape(64, 64))
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


def visualize_with_fourier(i, index, img, img_y, arch, out_path):
    img_reshaped = img[i].view(1, 2, 64, 64)
    img_y_reshaped = img_y[i].view(1, 2, 64, 64)

    # predict image
    prediction = eval_model(img_reshaped, arch)

    real_pred = prediction[0, 0, :].numpy()
    imag_pred = prediction[0, 1, :].numpy()

    inp = img_reshaped.numpy()
    inp_real = inp[0, 0, :]
    inp_imag = inp[0, 1, :]

    inp_y = img_y_reshaped.numpy()

    real_truth = inp_y[0, 0, :]
    imag_truth = inp_y[0, 1, :]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 10))

    im1 = ax1.imshow(inp_real, cmap='RdBu')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax1.set_title(r'Real Input')
    fig.colorbar(im1, cax=cax, orientation='vertical')

    im2 = ax2.imshow(real_pred.reshape(64, 64), cmap='RdBu')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax2.set_title(r'Real Prediction')
    fig.colorbar(im2, cax=cax, orientation='vertical')

    im3 = ax3.imshow(real_truth, cmap='RdBu')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax3.set_title(r'Real Truth')
    fig.colorbar(im3, cax=cax, orientation='vertical')

    im4 = ax4.imshow(inp_imag, cmap='RdBu')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax4.set_title(r'Imaginary Input')
    fig.colorbar(im4, cax=cax, orientation='vertical')

    im5 = ax5.imshow(imag_pred.reshape(64, 64), cmap='RdBu')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax5.set_title(r'Imaginary Prediction')
    fig.colorbar(im5, cax=cax, orientation='vertical')

    im6 = ax6.imshow(imag_truth, cmap='RdBu')
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax6.set_title(r'Imaginray Truth')
    fig.colorbar(im6, cax=cax, orientation='vertical')

    outpath = str(out_path).split('.')[0] + '_{}.{}'.format(i, str(out_path).split('.')[-1])
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
