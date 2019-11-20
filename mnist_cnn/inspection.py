import numpy as np
import matplotlib.pyplot as plt
import torch
from dl_framework.callbacks import view_tfm
import pandas as pd
from mnist_cnn.utils import normalize


def training_stats(run):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    run.recorder.plot_lr()
    plt.subplot(132)
    run.recorder.plot_loss()
    plt.tight_layout()


def get_normalization(img, norm_path):
    norm = pd.read_csv(norm_path)
    train_mean = torch.tensor(norm['train_mean'].values[0]).float()
    train_std = torch.tensor(norm['train_std'].values[0]).float()
    img_mean = img[~torch.isinf(img)].mean()
    img[torch.isinf(img)] = img_mean
    img_normed = normalize(img, train_mean, train_std)
    assert not torch.isinf(img_normed).any()
    return img_normed


def get_eval_img(valid_ds, model, norm_path):
    x_t = valid_ds.x.float()
    rand = np.random.randint(0, len(x_t))
    img = x_t[rand].cuda()
    img = get_normalization(img, norm_path)
    h = int(np.sqrt(img.shape))
    img = img.view(-1, h, h).unsqueeze(1)
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
        axes[i][0].imshow(img.view(h, h).cpu(), cmap='RdGy_r',
                          vmax=img.max(), vmin=-img.max())
        axes[i][1].set_title('y_pred')
        im = axes[i][1].imshow(pred.view(h, h), vmin=valid_ds.y[rand].min(),
                               vmax=valid_ds.y[rand].max())
        axes[i][2].set_title('y_true')
        axes[i][2].imshow(valid_ds.y[rand].view(h, h),
                          vmin=valid_ds.y[rand].min(),
                          vmax=valid_ds.y[rand].max())
        fig.colorbar(im, cax=axes[i][3])
    plt.tight_layout()


def test_initialization(dl, model, layer):
    x, _ = next(iter(dl))
    mnist_view = view_tfm(1, 64, 64)
    x = mnist_view(x).cuda()
    print('mean:', x.mean())
    print('std:', x.std())
    p = model[layer](x)
    print('')
    print('after layer ' + str(layer))
    print('mean:', p.mean())
    print('std:', p.std())
