import numpy as np
import matplotlib.pyplot as plt
import torch
from training import view_tfm

def training_stats(run):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    run.recorder.plot_lr()
    plt.subplot(132)
    run.recorder.plot_loss()
#     plt.subplot(133)
#     run.recorder.plot()
    plt.tight_layout()

def get_eval_img(valid_ds, model):
    x_t = valid_ds.x.float()
    rand = np.random.randint(0, len(x_t))
    img = x_t[rand].cuda()
    h = int(np.sqrt(img.shape))
    img = img.view(-1, h, h).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()
    return img, pred, h, rand

def evaluate_model(valid_ds, model, nrows=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(18, 6*nrows),
                             gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    
    for i in range(nrows):
        img, pred, h, rand = get_eval_img(valid_ds, model)
        axes[i][0].set_title('x')
        axes[i][0].imshow(img.view(h, h).cpu(), cmap='RdGy_r', vmax=img.max(), vmin=-img.max())
        axes[i][1].set_title('y_pred')
        im = axes[i][1].imshow(pred.view(h, h), vmin=0, vmax=1)
        axes[i][2].set_title('y_true')
        axes[i][2].imshow(valid_ds.y[rand].view(h, h), vmin=0, vmax=1)
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