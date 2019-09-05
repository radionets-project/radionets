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

def evaluate(valid_ds, model):
    x_t = valid_ds.x.float()
    rand = np.random.randint(0, len(x_t))
    img = x_t[rand].cuda()
    h = int(np.sqrt(img.shape))
    img = img.view(-1, h, h).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        pred = model(img).cpu()

    fig, (ax0, ax1, ax2, cax) = plt.subplots(ncols=4, figsize=(18, 6), gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    ax0.set_title('x')
    ax0.imshow(img.view(h, h).cpu())
    ax1.set_title('y_pred')
    im = ax1.imshow(pred.view(h, h), vmin=0, vmax=1)
    ax2.set_title('y_true')
    ax2.imshow(valid_ds.y[rand].view(h, h), vmin=0, vmax=1)
    fig.colorbar(im, cax=cax)
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