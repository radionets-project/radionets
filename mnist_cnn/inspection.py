import numpy as np
import matplotlib.pyplot as plt
import torch

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
    x_t = valid_ds.x.float().cuda()
    h = np.sqrt(x_t.shape[1]).astype(int)
    x_t = x_t.view(-1, h, h).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        pred = model(x_t).cpu()

    img = np.random.randint(0, len(pred))
    print('Image:', img)
    fig, (ax0, ax1, ax2, cax) = plt.subplots(ncols=4, figsize=(18, 6), gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    ax0.set_title('x')
    ax0.imshow(valid_ds.x[img].view(h, h))
    ax1.set_title('y_pred')
    im = ax1.imshow(pred[img].view(h, h), vmin=0, vmax=1)
    ax2.set_title('y_true')
    ax2.imshow(valid_ds.y[img].view(h, h), vmin=0, vmax=1)
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