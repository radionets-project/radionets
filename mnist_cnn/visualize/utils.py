import matplotlib.pyplot as plt
import torch
import imp


def plot_mnist(img):
    plt.imshow(img, cmap="RdGy", vmin=-img.max(), vmax=img.max())
    plt.colorbar(label='Amplitude')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.tight_layout()


def load_pre_model(model, pretrained_path):
    model.load_state_dict(torch.load(pretrained_path))
    print('Pretrained model loaded and put to cuda.')
    return model


def load_architecture(arch_path):
    module = imp.load_source('get_model', arch_path)
    model = module.get_model()
    return model


def eval_model(img, model):
    model.eval()
    with torch.no_grad():
        pred = model(img.float())
    return pred
