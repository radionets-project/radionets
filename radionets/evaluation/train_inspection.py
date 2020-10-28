import torch
from pathlib import Path
import pandas as pd
from radionets.dl_framework.model import load_pre_model
from radionets.dl_framework.data import load_data, do_normalisation
import radionets.dl_framework.architecture as architecture
from radionets.evaluation.plotting import visualize_with_fourier, plot_results
from radionets.evaluation.utils import reshape_2d


def load_pretrained_model(arch_name, model_path, img_size=63):
    """
    Load model architecture and pretrained weigths.

    Parameters
    ----------
    arch_name: str
        name of the architecture (architectures are in dl_framework.architectures)
    model_path: str
        path to pretrained model

    Returns
    -------
    arch: architecture object
        architecture with pretrained weigths
    """
    if (
        arch_name == "filter_deep"
        or arch_name == "filter_deep_amp"
        or arch_name == "filter_deep_phase"
    ):
        arch = getattr(architecture, arch_name)(img_size)
    else:
        arch = getattr(architecture, arch_name)()
    load_pre_model(arch, model_path, visualize=True)
    return arch


def get_images(test_ds, num_images, norm_path=None):
    """
    Get n random test and truth images.

    Parameters
    ----------
    test_ds: h5_dataset
        data set with test images
    num_images: int
        number of test images
    norm_path: str
        path to normalization factors, if None: no normalization is applied

    Returns
    -------
    img_test: n 2d arrays
        test images
    img_true: n 2d arrays
        truth images
    """
    rand = torch.randint(0, len(test_ds), size=(num_images,))
    img_test = test_ds[rand][0]
    norm = "none"
    if norm_path != "none":
        norm = pd.read_csv(norm_path)
    img_test = do_normalisation(img_test, norm)
    img_true = test_ds[rand][1]
    # print(img_true.shape)
    if num_images == 1:
        img_test = img_test.unsqueeze(0)
        img_true = img_true.unsqueeze(0)
    return img_test, img_true


def eval_model(img, model):
    """
    Put model into eval mode and evaluate test images.

    Parameters
    ----------
    img: str
        test image
    model: architecture object
        architecture with pretrained weigths

    Returns
    -------
    pred: n 1d arrays
        predicted images
    """
    if len(img.shape) == (3):
        img = img.unsqueeze(0)
    model.eval()
    model.cuda()
    with torch.no_grad():
        pred = model(img.float().cuda())
    return pred.cpu()


def create_inspection_plots(learn, train_conf, num_images=3):
    test_ds = load_data(train_conf["data_path"], "test", fourier=train_conf["fourier"])
    img_test, img_true = get_images(test_ds, num_images, train_conf["norm_path"])
    pred = eval_model(img_test.cuda(), learn.model)
    model_path = train_conf["model_path"]
    out_path = Path(model_path).parent
    if train_conf["fourier"]:
        for i in range(len(img_test)):
            visualize_with_fourier(
                i, img_test[i], pred[i], img_true[i], amp_phase=True, out_path=out_path
            )
    else:
        plot_results(
            img_test.cpu(),
            reshape_2d(pred.cpu()),
            reshape_2d(img_true),
            out_path,
            save=True,
        )
