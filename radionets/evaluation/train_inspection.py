import click
import torch
import numpy as np
from pathlib import Path
from radionets.dl_framework.data import load_data
from radionets.evaluation.plotting import (
    visualize_with_fourier,
    plot_results,
    visualize_source_reconstruction,
)
from radionets.evaluation.utils import (
    reshape_2d,
    load_pretrained_model,
    get_images,
    eval_model,
)


def get_prediction(conf, num_images=None, rand=False):
    test_ds = load_data(
        conf["data_path"],
        mode="test",
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )
    model = load_pretrained_model(conf["arch_name"], conf["model_path"])
    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(
        test_ds, num_images, norm_path=conf["norm_path"], rand=rand
    )
    pred = eval_model(img_test, model)
    return pred, img_test, img_true


def create_inspection_plots(conf, num_images=3, rand=False):
    pred, img_test, img_true = get_prediction(conf, num_images, rand=rand)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation/"
    out_path.mkdir(parents=True, exist_ok=True)
    if conf["fourier"]:
        for i in range(len(img_test)):
            visualize_with_fourier(
                i,
                img_test[i],
                pred[i],
                img_true[i],
                amp_phase=conf["amp_phase"],
                out_path=out_path,
            )
    else:
        plot_results(
            img_test.cpu(),
            reshape_2d(pred.cpu()),
            reshape_2d(img_true),
            out_path,
            save=True,
        )


def create_source_plots(conf, num_images=3, rand=False):
    # (i, real_pred, imag_pred, real_truth, imag_truth, amp_phase, out_path):
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
    pred, img_test, img_true = get_prediction(conf, num_images, rand=rand)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    if not conf["fourier"]:
        click.echo("\n This is not a fourier dataset.\n")

    if conf["amp_phase"]:
        pred = pred.numpy()
        amp_pred = 10 ** (10 * pred[:, 0] - 10) - 1e-10
        amp_true = 10 ** (10 * img_true[:, 0] - 10) - 1e-10

        a = amp_pred * np.cos(pred[:, 1])
        b = amp_pred * np.sin(pred[:, 1])
        compl_pred = a + b * 1j

        a = amp_true * np.cos(img_true[:, 1])
        b = amp_true * np.sin(img_true[:, 1])
        compl_truth = a + b * 1j
    else:
        compl_pred = pred[:, 0] + pred[:, 1] * 1j
        compl_truth = img_true[:, 0] + img_true[:, 1] * 1j

    # inverse fourier transformation for prediction
    ifft_pred = np.abs(np.fft.ifft2(compl_pred))

    # inverse fourier transform for truth
    ifft_truth = np.abs(np.fft.ifft2(compl_truth))

    for i, (pred, truth) in enumerate(zip(ifft_pred, ifft_truth)):
        visualize_source_reconstruction(pred, truth, out_path, i)

    return np.abs(ifft_pred), np.abs(ifft_truth)
