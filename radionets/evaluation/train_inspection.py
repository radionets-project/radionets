import click
import torch
import numpy as np
from pathlib import Path
from radionets.dl_framework.data import load_data
from radionets.evaluation.plotting import (
    visualize_with_fourier,
    plot_results,
    visualize_source_reconstruction,
    histogram_jet_angles,
    histogram_dynamic_ranges,
    histogram_ms_ssim,
)
from radionets.evaluation.utils import (
    create_databunch,
    reshape_2d,
    load_pretrained_model,
    get_images,
    eval_model,
    get_ifft,
    pad_unsqueeze,
)
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.dynamic_range import calc_dr
from pytorch_msssim import ms_ssim
from tqdm import tqdm


def get_prediction(conf, num_images=None, rand=False):
    test_ds = load_data(
        conf["data_path"],
        mode="test",
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )
    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(
        test_ds, num_images, norm_path=conf["norm_path"], rand=rand
    )
    img_size = img_test.shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    pred = eval_model(img_test, model)
    return pred, img_test, img_true


def get_separate_prediction(conf, num_images=None, rand=False):
    """Get predictions for separate architectures.

    Parameters
    ----------
    conf : dict
        contains configurations
    num_images : int, optional
        number of evaluation images, by default None
    rand : bool, optional
        if true, selects random images, by default False

    Returns
    -------
    tuple of torch tensors
        predictions, input and true images
    """
    test_ds = load_data(
        conf["data_path"],
        mode="test",
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )
    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(
        test_ds, num_images, norm_path=conf["norm_path"], rand=rand
    )
    img_size = img_test.shape[-1]
    model_1 = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    model_2 = load_pretrained_model(conf["arch_name_2"], conf["model_path_2"], img_size)
    pred_1 = eval_model(img_test, model_1)
    pred_2 = eval_model(img_test, model_2)
    pred = torch.cat((pred_1, pred_2), dim=1)
    return pred, img_test, img_true


def create_inspection_plots(conf, num_images=3, rand=False):
    if conf["separate"]:
        pred, img_test, img_true = get_separate_prediction(conf, num_images, rand=rand)
    else:
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
                plot_format=conf["format"],
            )
    else:
        plot_results(
            img_test.cpu(),
            reshape_2d(pred.cpu()),
            reshape_2d(img_true),
            out_path,
            save=True,
            plot_format=conf["format"],
        )


def create_source_plots(conf, num_images=3, rand=False):
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
    if conf["separate"]:
        pred, img_test, img_true = get_separate_prediction(conf, num_images, rand=rand)
    else:
        pred, img_test, img_true = get_prediction(conf, num_images, rand=rand)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    if not conf["fourier"]:
        click.echo("\n This is not a fourier dataset.\n")

    pred = pred.numpy()

    # inverse fourier transformation for prediction
    ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

    # inverse fourier transform for truth
    ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

    for i, (pred, truth) in enumerate(zip(ifft_pred, ifft_truth)):
        visualize_source_reconstruction(
            pred,
            truth,
            out_path,
            i,
            dr=conf["vis_dr"],
            blobs=conf["vis_blobs"],
            msssim=conf["vis_ms_ssim"],
            plot_format=conf["format"],
        )

    return np.abs(ifft_pred), np.abs(ifft_truth)


def evaluate_viewing_angle(conf):
    # if conf["separate"]:
    #     pred, img_test, img_true = get_separate_prediction(conf)
    # else:
    #     pred, img_test, img_true = get_prediction(conf)
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)

    alpha_truths = []
    alpha_preds = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])
        m_truth, n_truth, alpha_truth = calc_jet_angle(torch.tensor(ifft_truth))
        m_pred, n_pred, alpha_pred = calc_jet_angle(torch.tensor(ifft_pred))
        alpha_truths.extend(alpha_truth)
        alpha_preds.extend(alpha_pred)

    alpha_truths = torch.tensor(alpha_truths)
    alpha_preds = torch.tensor(alpha_preds)
    histogram_jet_angles(
        alpha_truths, alpha_preds, out_path, plot_format=conf["format"],
    )


def evaluate_dynamic_range(conf):
    # if conf["separate"]:
    #     pred, img_test, img_true = get_separate_prediction(conf)
    # else:
    #     pred, img_test, img_true = get_prediction(conf)
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)

    dr_truths = np.array([])
    dr_preds = np.array([])

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

        dr_truth, dr_pred, _, _ = calc_dr(ifft_truth, ifft_pred)
        dr_truths = np.append(dr_truths, dr_truth)
        dr_preds = np.append(dr_preds, dr_pred)

    click.echo(
        f"\nMean dynamic range for true source distributions:\
            {round(dr_truths.mean())}\n"
    )
    click.echo(
        f"\nMean dynamic range for predicted source distributions:\
            {round(dr_preds.mean())}\n"
    )

    histogram_dynamic_ranges(
        dr_truths, dr_preds, out_path, plot_format=conf["format"],
    )


def evaluate_ms_ssim(conf):
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)

    vals = []

    if img_size < 160:
        click.echo(
            "\nThis is only a placeholder!\
                Images too small for meaningful ms ssim calculations.\n"
        )

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

        ifft_truth = pad_unsqueeze(torch.tensor(ifft_truth))
        ifft_pred = pad_unsqueeze(torch.tensor(ifft_pred))
        vals.extend(
            [
                ms_ssim(pred.unsqueeze(0), truth.unsqueeze(0), data_range=truth.max())
                for pred, truth in zip(ifft_pred, ifft_truth)
            ]
        )

    click.echo("\nCreating ms-ssim histogram.\n")
    vals = torch.tensor(vals)
    histogram_ms_ssim(
        vals, out_path, plot_format=conf["format"],
    )

    click.echo(f"\nThe mean ms-ssim value is {vals.mean()}.\n")
