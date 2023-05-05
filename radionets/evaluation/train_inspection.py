import re
import time
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from tqdm import tqdm

from radionets.dl_framework.clustering import gmmClustering, spectralClustering
from radionets.dl_framework.data import MojaveDataset, get_bundles, load_data
from radionets.dl_framework.metrics import binary_accuracy, iou_YOLOv6
from radionets.evaluation.blob_detection import calc_blobs, crop_first_component
from radionets.evaluation.contour import analyse_intensity, area_of_contour
from radionets.evaluation.coordinates import pixel2coordinate
from radionets.evaluation.dynamic_range import calc_dr
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.plotting import (
    hist_point,
    histogram_area,
    histogram_dynamic_ranges,
    histogram_gan_sources,
    histogram_jet_angles,
    histogram_mean_diff,
    histogram_ms_ssim,
    histogram_peak_intensity,
    histogram_sum_intensity,
    plot_beam_props,
    plot_contour,
    plot_counterjet_eval,
    plot_hist_counterjet,
    plot_hist_velocity,
    plot_hist_velocity_unc,
    plot_length_point,
    plot_loss,
    plot_results,
    plot_yolo_clustering,
    plot_yolo_eval,
    plot_yolo_mojave,
    plot_yolo_post_clustering,
    plot_yolo_velocity,
    visualize_sampled_unc,
    visualize_source_reconstruction,
    visualize_uncertainty,
    visualize_with_fourier,
    visualize_with_fourier_diff,
)
from radionets.evaluation.pointsources import flux_comparison
from radionets.evaluation.utils import (
    SymLogNorm,
    apply_normalization,
    apply_symmetry,
    calculate_velocity,
    create_databunch,
    create_sampled_databunch,
    eval_model,
    get_ifft,
    get_images,
    load_pretrained_model,
    mergeDictionary,
    preprocessing,
    process_prediction,
    read_pred,
    rescale_normalization,
    reshape_2d,
    sample_images,
    sampled_dataset,
    save_pred,
    scaling_log10_noisecut,
    sym_new,
    yolo_apply_nms,
    yolo_df,
    yolo_linear_fit,
)


def create_predictions(conf):
    if conf["model_path_2"] != "none":
        img = get_separate_prediction(conf)
    else:
        img = get_prediction(conf)

    model_path = conf["model_path"]
    name_model = Path(model_path).stem

    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_path) + f"/predictions_{name_model}.h5"

    if not conf["fourier"]:
        click.echo("\n This is not a fourier dataset.\n")

    save_pred(out_path, img)


def get_prediction(conf, mode="test"):
    test_ds = load_data(
        conf["data_path"],
        mode=mode,
        fourier=conf["fourier"],
        source_list=conf["source_list"],
    )

    num_images = conf["num_images"]
    rand = conf["random"]

    if num_images is None:
        num_images = len(test_ds)

    img_test, img_true, indices = get_images(test_ds, num_images, rand=rand)

    img_size = img_test.shape[-1]
    model, norm_dict = load_pretrained_model(
        conf["arch_name"], conf["model_path"], img_size
    )

    # Rescale if necessary
    img_test, norm_dict = apply_normalization(img_test, norm_dict)
    pred = eval_model(img_test, model)
    pred = rescale_normalization(pred, norm_dict)

    images = {"pred": pred, "inp": img_test, "true": img_true}

    if pred.shape[1] == 4:
        unc_amp = pred[:, 1, :]
        unc_phase = pred[:, 3, :]
        unc = torch.stack([unc_amp, unc_phase], dim=1)
        pred_1 = pred[:, 0, :]
        pred_2 = pred[:, 2, :]
        pred = torch.stack((pred_1, pred_2), dim=1)
        images["unc"] = unc
        images["pred"] = pred
        images["indices"] = indices

    if images["pred"].shape[-1] == 128:
        images = apply_symmetry(images)

    return images


def get_separate_prediction(conf):
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

    num_images = conf["num_images"]
    rand = conf["random"]

    if num_images is None:
        num_images = len(test_ds)
    img_test, img_true = get_images(test_ds, num_images, rand=rand)
    img_size = img_test.shape[-1]
    model_1, norm_dict = load_pretrained_model(
        conf["arch_name"], conf["model_path"], img_size
    )
    model_2, norm_dict = load_pretrained_model(
        conf["arch_name_2"], conf["model_path_2"], img_size
    )

    pred_1 = eval_model(img_test, model_1)
    pred_2 = eval_model(img_test, model_2)

    # test for uncertainty
    if pred_1.shape[1] == 2:
        pred_1 = pred_1[:, 0, :].unsqueeze(1)
        pred_2 = pred_2[:, 0, :].unsqueeze(1)

    pred = torch.cat((pred_1, pred_2), dim=1)
    return pred, img_test, img_true


def create_inspection_plots(conf, num_images=3, rand=False):
    model_path = conf["model_path"]
    path = str(Path(model_path).parent / "evaluation")
    name_model = Path(model_path).stem
    path += f"/predictions_{name_model}.h5"
    out_path = Path(model_path).parent / "evaluation/"

    img = read_pred(path)
    if img["pred"].shape[1] == 4:
        pred_1 = np.expand_dims(img["pred"][:, 0, :], axis=1)
        pred_2 = np.expand_dims(img["pred"][:, 2, :], axis=1)
        img["pred"] = np.concatenate([pred_1, pred_2], axis=1)
    if conf["fourier"]:
        if conf["diff"]:
            for i in range(len(img["inp"])):
                visualize_with_fourier_diff(
                    i,
                    img["pred"][i],
                    img["true"][i],
                    amp_phase=conf["amp_phase"],
                    out_path=out_path,
                    plot_format=conf["format"],
                )
        else:
            for i in range(len(img["inp"])):
                visualize_with_fourier(
                    i,
                    img["inp"][i],
                    img["pred"][i],
                    img["true"][i],
                    amp_phase=conf["amp_phase"],
                    out_path=out_path,
                    plot_format=conf["format"],
                )
    else:
        plot_results(
            # img_test.cpu(),
            # reshape_2d(pred.cpu()),
            # reshape_2d(img_true),
            out_path,
            save=True,
            plot_format=conf["format"],
        )


def after_training_plots(conf, num_images=3, rand=False, diff=True):
    """Create quickly inspection plots right after the training finished. Note, that
    these images are taken from the validation dataset and are therefore known by the
    network.

    Parameters
    ----------
    conf : dict
        contains configurations
    num_images : int, optional
        number of images to plot, by default 3
    rand : bool, optional
        take images randomly or from the beginning of the dataset, by default False
    diff : bool, optional
        show the difference or the input, by default True
    """
    conf["num_images"] = num_images
    conf["random"] = rand
    pred, img_test, img_true = get_prediction(conf, mode="valid")

    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation/"

    if conf["fourier"]:
        if diff:
            for i in range(len(img_test)):
                visualize_with_fourier_diff(
                    i,
                    pred[i],
                    img_true[i],
                    amp_phase=conf["amp_phase"],
                    out_path=out_path,
                    plot_format=conf["format"],
                )
        else:
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
    model_path = conf["model_path"]
    path = str(Path(model_path).parent / "evaluation")
    name_model = Path(model_path).stem
    path += f"/predictions_{name_model}.h5"
    out_path = Path(model_path).parent / "evaluation"

    img = read_pred(path)

    # inverse fourier transformation for prediction
    ifft_pred = get_ifft(img["pred"], amp_phase=conf["amp_phase"])

    # inverse fourier transform for truth
    ifft_truth = get_ifft(img["true"], amp_phase=conf["amp_phase"])

    if len(ifft_pred.shape) == 2:
        ifft_pred = np.expand_dims(ifft_pred, axis=0)
        ifft_truth = np.expand_dims(ifft_truth, axis=0)

    for i, (pred, truth) in enumerate(zip(ifft_pred, ifft_truth)):
        visualize_source_reconstruction(
            pred,
            truth,
            out_path,
            i,
            dr=conf["vis_dr"],
            msssim=conf["vis_ms_ssim"],
            plot_format=conf["format"],
        )

    return np.abs(ifft_pred), np.abs(ifft_truth)


def create_contour_plots(conf, num_images=3, rand=False):
    model_path = conf["model_path"]
    path = str(Path(model_path).parent / "evaluation")
    name_model = Path(model_path).stem
    path += f"/predictions_{name_model}.h5"
    out_path = Path(model_path).parent / "evaluation"

    if not conf["fourier"]:
        click.echo("\n This is not a fourier dataset.\n")

    img = read_pred(path)

    # inverse fourier transformation for prediction
    ifft_pred = get_ifft(img["pred"], amp_phase=conf["amp_phase"])

    # inverse fourier transform for truth
    ifft_truth = get_ifft(img["true"], amp_phase=conf["amp_phase"])

    for i, (pred, truth) in enumerate(zip(ifft_pred, ifft_truth)):
        plot_contour(pred, truth, out_path, i, plot_format=conf["format"])


def create_uncertainty_plots(conf, num_images=3, rand=False):
    """Create uncertainty plots in Fourier and image space.

    Parameters
    ----------
    conf : dict
        config information
    num_images : int, optional
        number of images to be plotted, by default 3
    rand : bool, optional
        True, if images should be taken randomly, by default False
    """
    model_path = conf["model_path"]
    path = str(Path(model_path).parent / "evaluation")
    name_model = Path(model_path).stem
    predictions_path = path + f"/predictions_{name_model}.h5"
    out_path = Path(model_path).parent / "evaluation/"

    img = read_pred(predictions_path)
    for i in range(len(img["pred"])):
        visualize_uncertainty(
            i,
            img["pred"][i],
            img["true"][i],
            img["unc"][i],
            amp_phase=conf["amp_phase"],
            out_path=out_path,
            plot_format=conf["format"],
        )

    name_model = Path(model_path).stem
    sampling_path = path + f"/sampled_imgs_{name_model}.h5"
    test_ds = sampled_dataset(sampling_path)
    mean_imgs, std_imgs, true_imgs = get_images(
        test_ds, num_images, rand=rand, indices=img["indices"]
    )

    # loop
    for i, (mean, std, true) in enumerate(zip(mean_imgs, std_imgs, true_imgs)):
        visualize_sampled_unc(
            i,
            mean,
            std,
            true,
            out_path=out_path,
            plot_format=conf["format"],
        )


def evaluate_viewing_angle(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    alpha_truths = []
    alpha_preds = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        m_truth, n_truth, alpha_truth = calc_jet_angle(torch.tensor(ifft_truth))
        m_pred, n_pred, alpha_pred = calc_jet_angle(torch.tensor(ifft_pred))

        alpha_truths.extend(abs(alpha_truth))
        alpha_preds.extend(abs(alpha_pred))

    alpha_truths = torch.tensor(alpha_truths)
    alpha_preds = torch.tensor(alpha_preds)

    click.echo("\nCreating histogram of jet angles.\n")
    dif = (alpha_preds - alpha_truths).numpy()
    histogram_jet_angles(dif, out_path, plot_format=conf["format"])
    if conf["save_vals"]:
        click.echo("\nSaving jet angle offsets.\n")
        out = Path(conf["save_path"])
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "jet_angles.txt", dif)


def evaluate_dynamic_range(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    dr_truths = np.array([])
    dr_preds = np.array([])

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

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

    click.echo("\nCreating histogram of dynamic ranges.\n")
    histogram_dynamic_ranges(dr_truths, dr_preds, out_path, plot_format=conf["format"])


def evaluate_ms_ssim(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    vals = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        val = ms_ssim(
            torch.tensor(ifft_pred).unsqueeze(1),
            torch.tensor(ifft_truth).unsqueeze(1),
            data_range=1,
            win_size=7,
            size_average=False,
        )
        vals.extend(val)

    click.echo("\nCreating ms-ssim histogram.\n")
    histogram_ms_ssim(vals, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean ms-ssim value is {np.mean(vals)}.\n")


def evaluate_mean_diff(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    vals = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        for pred, truth in zip(ifft_pred, ifft_truth):
            blobs_pred, blobs_truth = calc_blobs(pred, truth)
            flux_pred, flux_truth = crop_first_component(pred, truth, blobs_truth[0])
            vals.extend([(flux_pred.mean() - flux_truth.mean()) / flux_truth.mean()])

    click.echo("\nCreating mean_diff histogram.\n")
    vals = torch.tensor(vals) * 100
    histogram_mean_diff(vals, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean difference is {vals.mean()}.\n")

    if conf["save_vals"]:
        click.echo("\nSaving mean differences.\n")
        out = Path(conf["save_path"])
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "mean_diff.txt", vals)


def save_sampled(conf):
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    num_img = len(loader) * conf["batch_size"]
    model, norm_dict = load_pretrained_model(
        conf["arch_name"], conf["model_path"], img_size
    )
    if conf["model_path_2"] != "none":
        model_2, norm_dict = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    results = {}
    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        img_test, norm_dict = apply_normalization(img_test, norm_dict)
        pred = eval_model(img_test, model)
        pred = rescale_normalization(pred, norm_dict)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred_2 = rescale_normalization(pred, norm_dict)
            pred = torch.cat((pred, pred_2), dim=1)

        img = {"pred": pred, "inp": img_test, "true": img_true}
        # separate prediction and uncertainty
        unc_amp = pred[:, 1, :]
        unc_phase = pred[:, 3, :]
        unc = torch.stack([unc_amp, unc_phase], dim=1)
        pred_1 = pred[:, 0, :]
        pred_2 = pred[:, 2, :]
        pred = torch.stack((pred_1, pred_2), dim=1)
        img["unc"] = unc
        img["pred"] = pred

        result = sample_images(img["pred"], img["unc"], 1000, conf)

        # pad true image
        output = F.pad(input=img["true"], pad=(0, 0, 0, 63), mode="constant", value=0)
        img["true"] = sym_new(output, None)
        ifft_truth = get_ifft(img["true"], amp_phase=conf["amp_phase"])

        # add images to dict
        dict_img_true = {"true": ifft_truth}
        results = mergeDictionary(results, result)
        results = mergeDictionary(results, dict_img_true)

    # reshaping
    for key in results.keys():
        results[key] = results[key].reshape(num_img, img_size, img_size)

    name_model = Path(model_path).stem
    save_pred(str(out_path) + f"/sampled_imgs_{name_model}.h5", results)


def evaluate_ms_ssim_sampled(conf):
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    data_path = str(out_path) + "/sampled_imgs.h5"
    loader = create_sampled_databunch(data_path, conf["batch_size"])
    vals = []

    # iterate trough DataLoader
    for i, (samp, std, img_true) in enumerate(tqdm(loader)):
        val = ms_ssim(
            samp.unsqueeze(1),
            img_true.unsqueeze(1),
            data_range=1,
            win_size=7,
            size_average=False,
        )
        vals.extend(val)

    click.echo("\nCreating ms-ssim histogram.\n")
    vals = torch.tensor(vals)
    histogram_ms_ssim(vals, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean ms-ssim value is {vals.mean()}.\n")

    if conf["save_vals"]:
        click.echo("\nSaving area ratios.\n")
        out = Path(conf["save_path"])
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "area_ratios.txt", vals)


def evaluate_area_sampled(conf):
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    name_model = Path(model_path).stem
    data_path = str(out_path) + f"/sampled_imgs_{name_model}.h5"
    loader = create_sampled_databunch(data_path, conf["batch_size"])
    vals = []

    # iterate trough DataLoader
    for i, (samp, std, img_true) in enumerate(tqdm(loader)):
        for pred, truth in zip(samp, img_true):
            val = area_of_contour(pred, truth)
            vals.extend([val])

    click.echo("\nCreating eval_area histogram.\n")
    vals = torch.tensor(vals)
    histogram_area(vals, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean area ratio is {vals.mean()}.\n")

    if conf["save_vals"]:
        click.echo("\nSaving area ratios.\n")
        out = Path(conf["save_path"])
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "area_ratios.txt", vals)


def evaluate_intensity(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    ratios_sum = np.array([])
    ratios_peak = np.array([])

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        ratio_sum, ratio_peak = analyse_intensity(ifft_pred, ifft_truth)
        ratios_sum = np.append(ratios_sum, ratio_sum)
        ratios_peak = np.append(ratios_peak, ratio_peak)

    click.echo("\nCreating eval_intensity histogram.\n")
    histogram_sum_intensity(ratios_sum, out_path, plot_format=conf["format"])
    histogram_peak_intensity(ratios_peak, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean intensity ratio is {ratios_sum.mean()}.\n")


def evaluate_area(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    vals = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        for pred, truth in zip(ifft_pred, ifft_truth):
            val = area_of_contour(pred, truth)
            vals.extend([val])

    click.echo("\nCreating eval_area histogram.\n")
    vals = torch.tensor(vals)
    histogram_area(vals, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean area ratio is {vals.mean()}.\n")

    if conf["save_vals"]:
        click.echo("\nSaving area ratios.\n")
        out = Path(conf["save_path"])
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "area_ratios.txt", vals)


def evaluate_point(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)
    vals = []
    lengths = []

    for i, (img_test, img_true, source_list) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        fluxes_pred, fluxes_truth, length = flux_comparison(
            ifft_pred, ifft_truth, source_list
        )
        val = ((fluxes_pred - fluxes_truth) / fluxes_truth) * 100
        vals += list(val)
        lengths += list(length)

    vals = np.concatenate(vals).ravel()
    lengths = np.array(lengths, dtype="object")
    mask = lengths < 10

    click.echo("\nCreating pointsources histogram.\n")
    hist_point(vals, mask, out_path, plot_format=conf["format"])
    click.echo(f"\nThe mean flux difference is {vals.mean()}.\n")
    click.echo("\nCreating linear extent-mean flux diff plot.\n")
    plot_length_point(lengths, vals, mask, out_path, plot_format=conf["format"])


def evaluate_gan_sources(conf):
    model, model_2, loader, norm_dict, out_path = preprocessing(conf)

    img_size = loader.dataset[0][0][0].shape[-1]
    ratios = []
    num_zeros = []
    above_zeros = []
    below_zeros = []
    atols = [1e-4, 1e-3, 1e-2, 1e-1]

    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        ifft_pred, ifft_truth = process_prediction(
            conf, img_test, img_true, norm_dict, model, model_2
        )

        diff = ifft_pred - ifft_truth

        for atol in atols:
            zero = np.isclose(
                (np.zeros((ifft_truth.shape[0], img_size, img_size))), diff, atol=atol
            )
            num_zero = zero.sum(axis=-1).sum(axis=-1) / (img_size * img_size) * 100
            num_zeros += list(num_zero)

        ratio = np.abs(diff).max(axis=-1).max(axis=-1) / ifft_truth.max(axis=-1).max(
            axis=-1
        )

        below_zero = np.sum(diff < 0, axis=(1, 2)) / (img_size * img_size) * 100
        above_zero = np.sum(diff > 0, axis=(1, 2)) / (img_size * img_size) * 100

        ratios += list(ratio)
        above_zeros += list(above_zero)
        below_zeros += list(below_zero)

    num_images = (i + 1) * conf["batch_size"]
    ratios = np.array([ratios]).reshape(-1)
    num_zeros = np.array([num_zeros]).reshape(-1)
    above_zeros = np.array([above_zeros]).reshape(-1)
    below_zeros = np.array([below_zeros]).reshape(-1)
    click.echo("\nCreating GAN histograms.\n")
    histogram_gan_sources(
        ratios,
        num_zeros,
        above_zeros,
        below_zeros,
        num_images,
        out_path,
        plot_format=conf["format"],
    )
    click.echo(f"\nThe mean difference from maximum flux is {diff.mean()}.\n")
    click.echo(f"\nThe mean proportion of pixels close to 0 is {num_zeros.mean()}.\n")


def evaluate_yolo(conf):
    # create DataLoader
    loader = create_databunch(
        conf["data_path"],
        conf["fourier"],
        conf["source_list"],
        conf["batch_size"],
        conf["random"],
    )
    model_path = Path(conf["model_path"])
    out_path = model_path.parent / "evaluation" / model_path.stem
    out_path.mkdir(parents=True, exist_ok=True)

    plot_loss(
        model_path=model_path,
        out_path=out_path,
        metric_name="Overall IoU",
        save=True,
        plot_format="pdf",
    )

    model = load_pretrained_model(conf["arch_name"], model_path)[0]

    iou, eval_time = [], []
    plotted_images = 0
    for img_test, img_true in tqdm(loader):
        # img_true is soure list, not an image. Name is kept for consistency.
        start_time = time.time()
        pred = eval_model(img_test, model)
        eval_time.append(time.time() - start_time)
        iou.append(iou_YOLOv6(pred, img_true))
        if plotted_images < conf["num_images"]:
            for _ in range(len(img_test)):
                if plotted_images < conf["num_images"]:
                    plot_yolo_eval(
                        x=img_test,
                        y=img_true,
                        pred=pred,
                        out_path=out_path,
                        idx=plotted_images,
                        anchor_idx=0,
                    )
                    plotted_images += 1

    print(f"IoU of test dataset: {np.mean(np.array(iou)):.4}")
    print(f"Evaluation time per image: {np.mean(np.array(eval_time)):.4}")


def evaluate_mojave(conf):
    visualize = True
    reference = "image_intensity"  # image_intensity, comp_intensity, center

    loader = MojaveDataset(
        data_path=conf["data_path"],
        crop_size=128,
        # scaling=partial(scaling_log10_noisecut, thres_adj=0.5),
        # scaling=partial(scaling_log10_noisecut, thres_adj=1),
        scaling=partial(SymLogNorm, linthresh=1e-2, linthresh_rel=True, base=1.5),
    )
    # print(loader.source_list)
    print("Finished loading MOJAVE data")

    model_path = Path(conf["model_path"])
    out_path = model_path.parent / "evaluation" / model_path.stem
    out_path.mkdir(parents=True, exist_ok=True)

    model = load_pretrained_model(conf["arch_name"], model_path)[0]

    if conf["random"]:
        if conf["num_images"] > len(loader.source_list):
            conf["num_images"] = len(loader.source_list)
        selected_sources = np.random.choice(
            loader.source_list, conf["num_images"], replace=False
        )
    else:
        selected_sources = loader.source_list[5 : conf["num_images"] + 5]
    selected_sources = ["1142+198", "0149+710", "0035+413"]
    # selected_sources = ["1142+198"]
    # selected_sources = ["0149+710"]
    # selected_sources = ["0035+413"]
    print("Evaluated sources:", selected_sources)

    # Informations about beam properties
    bmaj = []  # clean beam major axis diameter (degrees)
    bmin = []  # clean beam minor axis diameter (degrees)
    bpa = []  # clean beam position angle (degrees)
    freq = []  # frequency value (Hz)

    stds = []
    v1 = []
    v2 = []
    v1_unc = []
    v2_unc = []
    eval_time = []
    for source in tqdm(selected_sources):
        # for source in selected_sources:
        img = torch.from_numpy(loader.open_source(source))
        # print(f"Evaluation of source {source} including {img.shape[0]} images.")

        st = time.time()
        pred = eval_model(img, model, amp_phase=conf["amp_phase"])
        outputs = yolo_apply_nms(pred, x=img)
        eval_time.append((time.time() - st) / img.shape[0])

        if visualize:
            for i in range(len(img)):
                date = loader.get_header(i, source)["DATE-OBS"]
                # continue
                if i < 3:
                    plot_yolo_mojave(
                        x=img[:, None],
                        pred=pred,
                        idx=i,
                        out_path=out_path,
                        name=source,
                        date=date,
                    )

        df = yolo_df(outputs=outputs, ds=loader, source=source)

        # Apply the cluster algorythm
        xy_mas = np.array([df["x_mas"], df["y_mas"]]).T
        # print("xy_mas shape:", xy_mas.shape)

        # st = time.time()

        # cluster_model = spectralClustering(xy_mas)
        # df["idx_comp"] = cluster_model.labels_

        cluster_model = gmmClustering(xy_mas, score_type="BIC")
        if cluster_model is None:
            print(f"No successful GMM cluster for source {source}!")
            df["idx_comp"] = np.zeros(len(xy_mas), dtype=int)
        else:
            df["idx_comp"] = cluster_model.predict(xy_mas)

        # print("Zeit für Clustering:", time.time() - st)

        # Change colums (just for the look)
        cols = df.columns.tolist()
        cols.insert(2, cols.pop(-1))
        df = df[cols]

        if visualize:
            plot_yolo_clustering(df=df, out_path=out_path, name=source)

        # Build mean of components, which are from the same image and the same cluster
        df = (
            df.groupby(["date", "idx_img", "idx_comp"], sort=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        df["intensity"] = img[
            df["idx_img"], df["y"].astype(int), df["x"].astype(int)
        ].numpy()

        if visualize:
            for i in range(len(img)):
                date = loader.get_header(i, source)["DATE-OBS"]
                # continue
                if i < 3:
                    plot_yolo_post_clustering(
                        x=img, df=df, idx=i, out_path=out_path, name=source, date=date
                    )

        if reference == "comp_intensity":
            # Find components with the maximum intensity -> main component
            idx_main = (
                df.groupby("idx_comp")["intensity"].mean(numeric_only=True).argmax()
            )

        # Calculate distance to main component
        df["distance"] = np.nan
        for i in range(len(img)):
            if reference == "image_intensity":
                x1, y1 = np.unravel_index(np.argmax(img[i], axis=None), img[i].shape)
                x1, y1 = pixel2coordinate(
                    loader.get_header(i, source), x1, y1, loader.crop_size, units=False
                )
            elif reference == "comp_intensity":
                main_component = df[(df["idx_img"] == i) & (df["idx_comp"] == idx_main)]
                x1 = main_component["x_mas"].to_numpy()
                y1 = main_component["y_mas"].to_numpy()
            elif reference == "center":
                x1, y1 = pixel2coordinate(
                    loader.get_header(i, source),
                    img[i].shape[-1] / 2,
                    img[i].shape[-1] / 2,
                    loader.crop_size,
                    units=False,
                )

            # Main component not detected in image, distance is not calculated
            # if not x1 or not y1:
            #     continue

            for index, row in df[df["idx_img"] == i].iterrows():
                x2 = row["x_mas"]
                y2 = row["y_mas"]
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                df.loc[index, "distance"] = dist
        # print(df)
        df = df.dropna()

        # Drop components with few appearances, because it often leads to unrealistic velocities
        for i in sorted(df["idx_comp"].unique()):
            if len(df[df["idx_comp"] == i]) < np.sqrt(len(img)):
                idx = df[df["idx_comp"] == i].index
                df.drop(idx, inplace=True)

        if df.empty:
            print(f"DataFrame is empty for source {source}!")
            continue

        # Perform fit (y=m*x+n) and calculate velocity
        df = yolo_linear_fit(df)

        if visualize:
            plot_yolo_velocity(df=df, out_path=out_path, name=source)

        # Compare velocities (fitted vs. MOJAVE (only max vel. stated))
        v_max = df["v"].max()
        v_argmax = df["v"].argmax()
        v_unc_max = df["v_unc"].values[v_argmax]

        # v_ncomps = (v == v_max).sum()
        v_ref = df["v_ref"].values[v_argmax]
        v_unc_ref = df["v_unc_ref"].values[v_argmax]
        # v_ncomps_ref = df["n_feat"].values[v_argmax]

        # print(rf"max velocity: {v_max:.2} \pm {v_unc_max:.2} by {v_ncomps} components")
        # print(
        #     rf"max velocity: {v_ref} \pm {v_unc_ref} by {int(v_ncomps_ref)} components"
        # )
        # print("v_max", v_max)

        if v_max != 0:
            v1.append(v_max)
            v2.append(v_ref)
            if v_unc_max != 0:
                std_env = (v_ref - v_max) / v_unc_max  # v1 + n*s1 = v2
                stds.append(std_env)
                v1_unc.append(v_unc_max / v_max)
                v2_unc.append(v_unc_ref / v_ref)

        for i in range(len(img)):
            hdr = loader.get_header(i, source)
            bmaj.append(hdr["BMAJ"])
            bmin.append(hdr["BMIN"])
            bpa.append(hdr["BPA"])
            freq.append(hdr["CRVAL3"])

    # if visualize:
    plot_beam_props(bmaj=bmaj, bmin=bmin, bpa=bpa, freq=freq, out_path=out_path)

    if stds:
        stds = np.array(stds)
        print(
            f"Lister velocity lie within {stds.mean():.2} \pm {stds.std():.2} standard deviations of the prediction\n"
        )

    eval_time = np.array(eval_time).mean()
    print(f"Average evalution time of YOLO & NMS: {eval_time:.8} s per image")

    v1 = np.array(v1)
    v2 = np.array(v2)
    # if visualize:
    if v1.size != 0 and v2.size != 0:
        plot_hist_velocity(v2 - v1, out_path=out_path)

    v1_unc = np.array(v1_unc)
    v2_unc = np.array(v2_unc)
    # if visualize:
    if v1_unc.size != 0 and v2_unc.size != 0:
        plot_hist_velocity_unc(v1_unc, v2_unc, out_path=out_path)


def evaluate_counterjet(conf):
    if "MOJAVE" in conf["data_path"]:
        loader = MojaveDataset(
            data_path=conf["data_path"],
            crop_size=256,
            # scaling=partial(scaling_log10_noisecut, thres_adj=0.5),
            # scaling=partial(scaling_log10_noisecut, thres_adj=1),
            scaling=partial(SymLogNorm, linthresh=1e-2, linthresh_rel=True, base=2),
        )
        print("Finished loading MOJAVE data")

        model_path = Path(conf["model_path"])
        out_path = model_path.parent / "evaluation" / model_path.stem
        out_path.mkdir(parents=True, exist_ok=True)

        model = load_pretrained_model(conf["arch_name"], model_path)[0]

        images = np.zeros((3, 1, 256, 256))  # images to plot
        images_value = np.array([1.0, 1.0, 0.0])  # prediction of images to plot
        acc = []  # accuracy on simulated data
        preds = []
        eval_time = []
        for i in tqdm(range(len(loader))):
            img = torch.from_numpy(loader.open_image(i))[None, None]
            # print(f"Evaluation of source {source} including {img.shape[0]} images.")

            start_time = time.time()
            pred = eval_model(img, model, amp_phase=conf["amp_phase"])
            eval_time.append(time.time() - start_time)

            pred = torch.sigmoid(pred)
            preds.append(pred)

            if pred < images_value[0]:
                images_value[0] = pred
                images[0, 0] = img
            if pred > images_value[2]:
                images_value[2] = pred
                images[2, 0] = img
            if np.abs(pred - 0.5) < np.abs(images_value[1] - 0.5):
                images_value[1] = pred
                images[1, 0] = img

        plot_counterjet_eval(
            x=images,
            pred=images_value,
            data_name="MOJAVE",
            out_path=out_path,
        )

        preds = torch.tensor(preds)

        threshold_cj = 0.9
        cj = (preds > threshold_cj).float().mean()
        ncj = (preds < 1 - threshold_cj).float().mean()
        unsure = 1 - cj - ncj
        print(f"No counterjet - Unsure - Counterjet: {ncj:.3} {unsure:.3} {cj:.3}")

        plot_hist_counterjet(
            x1=preds,
            threshold=threshold_cj,
            data_name="MOJAVE",
            out_path=out_path,
        )

    else:
        loader = create_databunch(
            conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
        )
        model_path = Path(conf["model_path"])
        out_path = model_path.parent / "evaluation" / model_path.stem
        out_path.mkdir(parents=True, exist_ok=True)

        plot_loss(
            model_path=model_path,
            out_path=out_path,
            metric_name="Accuracy",
            log_loss=True,
            save=True,
            plot_format="pdf",
        )

        model = load_pretrained_model(conf["arch_name"], model_path)[0]

        acc = []  # accuracy on simulated data
        preds, targets = [], []
        eval_time = []
        for img_test, img_true in tqdm(loader):
            # img_true is soure list, not an image. Name is kept for consistency.
            n_components = img_true.shape[1]
            amps = img_true[:, -int((n_components - 1) / 2) :, 0]
            amps_summed = torch.sum(amps, axis=1)
            target = (amps_summed > 0).float()
            targets.append(target)

            start_time = time.time()
            pred = eval_model(img_test, model)
            eval_time.append(time.time() - start_time)

            pred = torch.sigmoid(pred)
            preds.append(pred)
            acc.append(binary_accuracy(pred, img_true))

        plot_counterjet_eval(
            x=img_test,
            pred=pred,
            y=target,
            data_name="simulation",
            out_path=out_path,
        )

        preds = torch.cat(preds)
        targets = np.concatenate(targets)

        print(f"Accuracy of test dataset: {np.mean(np.array(acc)):.4}")

        threshold_cj = 0.9
        cj = (preds > threshold_cj).float().mean()
        ncj = (preds < 1 - threshold_cj).float().mean()
        unsure = 1 - cj - ncj
        print(f"No counterjet - Unsure - Counterjet: {ncj:.3} {unsure:.3} {cj:.3}")

        plot_hist_counterjet(
            x1=preds,
            x2=targets,
            threshold=threshold_cj,
            data_name="simulation",
            out_path=out_path,
        )

    print(f"Evaluation time per image: {np.mean(np.array(eval_time)):.4}")
