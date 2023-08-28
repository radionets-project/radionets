from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from tqdm import tqdm

from radionets.dl_framework.data import load_data
from radionets.evaluation.blob_detection import calc_blobs, crop_first_component
from radionets.evaluation.contour import analyse_intensity, area_of_contour
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
    histogram_unc,
    plot_contour,
    plot_length_point,
    plot_results,
    visualize_sampled_unc,
    visualize_source_reconstruction,
    visualize_uncertainty,
    visualize_with_fourier,
    visualize_with_fourier_diff,
)
from radionets.evaluation.pointsources import flux_comparison
from radionets.evaluation.utils import (
    apply_normalization,
    apply_symmetry,
    check_samp_file,
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
    sym_new,
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
        unc_amp = torch.sqrt(pred[:, 1, :])
        unc_phase = torch.sqrt(pred[:, 3, :])
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
    vals = np.array([])

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
        vals = np.append(vals, val)

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

    samp_file = check_samp_file(conf)
    if samp_file:
        if click.confirm("Existing sampling file found. Overwrite?", abort=True):
            click.echo("Overwriting sampling file!")

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
        unc_amp = pred[:, 1]
        unc_phase = pred[:, 3]
        unc_amp[unc_amp > 0] = np.sqrt(unc_amp[unc_amp > 0])
        unc_phase[unc_phase > 0] = np.sqrt(unc_phase[unc_phase > 0])

        if "std_real" in norm_dict:
            unc_amp = unc_amp * norm_dict["std_real"] + norm_dict["mean_real"]
            unc_phase = unc_phase * norm_dict["std_imag"] + norm_dict["mean_imag"]

        unc = torch.stack([unc_amp, unc_phase], dim=1)
        pred_1 = pred[:, 0, :]
        pred_2 = pred[:, 2, :]
        pred = torch.stack((pred_1, pred_2), dim=1)
        img["unc"] = unc
        img["pred"] = pred

        result = sample_images(img["pred"], img["unc"], 100, conf)

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


def evaluate_unc(conf):
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    name_model = Path(model_path).stem
    data_path = str(out_path) + f"/sampled_imgs_{name_model}.h5"
    loader = create_sampled_databunch(data_path, conf["batch_size"])
    vals = np.array([])

    # iterate trough DataLoader
    for i, (samp, std, img_true) in enumerate(tqdm(loader)):
        threshold = (img_true.max(-1)[0].max(-1)[0] * 0.01).reshape(
            img_true.shape[0], 1, 1
        )
        mask = img_true >= threshold

        # calculate on one image at a time
        for i in range(samp.shape[0]):
            mask_pos = samp[i][mask[i]] + std[i][mask[i]]
            mask_neg = samp[i][mask[i]] - std[i][mask[i]]

            cond = (img_true[i][mask[i]] <= mask_pos) & (
                img_true[i][mask[i]] >= mask_neg
            )

            val = np.where(cond, 1, 0).sum() / img_true[i][mask[i]].shape * 100
            vals = np.append(vals, val)

    histogram_unc(vals, out_path, plot_format=conf["format"])


def evaluate_intensity_sampled(conf):
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    name_model = Path(model_path).stem
    data_path = str(out_path) + f"/sampled_imgs_{name_model}.h5"
    loader = create_sampled_databunch(data_path, conf["batch_size"])
    ratios_sum = np.array([])
    ratios_peak = np.array([])

    # iterate trough DataLoader
    for i, (samp, std, img_true) in enumerate(tqdm(loader)):
        samp = samp.numpy()
        img_true = img_true.numpy()
        ratio_sum, ratio_peak = analyse_intensity(samp, img_true)
        ratios_sum = np.append(ratios_sum, ratio_sum)
        ratios_peak = np.append(ratios_peak, ratio_peak)

    click.echo("\nCreating eval_intensity histogram.\n")
    histogram_sum_intensity(ratios_sum, out_path, plot_format=conf["format"])
    histogram_peak_intensity(ratios_peak, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean intensity ratio is {ratios_sum.mean()}.\n")


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
