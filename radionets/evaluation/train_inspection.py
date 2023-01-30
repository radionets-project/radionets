import click
import torch
import numpy as np
from pathlib import Path
from radionets.dl_framework.data import load_data
from radionets.evaluation.plotting import (
    visualize_with_fourier_diff,
    visualize_with_fourier,
    plot_results,
    visualize_source_reconstruction,
    histogram_jet_angles,
    histogram_dynamic_ranges,
    histogram_ms_ssim,
    histogram_mean_diff,
    histogram_area,
    histogram_gan_sources,
    plot_contour,
    hist_point,
    plot_length_point,
    visualize_uncertainty,
    visualize_sampled_unc,
)
from radionets.evaluation.utils import (
    create_databunch,
    create_sampled_databunch,
    reshape_2d,
    load_pretrained_model,
    get_images,
    eval_model,
    get_ifft,
    pad_unsqueeze,
    save_pred,
    read_pred,
    sample_images,
    mergeDictionary,
    sampled_dataset,
)
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.dynamic_range import calc_dr
from radionets.evaluation.blob_detection import calc_blobs, crop_first_component
from radionets.evaluation.contour import area_of_contour
from radionets.evaluation.pointsources import flux_comparison
from pytorch_msssim import ms_ssim
from tqdm import tqdm


def create_predictions(conf):
    if conf["model_path_2"] != "none":
        img = get_separate_prediction(conf)
    else:
        img = get_prediction(conf)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_path) + "/predictions.h5"

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
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)

    pred = eval_model(img_test, model)
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
    model_1 = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    model_2 = load_pretrained_model(conf["arch_name_2"], conf["model_path_2"], img_size)

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
    path += "/predictions.h5"
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
    path += "/predictions.h5"
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
            blobs=conf["vis_blobs"],
            msssim=conf["vis_ms_ssim"],
            plot_format=conf["format"],
        )

    return np.abs(ifft_pred), np.abs(ifft_truth)


def create_contour_plots(conf, num_images=3, rand=False):
    model_path = conf["model_path"]
    path = str(Path(model_path).parent / "evaluation")
    path += "/predictions.h5"
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
    predictions_path = path + "/predictions.h5"
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

    sampling_path = path + "/sampled_imgs.h5"
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
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    alpha_truths = []
    alpha_preds = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):
        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

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
    # create Dataloader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    dr_truths = np.array([])
    dr_preds = np.array([])

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

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

    click.echo("\nCreating histogram of dynamic ranges.\n")
    histogram_dynamic_ranges(dr_truths, dr_preds, out_path, plot_format=conf["format"])


def evaluate_ms_ssim(conf):
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    vals = []

    if img_size < 160:
        click.echo(
            "\nThis is only a placeholder!\
                Images too small for meaningful ms ssim calculations.\n"
        )

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

        if img_size < 160:
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
    histogram_ms_ssim(vals, out_path, plot_format=conf["format"])

    click.echo(f"\nThe mean ms-ssim value is {vals.mean()}.\n")


def evaluate_mean_diff(conf):
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    vals = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

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
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    results = {}
    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        img = {"pred": pred, "inp": img_test, "true": img_true}
        if pred.shape[1] == 4:
            unc_amp = pred[:, 1, :]
            unc_phase = pred[:, 3, :]
            unc = torch.stack([unc_amp, unc_phase], dim=1)
            pred_1 = pred[:, 0, :]
            pred_2 = pred[:, 2, :]
            pred = torch.stack((pred_1, pred_2), dim=1)
            img["unc"] = unc
            img["pred"] = pred

        result = sample_images(img["pred"], img["unc"], 100)
        ifft_truth = get_ifft(img["true"], amp_phase=conf["amp_phase"])
        dict_img_true = {"true": ifft_truth}
        # print(dict_img_true["true"].shape)
        results = mergeDictionary(results, result)
        results = mergeDictionary(results, dict_img_true)
    for key in results.keys():
        results[key] = results[key].reshape(num_img, img_size, img_size)
    save_pred(str(out_path) + "/sampled_imgs.h5", results)


def evaluate_area_sampled(conf):
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    data_path = str(out_path) + "/sampled_imgs.h5"
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


def evaluate_area(conf):
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    vals = []

    # iterate trough DataLoader
    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

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
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    vals = []
    lengths = []

    for i, (img_test, img_true, source_list) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

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
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)
    if conf["model_path_2"] != "none":
        model_2 = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    ratios = []
    num_zeros = []
    above_zeros = []
    below_zeros = []
    atols = [1e-4, 1e-3, 1e-2, 1e-1]

    for i, (img_test, img_true) in enumerate(tqdm(loader)):

        pred = eval_model(img_test, model)
        if conf["model_path_2"] != "none":
            pred_2 = eval_model(img_test, model_2)
            pred = torch.cat((pred, pred_2), dim=1)

        ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
        ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

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
