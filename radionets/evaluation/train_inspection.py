# import bdsf
import click
from functools import partial
import numpy as np
from pathlib import Path
from pytorch_msssim import ms_ssim
import re
import time
from tqdm import tqdm
import torch

from radionets.dl_framework.clustering import (
    spectralClustering,
    gmmClustering,
)
from radionets.dl_framework.data import (
    get_bundles,
    load_data,
    MojaveDataset,
)
from radionets.dl_framework.metrics import (
    iou_YOLOv6,
    binary_accuracy,
)
from radionets.evaluation.blob_detection import calc_blobs, crop_first_component
from radionets.evaluation.contour import area_of_contour
from radionets.evaluation.dynamic_range import calc_dr
from radionets.evaluation.jet_angle import calc_jet_angle
from radionets.evaluation.plotting import (
    histogram_area,
    histogram_dynamic_ranges,
    histogram_gan_sources,
    histogram_jet_angles,
    histogram_mean_diff,
    histogram_ms_ssim,
    hist_point,
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
    plot_yolo_post_clustering,
    plot_yolo_eval,
    plot_yolo_mojave,
    plot_yolo_velocity,
    visualize_source_reconstruction,
    visualize_with_fourier_diff,
    visualize_with_fourier,
)
from radionets.evaluation.pointsources import flux_comparison
from radionets.evaluation.utils import (
    calculate_velocity,
    create_databunch,
    eval_model,
    get_ifft,
    get_images,
    load_pretrained_model,
    pad_unsqueeze,
    reshape_2d,
    read_pred,
    save_pred,
    scaling_log10_noisecut,
    SymLogNorm,
    yolo_apply_nms,
    yolo_df,
    yolo_linear_fit,
)


def create_predictions(conf):
    if conf["model_path_2"] != "none":
        pred, img_test, img_true = get_separate_prediction(conf)
    else:
        pred, img_test, img_true = get_prediction(conf)
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_path) + "/predictions.h5"

    if not conf["fourier"]:
        click.echo("\n This is not a fourier dataset.\n")

    pred = pred.numpy()
    save_pred(out_path, pred, img_test, img_true, "pred", "img_test", "img_true")


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

    img_test, img_true = get_images(test_ds, num_images, rand=rand)

    img_size = img_test.shape[-1]
    model = load_pretrained_model(conf["arch_name"], conf["model_path"], img_size)

    pred = eval_model(img_test, model)

    # test for uncertainty
    if pred.shape[1] == 4:
        pred_1 = pred[:, 0, :].unsqueeze(1)
        pred_2 = pred[:, 2, :].unsqueeze(1)
        pred = torch.cat((pred_1, pred_2), dim=1)

    return pred, img_test, img_true


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

    pred, img_test, img_true = read_pred(path)
    if conf["fourier"]:
        if conf["diff"]:
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

    pred, img_test, img_true = read_pred(path)

    # inverse fourier transformation for prediction
    ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

    # inverse fourier transform for truth
    ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

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

    pred, img_test, img_true = read_pred(path)

    # inverse fourier transformation for prediction
    ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

    # inverse fourier transform for truth
    ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])

    for i, (pred, truth) in enumerate(zip(ifft_pred, ifft_truth)):
        plot_contour(pred, truth, out_path, i, plot_format=conf["format"])


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


def evaluate_yolo(conf):
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
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

    model = load_pretrained_model(conf["arch_name"], model_path)

    iou = []
    plotted_images = 0
    for img_test, img_true in tqdm(loader):
        # img_true is soure list, not an image. Name is kept for consistency.
        pred = eval_model(img_test, model)
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

    print(f"IoU of test dataset: {np.mean(np.array(iou)):.2}")


# def evaluate_pybdsf(conf):
#     bundle_paths = get_bundles(conf["data_path"])
#     data = np.sort(
#         [path for path in bundle_paths if re.findall("samp_test.*\.fits", path.name)]
#     )

#     if conf["random"]:
#         if conf["num_images"] > len(data):
#             conf["num_images"] = len(data)
#         data = np.random.choice(data, conf["num_images"], replace=False)
#     else:
#         data = data[: conf["num_images"]]

#     # iou = []
#     for filename in data:
#         print(filename)
#         img = bdsf.process_image(
#             filename,
#             beam=(3.04e-7, 1.57e-7, -5.57),  # values are average of MOJAVE data
#             frequency=1.53e10,
#             quiet=True,
#         )
#         print(img.model_gaus_arr)
#         # print(img.opts.to_list())
#         # iou.append(iou_YOLOv6(pred, img_true))
#         if conf["vis_pred"]:
#             img.show_fit()

#     # Remove logfiles to allow more than one evaluation run
#     logfiles = [p for p in Path(conf["data_path"]).rglob("*.log")]
#     [logfile.unlink() for logfile in logfiles]

#     # print(f"IoU of test dataset: {np.mean(np.array(iou)):.2}")


def evaluate_mojave(conf):
    visualize = True

    loader = MojaveDataset(
        data_path=conf["data_path"],
        crop_size=128,
        # scaling=partial(scaling_log10_noisecut, thres_adj=0.5),
        # scaling=partial(scaling_log10_noisecut, thres_adj=1),
        scaling=partial(SymLogNorm, linthresh=1e-2, linthresh_rel=True, base=2),
    )
    # print(loader.source_list)
    print("Finished loading MOJAVE data")

    model_path = Path(conf["model_path"])
    out_path = model_path.parent / "evaluation" / model_path.stem
    out_path.mkdir(parents=True, exist_ok=True)

    model = load_pretrained_model(conf["arch_name"], model_path)

    if conf["random"]:
        if conf["num_images"] > len(loader.source_list):
            conf["num_images"] = len(loader.source_list)
        selected_sources = np.random.choice(
            loader.source_list, conf["num_images"], replace=False
        )
    else:
        selected_sources = loader.source_list[5 : conf["num_images"] + 5]
    # selected_sources = ["1142+198"]
    # selected_sources = ["0149+710"]
    # selected_sources = ["0035+413"]
    # print("Evaluated sources:", selected_sources)

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

        # Find components with the maximum intensity -> main component
        idx_main = df.groupby("idx_comp")["intensity"].mean(numeric_only=True).argmax()

        # Calculate distance to main component
        df["distance"] = np.nan
        for i in range(len(img)):
            main_component = df[(df["idx_img"] == i) & (df["idx_comp"] == idx_main)]
            x1 = main_component["x_mas"].to_numpy()
            y1 = main_component["y_mas"].to_numpy()

            # Main component not detected in image, distance is not calculated
            if not x1 or not y1:
                continue

            for index, row in df[df["idx_img"] == i].iterrows():
                x2 = row["x_mas"]
                y2 = row["y_mas"]
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                df.loc[index, "distance"] = dist
        # print(df)
        df = df.dropna()

        # Drop components with one appearance, because it leads to no velocity
        for i in sorted(df["idx_comp"].unique()):
            if len(df[df["idx_comp"] == i]) == 1:
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
            crop_size=128,
            # scaling=partial(scaling_log10_noisecut, thres_adj=0.5),
            # scaling=partial(scaling_log10_noisecut, thres_adj=1),
            scaling=partial(SymLogNorm, linthresh=1e-2, linthresh_rel=True, base=2),
        )
        # print(loader.source_list)
        print("Finished loading MOJAVE data")

        model_path = Path(conf["model_path"])
        out_path = model_path.parent / "evaluation" / model_path.stem
        out_path.mkdir(parents=True, exist_ok=True)

        model = load_pretrained_model(conf["arch_name"], model_path)

        images = np.zeros((3, 128, 128))    # images to plot
        images_value = np.array([1, 1, 0])          # prediction of images to plot
        acc = []  # accuracy on simulated data
        preds = []
        for source in tqdm(loader.source_list):
            img = torch.from_numpy(loader.open_source(source))
            # print(f"Evaluation of source {source} including {img.shape[0]} images.")

            pred = torch.sigmoid(eval_model(img, model, amp_phase=conf["amp_phase"]))
            preds.append(pred)

            if pred.min() < images_value[0]:
                images_value[0] = pred.min()
                images[0] = img[pred.argmin()]
            if pred.max() > images_value[2]:
                images_value[2] = pred.max()
                images[2] = img[pred.argmax()]
            if np.abs(pred - 0.5).min() < np.abs(images_value[1] - 0.5).min():
                images_value[1] = pred[np.abs(pred - 0.5).argmin()]
                images[1] = img[np.abs(pred - 0.5).argmin()]

        plot_counterjet_eval(
            x=images,
            pred=images_value,
            data_name="MOJAVE",
            out_path=out_path,
        )

        preds = torch.cat(preds)

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
            save=True,
            plot_format="pdf",
        )

        model = load_pretrained_model(conf["arch_name"], model_path)

        acc = []  # accuracy on simulated data
        preds, targets = [], []
        for img_test, img_true in tqdm(loader):
            # img_true is soure list, not an image. Name is kept for consistency.
            n_components = img_true.shape[1]
            amps = img_true[:, -int((n_components - 1) / 2) :, 0]
            amps_summed = torch.sum(amps, axis=1)
            target = (amps_summed > 0).float()
            targets.append(target)

            pred = torch.sigmoid(eval_model(img_test, model))
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

        print(f"Accuracy of test dataset: {np.mean(np.array(acc)):.3}")

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
            out_path=out_path)
