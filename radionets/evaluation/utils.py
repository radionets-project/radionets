import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Distance
from astropy.modeling import fitting, models
from astropy.utils.exceptions import AstropyWarning
from numba import set_num_threads, vectorize
from PIL import Image
from torch.utils.data import DataLoader

import radionets.dl_framework.architecture as architecture
from radionets.dl_framework.data import load_data
from radionets.dl_framework.model import load_pre_model
from radionets.dl_framework.utils import decode_yolo_box, xywh2xyxy
from radionets.evaluation.coordinates import pixel2coordinate


def source_list_collate(batch):
    """Collate function for the DataLoader with source list

    Parameters
    ----------
    batch : tuple
        input and target images alongside with the corresponding source_list

    Returns
    -------
    tuple
        stacked images and list for source_list values
    """
    length = len(batch[0])

    if length == 1:
        x = [item[0] for item in batch]
        return torch.stack(x)

    elif length == 2:
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return torch.stack(x), torch.stack(y)

    elif length == 3:
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        z = [item[2][0] for item in batch]
        return torch.stack(x), torch.stack(y), z

    else:
        print("\nData shape not implemented\n")


def create_databunch(data_path, fourier, source_list, batch_size, shuffel=False):
    """Create a dataloader object, which feeds the data batch-wise

    Parameters
    ----------
    data_path : str
        path to the data
    fourier : bool
        true, if data in Fourier space is used
    source_list : bool
        true, if source_list data is used
    batch_size : int
        number of images for one batch

    Returns
    -------
    DataLoader
        dataloader object
    """
    # Load data sets
    test_ds = load_data(
        data_path, mode="test", fourier=fourier, source_list=source_list
    )

    # Create databunch with defined batchsize and check for source_list
    if source_list:
        data = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=shuffel,
            collate_fn=source_list_collate,
        )
    else:
        data = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffel)
    return data


def create_sampled_databunch(data_path, batch_size):
    """Create a dataloader object, which feeds the data batch-wise

    Parameters
    ----------
    data_path : str
        path to the data
    fourier : bool
        true, if data in Fourier space is used
    source_list : bool
        true, if source_list data is used
    batch_size : int
        number of images for one batch

    Returns
    -------
    DataLoader
        dataloader object
    """
    # Load data sets
    test_ds = sampled_dataset(data_path)

    data = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return data


def read_config(config):
    """Parse the toml config file

    Parameters
    ----------
    config : dict
        dict which contains the configurations loaded with toml.load

    Returns
    -------
    dict
        dict containing all configurations with unique keywords
    """
    eval_conf = {}
    eval_conf["data_path"] = config["paths"]["data_path"]
    eval_conf["model_path"] = config["paths"]["model_path"]
    eval_conf["model_path_2"] = config["paths"]["model_path_2"]

    eval_conf["quiet"] = config["mode"]["quiet"]

    eval_conf["format"] = config["general"]["output_format"]
    eval_conf["fourier"] = config["general"]["fourier"]
    eval_conf["amp_phase"] = config["general"]["amp_phase"]
    eval_conf["arch_name"] = config["general"]["arch_name"]
    eval_conf["source_list"] = config["general"]["source_list"]
    eval_conf["arch_name_2"] = config["general"]["arch_name_2"]
    eval_conf["diff"] = config["general"]["diff"]

    eval_conf["vis_pred"] = config["inspection"]["visualize_prediction"]
    eval_conf["vis_source"] = config["inspection"]["visualize_source_reconstruction"]
    eval_conf["sample_unc"] = config["inspection"]["sample_uncertainty"]
    eval_conf["unc"] = config["inspection"]["visualize_uncertainty"]
    eval_conf["plot_contour"] = config["inspection"]["visualize_contour"]
    eval_conf["vis_dr"] = config["inspection"]["visualize_dynamic_range"]
    eval_conf["vis_ms_ssim"] = config["inspection"]["visualize_ms_ssim"]
    eval_conf["num_images"] = config["inspection"]["num_images"]
    eval_conf["random"] = config["inspection"]["random"]

    eval_conf["viewing_angle"] = config["eval"]["evaluate_viewing_angle"]
    eval_conf["dynamic_range"] = config["eval"]["evaluate_dynamic_range"]
    eval_conf["ms_ssim"] = config["eval"]["evaluate_ms_ssim"]
    eval_conf["intensity"] = config["eval"]["evaluate_intensity"]
    eval_conf["mean_diff"] = config["eval"]["evaluate_mean_diff"]
    eval_conf["area"] = config["eval"]["evaluate_area"]
    eval_conf["batch_size"] = config["eval"]["batch_size"]
    eval_conf["point"] = config["eval"]["evaluate_point"]
    eval_conf["predict_grad"] = config["eval"]["predict_grad"]
    eval_conf["gan"] = config["eval"]["evaluate_gan"]
    eval_conf["yolo"] = config["eval"]["evaluate_yolo"]
    eval_conf["mojave"] = config["eval"]["evaluate_mojave"]
    eval_conf["counterjet"] = config["eval"]["evaluate_counterjet"]
    eval_conf["save_vals"] = config["eval"]["save_vals"]
    eval_conf["save_path"] = config["eval"]["save_path"]
    return eval_conf


def reshape_2d(array):
    """
    Reshape 1d arrays into 2d ones.

    Parameters
    ----------
    array: 1d array
        input array

    Returns
    -------
    array: 2d array
        reshaped array
    """
    shape = [int(np.sqrt(array.shape[-1]))] * 2
    return array.reshape(-1, *shape)


def make_axes_nice(
    fig, ax, im, title=False, phase=False, phase_diff=False, unc=False, objectness=False
):
    """Create nice colorbars with bigger label size for every axis in a subplot.
    Also use ticks for the phase.
    Parameters
    ----------
    fig : figure object
        current figure
    ax : axis object
        current axis
    im : ndarray
        plotted image
    title : str
        title of subplot
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if title:
        ax.set_title(title)
    if phase:
        cbar = fig.colorbar(
            im,
            cax=cax,
            orientation="vertical",
            ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        )
        cbar.set_label("Phase / rad")
    elif phase_diff:
        cbar = fig.colorbar(
            im,
            cax=cax,
            orientation="vertical",
            ticks=[-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi],
        )
        cbar.set_label("Phase / rad")
    elif unc:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label(r"$\sigma^2$ / a.u.")
    elif objectness:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("Objectness")
    else:
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("Specific Intensity / a.u.")

    if phase:
        # set ticks for colorbar
        cbar.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    elif phase_diff:
        # set ticks for colorbar
        cbar.ax.set_yticklabels([r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"])


def check_vmin_vmax(inp):
    """
    Check wether the absolute of the maxmimum or the minimum is bigger.
    If the minimum is bigger, return value with minus. Otherwise return
    maximum.
    Parameters
    ----------
    inp : float
        input image
    Returns
    -------
    float
        negative minimal or maximal value
    """
    if np.abs(inp.min()) > np.abs(inp.max()):
        a = -inp.min()
    else:
        a = inp.max()
    return a


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
        "filter_deep" in arch_name
        or "resnet" in arch_name
        or "Uncertainty" in arch_name
    ):
        arch = getattr(architecture, arch_name)(img_size)
    else:
        arch = getattr(architecture, arch_name)()
    norm_dict = load_pre_model(arch, model_path, visualize=True)
    return arch, norm_dict


def get_images(test_ds, num_images, rand=False, indices=None):
    """
    Get n random test and truth images or mean, standard deviation and
    true images from an already sampled dataset.

    Parameters
    ----------
    test_ds: h5_dataset
        data set with test images
    num_images: int
        number of test images
    rand: bool
        true if images should be drawn random
    indices: list
        list of indices to be used

    Returns
    -------
    img_test: n 2d arrays
        test images
    img_true: n 2d arrays
        truth images
    """
    if hasattr(test_ds, "amp_phase"):
        indices = torch.arange(num_images)
        if rand:
            indices = torch.randint(0, len(test_ds), size=(num_images,))

            # remove dublicate indices
            while len(torch.unique(indices)) < len(indices):
                new_indices = torch.randint(
                    0, len(test_ds), size=(num_images - len(torch.unique(indices)),)
                )
                indices = torch.cat((torch.unique(indices), new_indices))

            # sort after getting indices
            indices, _ = torch.sort(indices)

        img_test = test_ds[indices][0]
        img_true = test_ds[indices][1]
        img_test = img_test[:, :, :65, :]
        img_true = img_true[:, :, :65, :]
        return img_test, img_true, indices
    else:
        mean = test_ds[indices][0]
        std = test_ds[indices][1]
        img_true = test_ds[indices][2]
        return mean, std, img_true


def eval_model(img, model, amp_phase: bool = True):
    """
    Put model into eval mode and evaluate test images.

    Parameters
    ----------
    img: str
        test image
    model: architecture object
        architecture with pretrained weigths
    amp_phase: bool
        amp_phase keyword in config

    Returns
    -------
    pred: n 1d arrays
        predicted images
    """
    if len(img.shape) == (3):
        if amp_phase:
            img = img.unsqueeze(0)
        else:
            img = img.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            pred = model.cuda()(img.float().cuda())
        else:
            pred = model(img.float())

    if isinstance(pred, list):
        for p in pred:
            p.cpu()
        return pred
    else:
        return pred.cpu()


def get_ifft(array, amp_phase=False, scale=False):
    """Compute the inverse Fourier transformation

    Parameters
    ----------
    array : ndarray
        array with shape (2, img_size, img_size) with optional batch size
    amp_phase : bool, optional
        true, if splitting in amplitude and phase was used, by default True

    Returns
    -------
    ndarray
        image(s) in image space
    """
    if len(array.shape) == 3:
        array = array.unsqueeze(0)
    if amp_phase:
        if scale:
            amp = 10 ** (10 * array[:, 0] - 10) - 1e-10
        else:
            amp = array[:, 0]

        a = amp * np.cos(array[:, 1])
        b = amp * np.sin(array[:, 1])
        compl = a + b * 1j
    else:
        compl = array[:, 0] + array[:, 1] * 1j
    if compl.shape[0] == 1:
        compl = compl.squeeze(0)
    return np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(compl))))


def fft_pred(pred, truth, amp_phase=True):
    """
    Transform predicted image and true image to local domain.

    Parameters
    ----------
    pred: 4D array [1, channel, height, width]
        prediction from eval_model
    truth: 3D array [channel, height, width]
        true image
    amp_phase: Bool
        trained on Amp/Phase or Re/Im

    Returns
    -------
    ifft_pred, ifft_true: two 2D arrays [height, width]
        predicted and true image in local domain
    """
    a = pred[:, 0, :, :]
    b = pred[:, 1, :, :]

    a_true = truth[0, :, :]
    b_true = truth[1, :, :]

    if amp_phase:
        amp_pred_rescaled = (10 ** (10 * a) - 1) / 10**10
        phase_pred = b

        amp_true_rescaled = (10 ** (10 * a_true) - 1) / 10**10
        phase_true = b_true

        compl_pred = amp_pred_rescaled * np.exp(1j * phase_pred)
        compl_true = amp_true_rescaled * np.exp(1j * phase_true)
    else:
        compl_pred = a + 1j * b
        compl_true = a_true + 1j * b_true

    ifft_pred = np.fft.ifft2(compl_pred)
    ifft_true = np.fft.ifft2(compl_true)

    return np.absolute(ifft_pred)[0], np.absolute(ifft_true)


def non_max_suppression(
    boxes,
    obj_thres: float = 0.25,
    max_nms: int = 1000,
    iou_thres: float = 0.3,
    max_det: int = 30,
):
    """Non-maximum suppression (nms) for one batch.

    Parameters
    ----------
    boxes: ndarray
        boxes of shape (bs, n_boxes, 6), where 6 is: x, y, width, height, objectness, rotation
    obj_thres: float
        only take boxes with objectness above this value into account for nms, range: [0, 1]
    max_nms: int
        maximum number of boxes put into torchvision.ops.nms()
    iou_thres: float
        nms parameter: discards all overlapping boxes with IoU > iou_thres
    max_det: int
        number of boxes to return

    Returns
    -------
    output: list
        reduced number of boxes; list, because each image in batch can have
        different number of boxes
    """
    output = []

    for box in boxes:
        boxes_candidates = box[..., 4] > obj_thres  # candidates
        box = box[boxes_candidates]  # filter candidates

        if not box.shape[0]:  # if no box remains, skip the next image
            output.append(torch.tensor([]).to(boxes.device))
            continue
        elif box.shape[0] > max_nms:  # if boxes exceeds maximum for nms
            box = box[box[:, 4].argsort(descending=True)[:max_nms]]

        keep_box_idx = torchvision.ops.nms(xywh2xyxy(box[:, :4]), box[:, 4], iou_thres)
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output.append(box[keep_box_idx])

    return output


def yolo_apply_nms(
    pred, x=None, strides=None, rel_obj_thres: float = 0.5, iou_thres: float = 0.25
):
    """Perform basic evaluation steps after YOLO model.
    From YOLO output (pred) to list of boxes after nms.
    x or strides have to be provided.

    Parameters
    ----------
    pred: list
        list of feature maps, each of shape (bs, a, my, mx, 6)
    x: 4d-array
        input image (bs, 1, ny, nx)
    strides: 1d-array or list
        strides used during training
    rel_obj_thres: float
        relative objectness threshold for nms.
    iou_thres: float
        nms parameter: discards all overlapping boxes with IoU > iou_thres

    Returns
    -------
    outputs: list
        boxes after nms
    """
    assert not (x is None and strides is None), 'Provide "x" or "strides"'

    if x is not None and strides is not None:
        strides_calc = get_strides(x, pred)
        assert (
            strides == strides_calc
        ), f"Provided strides {strides} not equal calculated strides {strides_calc}."
    elif x is not None:
        strides = get_strides(x, pred)

    if not torch.is_tensor(strides):
        strides = torch.tensor(strides)

    obj_map = objectness_mapping(pred)
    obj_max = np.max(obj_map)

    outputs = []
    for i_anchor in range(pred[0].shape[1]):
        # use boxes from prediction of highest resolution
        boxes = decode_yolo_box(pred[0][:, [i_anchor]], strides[0])
        boxes[..., 4] = torch.tensor(obj_map[:, [i_anchor]]).to(boxes.device)
        boxes_nms = non_max_suppression(
            boxes.reshape(boxes.shape[0], -1, 6),
            obj_thres=rel_obj_thres * obj_max,
            iou_thres=iou_thres,
        )
        if i_anchor == 0:
            outputs = boxes_nms
        else:
            for i_img in range(len(boxes_nms)):
                outputs[i_img] = torch.cat((outputs[i_img], boxes_nms[i_img]))

    return outputs


def objectness_mapping(
    pred, calc: str = "mul", reduction: str = "mean", scaling: str = "sigmoid"
):
    """Mapping objectnesses of all feature maps together

    Parameters
    ----------
    pred: list
        list of feature maps, each of shape (bs, a, ny, nx, 6)
    calc: string
        operator between feature maps;
        "mul" for multiplication, "sum" for summation
    reduction: string
        specifies the reduction to apply to the output when calc is "sum";
        "mean" for average
    scaling: string
        apply a scaling to objectness; "sigmoid"

    Returns
    -------
    out: ndarray
        merge objectness predictions with size of the largest feature map
    """
    bs, a, ny, nx, _ = pred[0].shape

    if calc == "mul":
        out = np.ones((bs, a, ny, nx))
    elif calc == "sum":
        out = np.zeros((bs, a, ny, nx))
    else:
        print("Invalid value for argument 'calc'. Use 'mul' or 'sum'.")
        quit()

    for feature_map in pred:
        for i in range(bs):
            for j in range(a):
                if scaling == "sigmoid":
                    image = (
                        torch.sigmoid(feature_map[i, j, :, :, 4]).detach().cpu().numpy()
                    )
                else:
                    image = feature_map[i, j, :, :, 4]

                image = Image.fromarray(np.uint8(image * 255))
                image = image.resize([ny, nx], Image.Resampling.BOX)
                image = np.array(image) / 255

                if calc == "mul":
                    out[i, j] *= image
                elif calc == "sum":
                    out[i, j] += image

    if calc == "sum" and reduction == "mean":
        out /= len(pred)

    return out


def get_strides(x, pred):
    """Calculate strides of YOLO model from input x and prediction pred.

    Parameters
    ----------
    x: 4d-array
        input image (bs, 1, ny, nx)
    pred: list
        list of feature maps, each of shape (bs, a, my, mx, 6)

    Returns
    -------
    strides: ndarray
        array with strides of shape (n feature maps,)
    """
    strides = np.empty(len(pred))
    for i, feature_map in enumerate(pred):
        strides[i] = x.shape[-1] / feature_map.shape[-2]

    return strides


def scaling_log10_noisecut(img, thres_adj: float = 1):
    """Perform log10 scaling with a noise cut.

    Parameters
    ----------
    img: ndarray
        input image
    thres_adj: float
        threshold adjustment for mask

    Returns
    -------
    img: ndarray
        manipulated input image
    """
    img = img.copy()

    # noise is fluctuation around zero: take max negative value as threshold for noise removing
    img[img < np.abs(img.min()) * thres_adj] = -1  # -1 so value gets NaN in log

    # make jet more visible
    with warnings.catch_warnings():  # on purpose: -1 will lead to invalid values
        warnings.filterwarnings("ignore", message="invalid value encountered in log10")
        img = np.log10(img)

    # rescale between 0 and 1
    img -= np.nanmin(img)  # pytorch has no nanmin or similar
    img /= np.nanmax(img)  # pytorch has no nanmax or similar

    img[np.isnan(img)] = 0
    return img


def SymLogNorm(
    x,
    linthresh: float = 2,
    linthresh_rel: bool = False,
    linscale: float = 1,
    base: float = 10,
    minmax: bool = True,
):
    """Symmetric logarithmic normalisation based on matplotlib.colors.SymLogNorm
    Additionaly with the option to use a MinMaxScaler.

    Parameters
    ----------
    x: ndarray
        input data
    linthresh: float
        threshold at which a linear scale is used, >0
    linthresh_rel: bool
        set linthresh as absolute value or relative to input image
    linscale: float
        factor for linear scale, >0
    base: float
        base for logarithm
    minmax: bool
        if true, MinMax-Scaling is applied to output
    """
    if linthresh <= 0.0:
        raise ValueError("'linthresh' must be positive")
    if linscale <= 0.0:
        raise ValueError("'linscale' must be positive")
    if base <= 1.0:
        raise ValueError("'base' must be larger than 1")

    if linthresh_rel:
        linthresh *= np.max(x)

    linscale_adj = linscale / (1.0 - base**-1)
    abs_x = np.abs(x)
    out = (
        np.sign(x)
        * linthresh
        * (linscale_adj + np.log(abs_x / linthresh) / np.log(base))
    )
    inside = abs_x <= linthresh
    out[inside] = x[inside] * linscale_adj

    if minmax:
        out -= out.min()
        out /= out.max()

    return out


def calculate_velocity(theta: float, z: float):
    """Calculate velocity of jet components.
    Parallax: l = d * tan(theta)
        Using small angle approximation: l = d * theta

    Parameters
    ----------
    theta: float
        angle in mas
    z: float
        redshift of source

    Returns
    -------
    vc: float
        velocity in c (speed of light)
    """
    d = Distance(z=z)
    theta = (theta * u.mas).to(u.rad).value
    length = d * theta

    v = length / u.s
    vc = v.to(u.km / u.s) / const.c.to("km/s")

    return vc


def yolo_df(outputs: list, ds, source: str):
    """Creates a pandas dataframe to sort outputs for further calculations.

    Parameters
    ----------
    outputs: list
        list of outputs from nms
    ds:
        MojaveDataset class
    source: str
        name of the source

    Returns
    -------
    df: pandas.DataFrame
        dataframe with reconstructed properties
    """
    pred_list = []
    for i, output in enumerate(outputs):
        output = output.cpu().detach().numpy()
        header = ds.get_header(i, source)
        # redshift = get_redshift(source)  # ds.source not defined in "evaluate_mojave"
        for out in output:
            pred_dict = {}
            pred_dict["date"] = pd.Timestamp(header["DATE-OBS"])
            pred_dict["idx_img"] = i
            pred_dict["redshift"] = header["REDSHIFT"]
            for j in range(100):
                try:
                    # print(j, header[f"V{j+1}"], header[f"V_UNC{j+1}"])
                    pred_dict[f"v_ref{j+1}"] = header[f"V{j+1}"]
                    pred_dict[f"v_unc_ref{j+1}"] = header[f"V_UNC{j+1}"]
                except KeyError:
                    break
            pred_dict["x"] = out[0]
            pred_dict["y"] = out[1]
            pred_dict["sx"] = out[2]
            pred_dict["sy"] = out[3]
            pred_dict["objectness"] = out[4]
            pred_dict["rotation"] = out[5] * 180
            pred_dict["x_mas"], pred_dict["y_mas"] = pixel2coordinate(
                header, pred_dict["x"], pred_dict["y"], ds.crop_size, units=False
            )

            pred_list.append(pred_dict)

    df = pd.DataFrame(pred_list)

    return df


def yolo_linear_fit(df):
    """Linear fit with astropy in MOJAVE evaluation.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe with reconstructed properties

    Returns
    -------
    df: pandas.DataFrame
        dataframe including fit parameters and errors
    """
    # print(df)
    for i in df["idx_comp"].unique():
        # in seconds, pandas returns in ns
        x = df[df["idx_comp"] == i]["date"].astype(int) / 1e9
        # in mas
        y = df[df["idx_comp"] == i]["distance"]

        # scipy.optimize.curve_fit gives same results, but is slower (03.2023)
        fit = fitting.LevMarLSQFitter()
        line_init = models.Linear1D()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            # print(len(x))
            fitted_line = fit(line_init, x, y)

        parameters = fitted_line.parameters
        param_cov = fit.fit_info["param_cov"]
        if param_cov is not None:
            errors = np.sqrt(np.diag(param_cov))
        elif len(x) == 2 and len(y) == 2:
            errors = np.array([0, 0])
        else:
            print("x", x)
            print("y", y)

        v = calculate_velocity(parameters[0], df["redshift"].values[0])
        v_unc = calculate_velocity(errors[0], df["redshift"].values[0])

        df.loc[df["idx_comp"] == i, "v"] = v
        df.loc[df["idx_comp"] == i, "v_unc"] = v_unc

        # Needed for plotting the fit
        df.loc[df["idx_comp"] == i, "fit_param_m"] = parameters[0]
        df.loc[df["idx_comp"] == i, "fit_param_b"] = parameters[1]

    return df


def save_pred(path, img):
    """
    write test data and predictions to h5 file
    x: predictions of truth of test data
    y: input image of the test data
    z: truth of the test data
    """
    with h5py.File(path, "w") as hf:
        for key, value in img.items():
            hf.create_dataset(key, data=value)
        hf.close()


def read_pred(path):
    """
    read data saved with save_pred from h5 file
    x: predictions of truth of test data
    y: input image of the test data
    z: truth of the test data
    """
    images = {}
    with h5py.File(path, "r") as hf:
        for key in hf.keys():
            images[key] = np.array(hf[key])
        hf.close()
    return images


def check_outpath(model_path):
    """Checks if there is already a predictions file in the evaluation folder

    Parameters
    ----------
    model_path : str
        path to the model

    Returns
    -------
    bool
        true, if the file exists
    """
    name_model = Path(model_path).stem
    model_path = Path(model_path).parent / "evaluation" / f"predictions_{name_model}.h5"
    path = Path(model_path)
    exists = path.exists()
    return exists


def sym_new(image, key):
    """
    Symmetry function to complete the images

    Parameters
    ----------
    image : torch.Tensor
        (stack of) half images

    Returns
    -------
    torch.Tensor
        quadratic images after utilizing symmetry
    """
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    if len(image.shape) == 3:
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    upper_half = image[:, :, :64, :].clone()
    a = torch.rot90(upper_half, 2, dims=[-2, -1])

    image[:, 0, 65:, 1:] = a[:, 0, :-1, :-1]
    image[:, 0, 65:, 0] = a[:, 0, :-1, -1]

    if key == "unc":
        image[:, 1, 65:, 1:] = a[:, 1, :-1, :-1]
        image[:, 1, 65:, 0] = a[:, 1, :-1, -1]
    else:
        image[:, 1, 65:, 1:] = -a[:, 1, :-1, :-1]
        image[:, 1, 65:, 0] = -a[:, 1, :-1, -1]

    return image


def apply_symmetry(img_dict):
    """
    Pads and applies symmetry to half images. Takes a dict as input

    Parameters
    ----------
    img_dict : dict
        input dict which contains the half images

    Returns
    -------
    dict
        input dict with quadratic images
    """
    for key in img_dict:
        if key != "indices":
            if isinstance(img_dict[key], np.ndarray):
                img_dict[key] = torch.tensor(img_dict[key])
            output = F.pad(
                input=img_dict[key], pad=(0, 0, 0, 63), mode="constant", value=0
            )
            output = sym_new(output, key)
            img_dict[key] = output

    return img_dict


def even_better_symmetry(x):
    upper_half = x[:, :, 0 : x.shape[2] // 2, :].copy()
    upper_left = upper_half[:, :, :, 0 : upper_half.shape[3] // 2].copy()
    upper_right = upper_half[:, :, :, upper_half.shape[3] // 2 :].copy()
    a = np.flip(upper_left, axis=2)
    b = np.flip(upper_right, axis=2)
    a = np.flip(a, axis=3)
    b = np.flip(b, axis=3)

    upper_half[:, :, :, 0 : upper_half.shape[3] // 2] = b
    upper_half[:, :, :, upper_half.shape[3] // 2 :] = a

    x[:, 0, x.shape[2] // 2 :, :] = upper_half[:, 0]
    x[:, 1, x.shape[2] // 2 :, :] = -upper_half[:, 1]
    return x


@vectorize(["float64(float64, float64, float64, float64)"], target="cpu")
def tn_numba_vec_cpu(mu, sig, a, b):
    rv = np.random.normal(loc=mu, scale=sig)
    cond = rv > a and rv < b
    while not cond:
        rv = np.random.normal(loc=mu, scale=sig)
        cond = rv > a and rv < b

    return rv


@vectorize(["float64(float64, float64, float64, float64)"], target="parallel")
def tn_numba_vec_parallel(mu, sig, a, b):
    rv = np.random.normal(loc=mu, scale=sig)
    cond = rv > a and rv < b
    while not cond:
        rv = np.random.normal(loc=mu, scale=sig)
        cond = rv > a and rv < b

    return rv


def trunc_rvs(mu, sig, num_samples, mode, target="cpu", nthreads=1):
    if mode == "amp":
        a = 0
        b = np.inf
    elif mode == "phase":
        a = -np.pi
        b = np.pi
    elif mode == "real" or mode == "imag":
        a = -np.inf
        b = np.inf
    else:
        raise ValueError("Unsupported mode, use either ``phase`` or ``amp``.")
    mu = np.tile(mu, (num_samples, 1, 1, 1))
    sig = np.tile(sig, (num_samples, 1, 1, 1))

    if target == "cpu":
        if nthreads > 1:
            raise ValueError(
                f"Target is ``cpu`` but nthreads is {nthreads}, "
                "use target=``parallel`` instead."
            )
        res = tn_numba_vec_cpu(mu, sig, a, b)
    elif target == "parallel":
        if nthreads == 1:
            raise ValueError(
                "Target is ``parallel`` but nthreaads is 1, use target=``cpu`` instead."
            )
        set_num_threads(int(nthreads))
        res = tn_numba_vec_parallel(mu, sig, a, b)
    else:
        raise ValueError("Unsupported target, use cpu or parallel.")

    return res.swapaxes(0, 1)


def sample_images(mean, std, num_samples, conf):
    """Samples for every pixel in Fourier space from a truncated Gaussian distribution
    based on the output of the network.

    Parameters
    ----------
    mean : torch.tensor
        mean values of the pixels with shape (number of images, number of samples,
        image size // 2 + 1, image_size)
    std : torch.tensor
        uncertainty values of the pixels with shape (number of images,
        number of samples, image size // 2 + 1, image_size)
    num_samples : int
        number of samples in Fourier space

    Returns
    -------
    dict
        resulting mean and standard deviation
    """
    mean_amp, mean_phase = mean[:, 0], mean[:, 1]
    std_amp, std_phase = std[:, 0], std[:, 1]
    num_img = mean_amp.shape[0]

    if conf["amp_phase"]:
        mode = ["amp", "phase"]
    else:
        mode = ["real", "imag"]

    # amplitude
    sampled_gauss_amp = trunc_rvs(
        mu=mean_amp,
        sig=std_amp,
        mode=mode[0],
        num_samples=num_samples,
    ).reshape(num_img * num_samples, 65, 128)

    # phase
    sampled_gauss_phase = trunc_rvs(
        mu=mean_phase,
        sig=std_phase,
        mode=mode[1],
        num_samples=num_samples,
    ).reshape(num_img * num_samples, 65, 128)

    # masks
    if conf["amp_phase"]:
        mask_invalid_amp = sampled_gauss_amp <= (0 - 1e-4)
        mask_invalid_phase = (sampled_gauss_phase <= (-np.pi - 1e-4)) | (
            sampled_gauss_phase >= (np.pi + 1e-4)
        )

        assert mask_invalid_amp.sum() == 0
        assert mask_invalid_phase.sum() == 0

    sampled_gauss = np.stack([sampled_gauss_amp, sampled_gauss_phase], axis=1)

    # pad resulting images and utilize symmetry
    sampled_gauss = F.pad(
        input=torch.tensor(sampled_gauss), pad=(0, 0, 0, 63), mode="constant", value=0
    )
    sampled_gauss_symmetry = sym_new(sampled_gauss, None)

    fft_sampled_symmetry = get_ifft(
        sampled_gauss_symmetry, amp_phase=conf["amp_phase"], scale=False
    ).reshape(num_img, num_samples, 128, 128)

    results = {
        "mean": fft_sampled_symmetry.mean(axis=1),
        "std": fft_sampled_symmetry.std(axis=1),
    }
    return results


def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = np.append(dict_1[key], value)
    return dict_3


class sampled_dataset:
    def __init__(self, bundle_path):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        if bundle_path == []:
            raise ValueError("No bundles found! Please check the names of your files.")
        self.bundle_path = bundle_path

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        bundle = h5py.File(self.bundle_path, "r")
        data = bundle["mean"]
        return data.shape[0]

    def __getitem__(self, i):
        mean = self.open_image("mean", i)
        std = self.open_image("std", i)
        true = self.open_image("true", i)
        return mean, std, true

    def open_image(self, var, i):
        bundle = h5py.File(self.bundle_path, "r")
        data = bundle[var]
        data = data[i]
        return data


def apply_normalization(img_test, norm_dict):
    """
    Applies one of currently two normalization methods if the training was normalized

    Parameters
    ----------
    img_test : torch.Tensor
        input image
    norm_dict : dictionary
        either empty (no normalization) or containing the factors

    Returns
    -------
    img_test : torch.Tensor
        normalized image
    norm_dict : dictionary
        updated dictionary
    """
    if norm_dict and "mean_real" in norm_dict:
        img_test[:, 0][img_test[:, 0] != 0] = (
            img_test[:, 0][img_test[:, 0] != 0] - norm_dict["mean_real"]
        ) / norm_dict["std_real"]

        img_test[:, 1][img_test[:, 1] != 0] = (
            img_test[:, 1][img_test[:, 1] != 0] - norm_dict["mean_imag"]
        ) / norm_dict["std_imag"]

    elif norm_dict and "max_scaling" in norm_dict:
        max_factors_real = torch.amax(img_test[:, 0], dim=(-2, -1), keepdim=True)
        max_factors_imag = torch.amax(
            torch.abs(img_test[:, 1]), dim=(-2, -1), keepdim=True
        )
        img_test[:, 0] *= 1 / torch.amax(img_test[:, 0], dim=(-2, -1), keepdim=True)
        img_test[:, 1] *= 1 / torch.amax(
            torch.abs(img_test[:, 1]), dim=(-2, -1), keepdim=True
        )
        norm_dict["max_factors_real"] = max_factors_real
        norm_dict["max_factors_imag"] = max_factors_imag

    return img_test, norm_dict


def rescale_normalization(pred, norm_dict):
    """
    Rescale the prediction after normalized training

    Parameters
    ----------
    pred : torch.Tensor
        predicted image
    norm_dict : dictionary
        either empty (no normalization) or containing the factors

    Returns
    -------
    pred : torch.Tensor
        recaled predicted image
    """
    if norm_dict and "mean_real" in norm_dict:
        pred[:, 0] = pred[:, 0] * norm_dict["std_real"] + norm_dict["mean_real"]
        pred[:, 1] = pred[:, 1] * norm_dict["std_imag"] + norm_dict["mean_imag"]

    elif norm_dict and "max_scaling" in norm_dict:
        pred[:, 0] *= norm_dict["max_factors_real"]
        pred[:, 1] *= norm_dict["max_factors_imag"]

    return pred


def preprocessing(conf):
    """
    Makes the necessary preprocessing for the evaluation methods analyzing the whole
    test dataset

    Parameters
    ----------
    conf : dictionary
        config file containing the settings

    Returns
    -------
    model : architecture
        model initialized with save file
    model_2 : architecture
        model initialized with save file
    loader : torch.Dataloader
        feeds the data batch-wise
    norm_dict : dictionary
        dict containing the normalization factors
    out_path : Path object
        path to the evaluation folder
    """
    # create DataLoader
    loader = create_databunch(
        conf["data_path"], conf["fourier"], conf["source_list"], conf["batch_size"]
    )
    model_path = conf["model_path"]
    out_path = Path(model_path).parent / "evaluation"
    out_path.mkdir(parents=True, exist_ok=True)

    img_size = loader.dataset[0][0][0].shape[-1]
    model, norm_dict = load_pretrained_model(
        conf["arch_name"], conf["model_path"], img_size
    )

    # Loads second model if the two channels were trainined separately
    model_2 = None
    if conf["model_path_2"] != "none":
        model_2, norm_dict = load_pretrained_model(
            conf["arch_name_2"], conf["model_path_2"], img_size
        )

    return model, model_2, loader, norm_dict, out_path


def process_prediction(conf, img_test, img_true, norm_dict, model, model_2):
    """
    Applies the normalization, gets and rescales a prediction and performs
    the inverse Fourier transformation.

    Parameters
    ----------
    conf : dictionary
        config files containing the settings
    img_test :  torch.Tensor
        input file for the network
    img_true : torch.tensor
        true image
    norm_dict : dictionary
        dict containing the normalization factors
    model : architecture
        model initialized with save file
    model_2 :
        model initialized with save file

    Returns
    -------
    ifft_pred : ndarray
        predicted source in image space
    ifft_truth : ndarray
        true source in image space
    """
    img_test, norm_dict = apply_normalization(img_test, norm_dict)
    pred = eval_model(img_test, model)
    pred = rescale_normalization(pred, norm_dict)
    if model_2 is not None:
        pred_2 = eval_model(img_test, model_2)
        pred_2 = rescale_normalization(pred_2, norm_dict)
        pred = torch.cat((pred, pred_2), dim=1)

    # apply symmetry
    if pred.shape[-1] == 128:
        img_dict = {"truth": img_true, "pred": pred}
        img_dict = apply_symmetry(img_dict)
        img_true = img_dict["truth"]
        pred = img_dict["pred"]

    ifft_truth = get_ifft(img_true, amp_phase=conf["amp_phase"])
    ifft_pred = get_ifft(pred, amp_phase=conf["amp_phase"])

    return ifft_pred, ifft_truth
