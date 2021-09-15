from math import sqrt
from skimage.feature import blob_log
import torch
import numpy as np


def calc_blobs(ifft_pred, ifft_truth):
    if isinstance(ifft_pred, torch.Tensor):
        ifft_pred = ifft_pred.numpy()
    if isinstance(ifft_truth, torch.Tensor):
        ifft_truth = ifft_truth.numpy()
    tresh = ifft_truth.max() * 0.1
    kwargs = {
        "min_sigma": 1,
        "max_sigma": 10,
        "num_sigma": 100,
        "threshold": tresh,
        "overlap": 0.9,
    }
    blobs_log_pred = blob_log(ifft_pred, **kwargs)
    blobs_log_truth = blob_log(ifft_truth, **kwargs)
    # Compute radii in the 3rd column.
    blobs_log_pred[:, 2] = blobs_log_pred[:, 2] * sqrt(2)
    blobs_log_truth[:, 2] = blobs_log_truth[:, 2] * sqrt(2)
    return blobs_log_pred, blobs_log_truth


def crop_first_component(pred, truth, blob_truth):
    """Returns the cropped image with the first component of the
    true image for both prediction and truth.

    Parameters
    ----------
    pred : ndarray
        predicted source image
    truth : ndarray
        true source image
    blob_truth : list
        list with the coordiantes for the first component

    Returns
    -------
    ndarray
        cropped images
    """
    y, x, r = blob_truth
    x_coord, y_coord = corners(y, x, r)
    flux_truth = truth[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]
    flux_pred = pred[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]
    return flux_pred, flux_truth


def corners(x, y, r):
    """Generates the value range for cropping the first component out of
    the images.

    Parameters
    ----------
    y : float
        y coordiante
    x : float
        x coordinate
    r : float
        radius of the first component

    Returns
    -------
    list
        start and end point for the cropping
    """
    r = int(np.round(r))
    x = int(x)
    y = int(y)
    x_coord = [x - r, x + r + 1]
    y_coord = [y - r, y + r + 1]

    return x_coord, y_coord
