"""Feature detection submodule."""

from __future__ import annotations

import logging
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import torch
from pandas import DataFrame
from skimage.feature import blob_log
from tqdm import tqdm

from radionets.utils.batch_size import AdaptiveBatchSize

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

LOGGER = logging.getLogger("radionets")


def calc_blobs(
    ifft_image: ArrayLike,
) -> tuple[NDArray, NDArray]:
    """Detect blobs using Laplacian of Gaussian in prediction
    and truth images.

    Parameters
    ----------
    ifft_pred : :class:`~torch.Tensor` or :func:`numpy.ndarray`
        Predicted image (inverse FFT result), shape (N, 3).
    ifft_truth : :class:`~torch.Tensor` or :func:`numpy.ndarray`
        Ground truth image (inverse FFT result), shape (N, 3).

    Returns
    -------
    blobs_log_pred : :func:`~numpy.ndarray`
        Detected blobs in prediction, shape (N, 3) with columns [y, x, radius].
    blobs_log_truth : :func:`~numpy.ndarray`
        Detected blobs in ground truth, shape (N, 3) with columns [y, x, radius].
    """
    if isinstance(ifft_image, torch.Tensor):
        ifft_image = ifft_image.detach().cpu().numpy()

    threshold = ifft_image.max() * 0.1
    kwargs = {
        "min_sigma": 1,
        "max_sigma": 10,
        "num_sigma": 100,
        "threshold": threshold,
        "overlap": 0.9,
    }

    blobs_log = blob_log(ifft_image, **kwargs)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    return blobs_log


def crop_first_component(
    pred: ArrayLike,
    truth: ArrayLike,
    blobs_target: ArrayLike | tuple,
) -> tuple[NDArray, NDArray]:
    """Return cropped images around the first component of the true image.

    Parameters
    ----------
    pred : :func:`~numpy.ndarray`
        Predicted source image.
    truth : :func:`~numpy.ndarray`
        True source image.
    blob_truth : list or tuple
        Coordinates (y, x, r) for the first component.

    Returns
    -------
    flux_pred : :func:`~numpy.ndarray`
        Cropped prediction image.
    flux_truth : :func:`~numpy.ndarray`
        Cropped truth image.
    """
    y, x, r = blobs_target[0], blobs_target[1], blobs_target[2]
    x_coord, y_coord = _corners(x, y, r)

    flux_truth = truth[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]
    flux_pred = pred[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]

    return flux_pred, flux_truth


def _corners(
    x: int | float,
    y: int | float,
    r: int | float,
) -> tuple[list[int], list[int]]:
    """Generate coordinate ranges for cropping the first component.

    Parameters
    ----------
    x : int or float
        X coordinate of the component center.
    y : int or float
        Y coordinate of the component center.
    r : int or float
        Radius of the first component.

    Returns
    -------
    x_coord : list of int
        Start and end indices for x-axis cropping.
    y_coord : list of int
        Start and end indices for y-axis cropping.
    """
    r = int(np.round(r))
    x = int(x)
    y = int(y)

    x_coord = [x - r, x + r + 1]
    y_coord = [y - r, y + r + 1]

    return x_coord, y_coord


def eval_mean_difference(config, preds, targets) -> None:
    LOGGER.info("Evaluating mean difference...")

    vals = []
    with AdaptiveBatchSize(
        preds, targets, initial_batch_size=config.general.batch_size
    ) as batched:
        for preds_batch, targets_batch in tqdm(batched):
            blobs_target = calc_blobs(targets_batch)[0].copy()
            flux_preds, flux_targets = crop_first_component(
                preds_batch, targets_batch, blobs_target
            )

            vals_batch = (
                (flux_preds.mean() - flux_targets.mean()) / flux_targets.mean() * 100
            )
            vals.append(vals_batch)

    vals = np.asarray(vals)
    LOGGER.info(f"Mean difference: {vals.mean()}")

    file_path = config.paths.save_path / "mean_diff.csv"
    DataFrame(data={"mean_diff": vals}).to_csv(file_path, index=False)
