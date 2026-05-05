from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


__all__ = ["eval_area", "eval_intensity", "intensity_ratio", "source_area_ratio"]

LOGGER = logging.getLogger("radionets")


def _compute_source_area(vertices: ArrayLike) -> float:
    """Helper function to compute area of a source
    using the shoelace formula.

    Parameters
    ----------
    vertices : :func:`~numpy.ndarray`, shape (N, 2)
        Polygon (source) vertices as (x, y) coordinates.

    Returns
    -------
    float
        Area of the source.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    s1 = np.dot(x, np.roll(y, -1))
    s2 = np.dot(y, np.roll(x, -1))

    return 0.5 * np.abs(s1 - s2)


def source_area_ratio(
    ifft_pred: ArrayLike,
    ifft_target: ArrayLike,
    threshold: float = 0.05,
) -> list[float] | float:
    """Compute area ratio at 5% of the maximum of prediction and target.

    Parameters
    ----------
    ifft_pred : ndarray
        Predicted source image(s).
    ifft_target : ndarray
        Target source image(s).
    threshold : float, optional
        Percentile threshold of maximum (true) flux computed on the target.
        Default: 0.05

    Returns
    -------
    list[float] | float
        Ratio of predicted and targeted source areas. If batch size is 1,
        returns ratio as float.
    """
    if isinstance(ifft_target, torch.Tensor):
        levels = ifft_target.amax(dim=[-2, -1]) * threshold
    else:
        levels = ifft_target.max(axis=(-2, -1)) * threshold

    fig, ax = plt.subplots()
    cs_pred = ax.contour(ifft_pred, levels=[levels])
    cs_target = ax.contour(ifft_target, levels=[levels])
    plt.close(fig)

    area_pred = np.array(
        [_compute_source_area(path.vertices) for path in cs_pred.get_paths()]
    )
    area_target = np.array(
        [_compute_source_area(path.vertices) for path in cs_target.get_paths()]
    )

    return area_pred.sum() / area_target.sum()


def intensity_ratio(
    pred: ArrayLike, target: ArrayLike, threshold=0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Compute intensity ratios between prediction
    and ground target images.

    Parameters
    ----------
    pred : :func:`~numpy.ndarray`, shape (..., H, W)
        Prediction image(s).
    target : :func:`~numpy.ndarray`, shape (..., H, W)
        Ground target image(s).

    Returns
    -------
    sum_ratio : :func:`~numpy.ndarray`
        Ratio of summed intensities (prediction / target).
    peak_ratio : :func:`~numpy.ndarray`
        Ratio of peak intensities (prediction / target).
    """
    if pred.ndim == 2:
        pred = pred[None, ...]

    if target.ndim == 2:
        target = target[None, ...]

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    threshold = target.max(axis=(-2, -1), keepdims=True) * threshold

    source_target = np.where(target > threshold, target, 0)
    source_pred = np.where(pred > threshold, pred, 0)

    sum_ratio = source_pred.sum(axis=(-2, -1)) / source_target.sum(axis=(-2, -1))
    peak_ratio = source_pred.max(axis=(-2, -1)) / source_target.max(axis=(-2, -1))

    return sum_ratio, peak_ratio


def eval_intensity(config, preds: torch.Tensor, targets: torch.Tensor) -> None:
    LOGGER.info("Evaluating integrated flux and peak flux...")
    sum_ratio, peak_ratio = intensity_ratio(preds, targets)

    LOGGER.info(f"Mean integrated flux ratio: {sum_ratio.mean()}")
    LOGGER.info(f"Mean peak flux ratio: {peak_ratio.mean()}")

    file_path = config.paths.save_path / "flux_intensity.csv"
    DataFrame(data={"integrated_flux": sum_ratio, "peak_flux": peak_ratio}).to_csv(
        file_path, index=False
    )
    LOGGER.info(f"Saved to {file_path}")


def eval_area(config, preds: torch.Tensor, targets: torch.Tensor) -> None:
    LOGGER.info("Evaluating integrated flux and peak flux...")

    ratios = []
    for p, t in zip(preds, targets):
        ratios.append(source_area_ratio(p, t, config.evaluation.area.threshold))

    LOGGER.info(f"Mean area ratio: {np.array(ratios).mean()}")

    file_path = config.paths.save_path / "area_ratios.csv"
    DataFrame(data={"area_ratio": ratios}).to_csv(file_path, index=False)
    LOGGER.info(f"Saved to {file_path}")
