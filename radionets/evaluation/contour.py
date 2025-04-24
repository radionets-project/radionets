import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def compute_area_ratio(CS_pred, CS_truth):
    """Compute the ratio of the areas of truth and prediction.

    Parameters
    ----------
    CS_pred : contour object
        contour object of prediction
    CS_truth : contour object
        contour object of truth

    Returns
    -------
    float
        ratio between area of truth and prediction
    """
    areas_truth = np.array([])
    areas_pred = np.array([])

    for area in CS_truth.collections[0].get_paths():
        truth_x = area.vertices[:, 0]
        truth_y = area.vertices[:, 1]

        area_truth = 0.5 * np.sum(
            truth_y[:-1] * np.diff(truth_x) - truth_x[:-1] * np.diff(truth_y)
        )
        area_truth = np.abs(area_truth)
        areas_truth = np.append(areas_truth, area_truth)

    for area in CS_pred.collections[0].get_paths():
        pred_x = area.vertices[:, 0]
        pred_y = area.vertices[:, 1]

        area_pred = 0.5 * np.sum(
            pred_y[:-1] * np.diff(pred_x) - pred_x[:-1] * np.diff(pred_y)
        )
        area_pred = np.abs(area_pred)
        areas_pred = np.append(areas_pred, area_pred)

    return areas_pred.sum() / areas_truth.sum()


def area_of_contour(ifft_pred, ifft_truth):
    """Create first contour of prediction and truth and return
    the area ratio.

    Parameters
    ----------
    ifft_pred : ndarray
        source image of prediction
    ifft_truth : ndarray
        source image of truth

    Returns
    -------
    float
        area difference
    """
    mpl.use("Agg")

    levels = [ifft_truth.max() * 0.05]

    CS1 = plt.contour(ifft_pred, levels=levels)

    plt.close()

    CS2 = plt.contour(ifft_truth, levels=levels)

    val = compute_area_ratio(CS1, CS2)
    mpl.rcParams.update(mpl.rcParamsDefault)
    return val


def analyse_intensity(pred, truth):
    if len(pred.shape) == 2:
        pred = pred.reshape(1, pred.shape[-2], pred.shape[-1])
        truth = truth.reshape(1, truth.shape[-2], truth.shape[-1])

    threshold = (truth.max(-1).max(-1) * 0.1).reshape(truth.shape[0], 1, 1)
    source_truth = np.where(truth > threshold, truth, 0)
    source_pred = np.where(pred > threshold, pred, 0)

    sum_truth = source_truth.sum(-1).sum(-1)
    sum_pred = source_pred.sum(-1).sum(-1)

    peak_truth = source_truth.max(-1).max(-1)
    peak_pred = source_pred.max(-1).max(-1)

    return sum_pred / sum_truth, peak_pred / peak_truth
