import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def compute_area_difference(CS_pred, CS_truth):
    """Compute the difference of the areas of truth and prediction.

    Parameters
    ----------
    CS_pred : contour object
        contour object of prediction
    CS_truth : contour object
        contour object of truth

    Returns
    -------
    float
        difference between area of truth and prediction
    """
    pred_x = CS_pred.collections[0].get_paths()[0].vertices[:, 0]
    pred_y = CS_pred.collections[0].get_paths()[0].vertices[:, 1]

    truth_x = CS_truth.collections[0].get_paths()[0].vertices[:, 0]
    truth_y = CS_truth.collections[0].get_paths()[0].vertices[:, 1]

    area_pred = 0.5 * np.sum(
        pred_y[:-1] * np.diff(pred_x) - pred_x[:-1] * np.diff(pred_y)
    )
    area_pred = np.abs(area_pred)

    area_truth = 0.5 * np.sum(
        truth_y[:-1] * np.diff(truth_x) - truth_x[:-1] * np.diff(truth_y)
    )
    area_truth = np.abs(area_truth)

    return area_pred / area_truth


def area_of_contour(ifft_pred, ifft_truth):
    """Create first contour of prediction and truth and return
    the area difference.

    Parameters
    ----------
    ifft_pred : ndarray
        source image of prediction
    ifft_truth : ndarray
        source image of truth

    Returns
    -------
    float
        area differencw
    """
    mpl.use("Agg")

    levels = [ifft_truth.max() * 0.1]

    CS1 = plt.contour(ifft_pred, levels=levels)

    plt.close()

    CS2 = plt.contour(ifft_truth, levels=levels)

    val = compute_area_difference(CS1, CS2)
    mpl.rcParams.update(mpl.rcParamsDefault)
    return val
