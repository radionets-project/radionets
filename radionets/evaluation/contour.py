import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def area(CS1, CS2):
    pred_x = CS1.collections[0].get_paths()[0].vertices[:, 0]
    pred_y = CS1.collections[0].get_paths()[0].vertices[:, 1]

    truth_x = CS2.collections[0].get_paths()[0].vertices[:, 0]
    truth_y = CS2.collections[0].get_paths()[0].vertices[:, 1]

    area_pred = 0.5 * np.sum(
        pred_y[:-1] * np.diff(pred_x) - pred_x[:-1] * np.diff(pred_y)
    )
    area_pred = np.abs(area_pred)

    area_truth = 0.5 * np.sum(
        truth_y[:-1] * np.diff(truth_x) - truth_x[:-1] * np.diff(truth_y)
    )
    area_truth = np.abs(area_truth)

    return area_truth - area_pred


def area_of_contour(ifft_pred, ifft_truth):
    mpl.use("Agg")

    levels = [ifft_truth.max() * 0.1]

    CS1 = plt.contour(ifft_pred, levels=levels)

    plt.close()

    CS2 = plt.contour(ifft_truth, levels=levels)

    val = area(CS1, CS2)
    return val
