import numpy as np


def get_boxsize(num_corners, num_pixel=63):
    factors = np.array([0.3, 0.22, 0.16])
    size = int(num_pixel * factors[num_corners - 2])
    return size


def select_box(rms, sensitivity=1e-6):
    for arr in rms:
        arr[arr > sensitivity] = 0
    rms_boxes = rms.astype(bool).sum(axis=0)
    return rms_boxes


def compute_rms(batch, size):
    rms1 = rms2 = rms3 = rms4 = np.ones(len(batch)) * -1
    rms1 = np.sqrt((batch[:, :size, :size].reshape(-1, size ** 2) ** 2).mean(axis=1))
    rms2 = np.sqrt((batch[:, :size, -size:].reshape(-1, size ** 2) ** 2).mean(axis=1))
    rms3 = np.sqrt((batch[:, -size:, :size].reshape(-1, size ** 2) ** 2).mean(axis=1))
    rms4 = np.sqrt((batch[:, -size:, -size:].reshape(-1, size ** 2) ** 2).mean(axis=1))
    return np.stack([rms1, rms2, rms3, rms4], axis=0)


def get_rms(ifft_truth, ifft_pred):
    rms_4_truth = compute_rms(ifft_truth, get_boxsize(4))
    rms_boxes = select_box(rms_4_truth, 1e-6)
    rms_3_truth = compute_rms(ifft_truth, get_boxsize(3))
    select_box(rms_3_truth)
    rms_2_truth = compute_rms(ifft_truth, get_boxsize(2))
    select_box(rms_2_truth)

    rms_4_pred = compute_rms(ifft_pred, get_boxsize(4))
    rms_3_pred = compute_rms(ifft_pred, get_boxsize(3))
    rms_2_pred = compute_rms(ifft_pred, get_boxsize(2))

    rms_3_pred[rms_3_truth == 0] = 0
    rms_2_pred[rms_2_truth == 0] = 0

    rms_truth = np.zeros(len(rms_boxes))
    rms_truth[rms_boxes == 4] = (
        np.sqrt(rms_4_truth[0:4, rms_boxes == 4] ** 2).sum(axis=0) / 4
    )
    rms_truth[rms_boxes == 3] = (
        np.sqrt(rms_3_truth[0:4, rms_boxes == 3] ** 2).sum(axis=0) / 3
    )
    rms_truth[rms_boxes == 2] = (
        np.sqrt(rms_2_truth[0:4, rms_boxes == 2] ** 2).sum(axis=0) / 2
    )

    rms_pred = np.zeros(len(rms_boxes))
    rms_pred[rms_boxes == 4] = (
        np.sqrt(rms_4_pred[0:4, rms_boxes == 4] ** 2).sum(axis=0) / 4
    )
    rms_pred[rms_boxes == 3] = (
        np.sqrt(rms_3_pred[0:4, rms_boxes == 3] ** 2).sum(axis=0) / 3
    )
    rms_pred[rms_boxes == 2] = (
        np.sqrt(rms_2_pred[0:4, rms_boxes == 2] ** 2).sum(axis=0) / 2
    )
    corners = np.ones((rms_4_truth.shape[-1], 4))
    corners[rms_4_truth.swapaxes(1, 0) == 0] = 0
    return rms_truth, rms_pred, rms_boxes, corners


def calc_dr(ifft_truth, ifft_pred):
    rms_truth, rms_pred, rms_boxes, corners = get_rms(ifft_truth, ifft_pred)
    peak_vals_truth = ifft_truth.reshape(-1, ifft_truth.shape[-1] ** 2).max(axis=1)
    peak_vals_pred = ifft_pred.reshape(-1, ifft_pred.shape[-1] ** 2).max(axis=1)
    dr_truth = peak_vals_truth[rms_truth != 0] / rms_truth[rms_truth != 0]
    dr_pred = peak_vals_pred[rms_pred != 0] / rms_pred[rms_pred != 0]
    return dr_truth, dr_pred, rms_boxes, corners
