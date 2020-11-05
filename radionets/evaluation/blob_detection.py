from math import sqrt
from skimage.feature import blob_log


def calc_blobs(ifft_pred, ifft_truth):
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
