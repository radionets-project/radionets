import numpy as np


def flux_comparison(pred, truth, source_list):
    fluxes_pred = []
    fluxes_truth = []
    sigs_x = []
    sigs_y = []
    for i, element in enumerate(source_list):
        for blob in element.T:
            y, x, sig_x, sig_y = blob
            sig_x *= 2.35
            sig_y *= 2.35

            x_low = int(np.floor(x - sig_x))
            if x_low < 0:
                x_low = 0

            x_high = int(np.ceil(x + sig_x + 1))
            if x_high > 62:
                x_high = 62

            y_low = int(np.floor(y - sig_y))
            if y_low < 0:
                y_low = 0

            y_high = int(np.ceil(y + sig_y + 1))
            if y_high > 62:
                y_high = 62

            flux_truth = truth[i, int(x_low) : int(x_high), int(y_low) : int(y_high)]
            flux_pred = pred[i, int(x_low) : int(x_high), int(y_low) : int(y_high)]

            fluxes_pred.append(flux_pred.mean())
            fluxes_truth.append(flux_truth.mean())
            sigs_x.append(sig_x)
            sigs_y.append(sig_y)

    return (
        np.array(fluxes_pred),
        np.array(fluxes_truth),
        np.array(sigs_x),
        np.array(sigs_y),
    )
