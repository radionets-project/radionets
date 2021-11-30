import numpy as np


def get_min_max(element, index):
    min_val = element[index, :][element[4, :] == 1.0].min()
    max_val = element[index, :][element[4, :] == 1.0].max()
    arg_max = np.argmax(element[index, :][element[4, :] == 1.0])
    arg_min = np.argmin(element[index, :][element[4, :] == 1.0])

    return min_val, max_val, arg_min, arg_max


def get_length_extended(element):
    """Identify the first and the last gaussian component in the extended source and
    compute the distance. Last, add the extension of the source to the length.

    Parameters
    ----------
    element : ndarray
        array which contains all sources in the given image

    Returns
    -------
    ndarray
        length of extended source
    """

    x_min, x_max, arg_min_x, arg_max_x = get_min_max(element, 0)
    y_min, y_max, arg_min_y, arg_max_y = get_min_max(element, 1)

    sig_x_max = element[2, :][element[4, :] == 1.0][arg_max_x]
    sig_x_min = element[2, :][element[4, :] == 1.0][arg_min_x]
    sig_y_max = element[3, :][element[4, :] == 1.0][arg_max_y]
    sig_y_min = element[3, :][element[4, :] == 1.0][arg_min_y]

    # multiply with the factor for full width-half maximum
    extend_max = np.sqrt(sig_x_max ** 2 + sig_y_max ** 2) * 2.35
    extend_min = np.sqrt(sig_x_min ** 2 + sig_y_min ** 2) * 2.35
    # compute amount of vector
    laenge_x = (x_max - x_min) ** 2
    laenge_y = (y_max - y_min) ** 2
    laenge_extend = np.sqrt(laenge_x + laenge_y) + extend_max + extend_min

    return laenge_extend


def get_length_point(element):
    """Compute linear extend of pointsources.

    Parameters
    ----------
    element : ndarray
        array which contains all sources in the given image

    Returns
    -------
    ndarray
        lengths of the pointsources in the image
    """

    laenge_y_point = element[3, :][element[4, :] == 0.0] * 2
    laenge_x_point = element[2, :][element[4, :] == 0.0] * 2
    laenge_point = np.max([laenge_x_point, laenge_y_point], axis=0)

    return laenge_point


def flux_comparison(pred, truth, source_list):
    fluxes_pred = []
    fluxes_truth = []
    sigs_x = []
    sigs_y = []
    laenge = np.array([])
    for i, element in enumerate(source_list):
        mean_pred = np.array([])
        mean_truth = np.array([])
        for blob in element.T:
            y, x, sig_x, sig_y, mask = blob

            x_low = int(np.floor(x - sig_x))
            if x_low < 0:
                x_low = 0

            x_high = int(np.ceil(x + sig_x + 1))
            if x_high > 63:
                x_high = 63

            y_low = int(np.floor(y - sig_y))
            if y_low < 0:
                y_low = 0

            y_high = int(np.ceil(y + sig_y + 1))
            if y_high > 63:
                y_high = 63

            flux_truth = truth[i, int(x_low) : int(x_high), int(y_low) : int(y_high)]
            flux_pred = pred[i, int(x_low) : int(x_high), int(y_low) : int(y_high)]

            mean_pred = np.append(mean_pred, flux_pred.mean())
            mean_truth = np.append(mean_truth, flux_truth.mean())
            sigs_x.append(sig_x)
            sigs_y.append(sig_y)

        # sum over extended fluxes for truth and pred
        c = mean_pred[element[4, :] == 1.0].sum()
        mean_pred = np.append(mean_pred[element[4, :] == 0.0], c)
        c = mean_truth[element[4, :] == 1.0].sum()
        mean_truth = np.append(mean_truth[element[4, :] == 0.0], c)

        fluxes_pred.append(mean_pred)
        fluxes_truth.append(mean_truth)

        # append lengths for point and extended sources
        laenge_extend = get_length_extended(element)
        laenge_point = get_length_point(element)
        laenge = np.append(laenge, laenge_point)
        laenge = np.append(laenge, laenge_extend)

    return (
        np.array(fluxes_pred, dtype="object"),
        np.array(fluxes_truth, dtype="object"),
        laenge,
    )
