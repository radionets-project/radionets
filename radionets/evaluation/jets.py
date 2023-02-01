import numpy as np
from astropy.modeling import models, fitting
from radionets.evaluation.plotting import plot_fitgaussian


def fitgaussian_crop(data, amp_scale=0.97, crop_size=0.1):
    """
    Fitting a gaussian around the maximum
    Parameters
    ----------
    data: 2d array
        Image
    amp_scale: float
        Reduces the fitted amplitude, encounters (partially) the problem
        of overlapping gaussians: amplitude in image is too high -> fit is too
        high -> next gaussian is shifted to the outside
    crop_size: float
        proportionate size of the image after cropping


    Returns
    -------
    result_lmf: astropy model
        Fitted astropy model object
    """
    size = data.shape[-1]
    data[data < 0] = 0

    crop_dist = int((size * crop_size) // 2)
    maximum = np.unravel_index(data.argmax(), data.shape)
    crop_xmin = crop_xmax = crop_ymin = crop_ymax = crop_dist
    if maximum[0] < crop_dist:
        crop_xmin = maximum[0]
    if maximum[1] < crop_dist:
        crop_ymin = maximum[1]
    if size - maximum[0] < crop_dist:
        crop_xmax = size - maximum[0]
    if size - maximum[1] < crop_dist:
        crop_ymax = size - maximum[1]
    data_crop = data[
        maximum[0] - crop_xmin : maximum[0] + crop_xmax,
        maximum[1] - crop_ymin : maximum[1] + crop_ymax,
    ]

    M = models.Gaussian2D()
    lmf = fitting.LevMarLSQFitter()
    xx, yy = np.indices([data_crop.shape[0], data_crop.shape[1]])
    result_lmf = lmf(M, xx, yy, data_crop)
    # the parameters can't be adjusted directly, need help-array
    params = result_lmf.parameters
    params[0] *= amp_scale
    params[1] += maximum[0] - crop_xmin
    params[2] += maximum[1] - crop_ymin
    result_lmf.parameters = params
    return result_lmf


def fitgaussian_iterativ(
    data, i=0, visualize=False, path=None, save=False, plot_format="pdf"
):
    """
    Fitting a gaussian iteratively around the maxima.
    Fit -> Substract -> Fit -> Substract ... until stopping criteria
    Parameters
    ----------
    data: 2d array
        Image
    i: int
        Index of input image
    visualize: bool
        If the gauss should be plotted or not
    path: string
        Path to where the image is saved
    save: bool
        If the image is saved in path or not
    plot_format: str
        Format of the saved filed (png, pdf, ...)

    Returns
    -------
    result_lmf: astropy model
        Fitted astropy model object
    """
    if visualize and path is None:
        print("Visualize is True, but no path is given.")
    if not visualize and path is not None:
        print("Visualize is False, but a path is given.")

    params_list = []
    fit_list = []
    j = 0
    max_iterations = 10
    data_backup = data

    while data.max() > 0.05 and j < max_iterations:
        result_lmf = fitgaussian_crop(data)
        xx, yy = np.indices([data.shape[-1], data.shape[-1]])
        fit = result_lmf(xx, yy)
        params = result_lmf.parameters
        params[1], params[2] = params[2], params[1]
        params[3], params[4] = params[4], params[3]
        result_lmf.parameters = params
        # save, if gauss is not too narrow (e.g. one large pixel isn't meaningful here)
        if not np.array(params[3:5] < data.shape[-1] / 40).any():
            params_list.append(result_lmf)
            fit_list.append(fit)
        data -= fit
        j += 1
    if visualize:
        plot_fitgaussian(data_backup, fit_list, params_list, i, path, save, plot_format)

    return params_list
