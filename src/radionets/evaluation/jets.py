from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.modeling import fitting, models

from radionets.core.logging import _setup_logger

if TYPE_CHECKING:
    from torch import Tensor


__all__ = ["fitgaussian_crop", "fitgaussian_iterative"]

LOGGER = _setup_logger(namespace=__name__)


def fitgaussian_crop(
    image: Tensor | np.ndarray,
    amp_scale: float = 0.97,
    crop_size: float = 0.1,
):
    """Fitting a gaussian around the maximum

    Parameters
    ----------
    image : 2d array
        Image
    amp_scale : float
        Reduces the fitted amplitude, encounters (partially) the problem
        of overlapping gaussians: amplitude in image is too high -> fit is too
        high -> next gaussian is shifted to the outside
    crop_size : float
        proportionate size of the image after cropping

    Returns
    -------
    :class:`~astropy.modeling.models.Gaussian2D`
        Fitted astropy Gaussian model.
    """
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()

    size = image.shape[-1]
    image = image.clip(min=0)

    crop_dist = int((size * crop_size) // 2)
    maximum = np.unravel_index(image.argmax(), image.shape)

    crop_xmin = min(crop_dist, maximum[0])
    crop_xmax = min(crop_dist, size - maximum[0])
    crop_ymin = min(crop_dist, maximum[1])
    crop_ymax = min(crop_dist, size - maximum[1])

    image_crop = image[
        maximum[0] - crop_xmin : maximum[0] + crop_xmax,
        maximum[1] - crop_ymin : maximum[1] + crop_ymax,
    ]

    gaussian = models.Gaussian2D()
    lmf = fitting.LevMarLSQFitter()
    xx, yy = np.indices(image_crop.shape)
    result_lmf = lmf(gaussian, xx, yy, image_crop)

    # the parameters can't be adjusted directly, need help-array
    params = result_lmf.parameters
    params[0] *= amp_scale
    params[1] += maximum[0] - crop_xmin
    params[2] += maximum[1] - crop_ymin
    result_lmf.parameters = params

    return result_lmf


def fitgaussian_iterative(
    image: Tensor | np.ndarray,
    threshold: float = 0.05,
    max_iter: int = 10,
):
    """Fitting a gaussian iteratively around the maxima.
    Fit -> Subtract -> Fit -> Subtract ... until stopping criteria

    Parameters
    ----------
    image : :class:`~torch.Tensor` or :func:`~numpy.ndarray`
        Input image.
    threshold : float, optional
        The threshold at which the iterations are stopped.
        Default: 0.05
    max_iter : int, optional
        Maximum iterations. Default: 10

    Returns
    -------
    params_list : list of :class:`~astropy.modeling.models.Gaussian2D`
        List of fitted astropy model object(s).

    fits_list : list
        List of :func:`~numpy.ndarray` with fits.
    """
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()

    params_list = []
    fits_list = []

    for _ in range(max_iter):
        if image.max() <= threshold:
            break

        result_lmf = fitgaussian_crop(image)
        xx, yy = np.indices([image.shape[-1], image.shape[-1]])
        fit = result_lmf(xx, yy)

        params = result_lmf.parameters
        params[1], params[2] = params[2], params[1]
        params[3], params[4] = params[4], params[3]
        result_lmf.parameters = params

        # save, if gauss is not too narrow (e.g. one large pixel isn't meaningful here)
        if not np.any(params[3:5] < image.shape[-1] / 40):
            params_list.append(result_lmf)
            fits_list.append(fit)

        image -= fit

    return params_list, fits_list
