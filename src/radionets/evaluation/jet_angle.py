from math import pi

import torch
from pandas import DataFrame

from radionets.core.logging import _setup_logger

LOGGER = _setup_logger(namespace=__name__)


def bmul(vec: torch.Tensor, mat: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Expand vector for batchwise matrix multiplication.

    Parameters
    ----------
    vec : :class:`~torch.Tensor`, shape (B, N)
        Vector for multiplication.
    mat : :class:`~torch.Tensor`, shape (B, N, M)
        Matrix for multiplication.
    axis : int, optional
        Batch axis. Default: ``0``
    Returns
    -------
    :class:`~torch.Tensor`, shape (B, N, M)
        Product of matrix multiplication.
    """
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


def _im2array_value(
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transforms the image to an array of pixel coordinates and
    its intensities.

    Parameters
    ----------
    image: :class:`~torch.Tensor`, shape (B, H, W)
        Batch of images to be transformed.

    Returns
    -------
    x_coords : :class:`~torch.Tensor`, shape (B, H * W)
        Contains the x-position of every pixel in the image
    y_coords : :class:`~torch.Tensor`, shape (B, H * W)
        Contains the y-position of every pixel in the image
    value : :class:`~torch.Tensor`, shape (B, H * W)
        Contains the intensity value corresponding to every x-y-pair
    """
    # NOTE: This assumes quadratic images
    batch_size, img_size, _ = image.shape
    device = image.device

    a = torch.arange(img_size, device=device)
    grid_x, grid_y = torch.meshgrid(a, a, indexing="xy")

    x_coords = grid_x.ravel().unsqueeze(0).expand(batch_size, -1)
    y_coords = grid_y.ravel().unsqueeze(0).expand(batch_size, -1)
    value = image.reshape(-1, img_size**2)

    return x_coords, y_coords, value


def pca(
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the major components of an image. The image is treated
    as a 2D distribution.

    Parameters
    ----------
    image : :class:`~torch.Tensor`, shape (B, H, W)
        Images to be used as distribution

    Returns
    -------
    cog_x : :class:`~torch.Tensor`, shape (B, 1)
        X-position of the distributions center of gravity
    cog_y : :class:`~torch.Tensor`, shape (B, 1)
        Y-position of the distributions center of gravity
    psi : :class:`~torch.Tensor`, shape (B,)
        Angle between first major component and x-axis
    """
    pix_x, pix_y, image = _im2array_value(image)

    image_sum = image.sum(dim=1, keepdim=True)
    cog_x = (pix_x * image).sum(dim=1, keepdim=True) / image_sum
    cog_y = (pix_y * image).sum(dim=1, keepdim=True) / image_sum

    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    inp = torch.stack([delta_x, delta_y], dim=1)

    cov_w = bmul(
        (cog_x - 1 * torch.sum(image * image, dim=1).unsqueeze(-1) / cog_x).squeeze(1),
        (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
    )

    _, eig_vecs_torch = torch.linalg.eigh(cov_w, UPLO="U")
    psi_torch = torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1])

    return cog_x, cog_y, psi_torch


def jet_angle(
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the jet angle from an image consisting of
    (approx.) gaussian sources using a PCA.

    Parameters
    ----------
    image : :class:`~torch.Tensor`, shape (B, H, W)
        Input images

    Returns
    -------
    m : :class:`~torch.Tensor`, shape (B,)
        Slope of the line
    n : :class:`~torch.Tensor`, shape (B,)
        Intercept of the line
    alpha : :class:`~torch.Tensor`, shape (B,)
        Angle between the horizontal axis and the jet axis
    """
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)

    image = image.clone()

    # ignore negative pixels that can appear in predictions
    image = image.clamp(min=0)

    if image.ndim == 2:
        image = image.unsqueeze(0)

    # unpack first or first two dims to batch_size, e.g. if
    # ndim is 4 (num_batches, images_per_batch, H, W),
    # batch_size also contains the number of batches.
    # If only one batch and ndim is 3, batch_size is only the number
    # of images per batch
    *batch_size, img_size, _ = image.shape

    # only use pixels above 40% of peak flux
    max_vals = image.amax(dim=(-2, -1))
    threshold = (0.4 * max_vals).view(*batch_size, 1, 1)
    image = torch.where(image >= threshold, image, torch.zeros_like(image))

    _, _, alpha_pca = pca(image)

    # Search for sources with two maxima
    maxima = []
    for img in image:
        a = torch.where(img == img.max())
        if len(a[0]) > 1:
            # if two maxima are found, interpolate to the middle in x and y direction
            mid_x = (a[0][1] - a[0][0]) // 2 + a[0][0]
            mid_y = (a[1][1] - a[1][0]) // 2 + a[1][0]
            maxima.extend([(mid_x, mid_y)])
        else:
            maxima.extend([a])

    vals = torch.tensor(maxima)
    x_mid = vals[:, 0]
    y_mid = vals[:, 1]

    m = torch.tan(torch.tensor(pi / 2, device=image.device) - alpha_pca)
    n = y_mid - m * x_mid
    alpha = torch.rad2deg(alpha_pca)

    return m, n, alpha


def eval_jet_angle(config, preds, targets):
    LOGGER.info("Evaluating jet angle...")
    m_pred, n_pred, alpha_pred = jet_angle(torch.tensor(preds))
    m_target, n_target, alpha_target = jet_angle(torch.tensor(targets))

    diff = (alpha_pred - alpha_target).numpy()

    file_path = config.paths.save_path / "jet_angles_diff.csv"
    DataFrame(data={"diff": diff}).to_csv(file_path, index=False)

    LOGGER.info(f"Saved to {file_path}")
