import torch
from math import pi


def bmul(vec, mat, axis=0):
    """Expand vector for batchwise matrix multiplication.
    Parameters
    ----------
    vec : 2dtensor
        vector for multiplication
    mat : 3dtensor
        matrix for multiplication
    axis : int, optional
        batch axis, by default 0
    Returns
    -------
    3dtensor
        Product of matrix multiplication. (bs, n, m)
    """
    mat = mat.transpose(axis, -1)
    return (mat * vec.expand_as(mat)).transpose(axis, -1)


def pca(image):
    """
    Compute the major components of an image. The Image is treated as a
    distribution.
    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be used as distribution
    Returns
    -------
    cog_x: Skalar
            X-position of the distributions center of gravity
    cog_y: Skalar
            Y-position of the distributions center of gravity
    psi: Skalar
            Angle between first mjor component and x-axis
    """
    torch.set_printoptions(precision=16)

    pix_x, pix_y, image = im_to_array_value(image)

    cog_x = (torch.sum(pix_x * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)
    cog_y = (torch.sum(pix_y * image, axis=1) / torch.sum(image, axis=1)).unsqueeze(-1)

    delta_x = pix_x - cog_x
    delta_y = pix_y - cog_y

    inp = torch.cat([delta_x.unsqueeze(1), delta_y.unsqueeze(1)], dim=1)

    cov_w = bmul(
        (cog_x - 1 * torch.sum(image * image, axis=1).unsqueeze(-1) / cog_x).squeeze(1),
        (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
    )

    eig_vals_torch, eig_vecs_torch = torch.linalg.eigh(cov_w, UPLO="U")

    psi_torch = torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1])

    return cog_x, cog_y, psi_torch


def calc_jet_angle(image):
    """Caluclate the jet angle from an image created with gaussian sources. This
    is achieved by a PCA.

    Parameters
    ----------
    image : ndarray
        input image
    Returns
    -------
    float
        slope of the line
    float
        intercept of the line
    float
        angle between the horizontal axis and the jet axis
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    image = image.clone()
    img_size = image.shape[-1]
    # ignore negative pixels, which can appear in predictions
    image[image < 0] = 0

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    bs = image.shape[0]

    # only use brightest pixel
    max_val = torch.tensor([(i.max() * 0.4) for i in image])
    max_arr = (torch.ones(img_size, img_size, bs) * max_val).permute(2, 0, 1)
    image[image < max_arr] = 0

    _, _, alpha_pca = pca(image)

    # Search for sources with two maxima
    maxima = []
    for i in range(image.shape[0]):
        a = torch.where(image[i] == image[i].max())
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

    m = torch.tan(pi / 2 - alpha_pca)
    n = y_mid - m * x_mid
    alpha = (alpha_pca) * 180 / pi
    return m, n, alpha


def im_to_array_value(image):
    """
    Transforms the image to an array of pixel coordinates and the containt
    intensity

    Parameters
    ----------
    image: Image or 2DArray (N, M)
            Image to be transformed

    Returns
    -------
    x_coords: Numpy 1Darray (N*M, 1)
            Contains the x-pixel-position of every pixel in the image
    y_coords: Numpy 1Darray (N*M, 1)
            Contains the y-pixel-position of every pixel in the image
    value: Numpy 1Darray (N*M, 1)
            Contains the image-value corresponding to every x-y-pair

    """
    num = image.shape[0]
    pix = image.shape[-1]

    a = torch.arange(0, pix, 1)
    grid_x, grid_y = torch.meshgrid(a, a, indexing="xy")
    x_coords = torch.cat(num * [grid_x.flatten().unsqueeze(0)])
    y_coords = torch.cat(num * [grid_y.flatten().unsqueeze(0)])
    value = image.reshape(-1, pix ** 2)
    return x_coords, y_coords, value
