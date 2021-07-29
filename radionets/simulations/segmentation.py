import numpy as np
from scipy.ndimage import zoom
import warnings

from scipy.ndimage.measurements import maximum

warnings.filterwarnings("ignore", message="covariance is not positive-semidefinite")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


def flip_num(a, b):
    """
    Used to flip the two principial components of the covariance matrix for the
    elongated gaussian to get more randomness in simulations.

    Parameters
    ----------
    a, b: float
        Numbers to get flipped

    Returns
    a, b: float
        Flipped or not flipped numbers
    """
    if np.random.rand() < 0.5:
        return a, b
    else:
        return b, a


def create_pointsource(size=64, edge_factor=0.1):
    """
    Creates very small sources.

    Parameters
    ----------
    size: int
        Size of the simulated image
    edge_factor: float
        Determines the distance (in percent of the size of the image) of the
        source to the edges.
        If 0:  center of source can be on the edge
        If <0: center of source can be outside the image
        If >0: center of source is inside the image

    Returns
    -------
    hist: ndarray
        2d array of size of image containing one point-source
    """
    mean = np.random.randint(size * edge_factor, size * (1 - edge_factor), size=2)
    principal_scale = size ** 2 / np.random.uniform(10000, 30000)
    cov = np.zeros((2, 2))
    cov[0][0] = np.random.uniform(1, 1.5) * principal_scale
    cov[1][1] = np.random.uniform(1, 1.5) * principal_scale
    cov[0][1] = np.random.uniform(-1, 1) * principal_scale / 5
    cov[1][0] = np.random.uniform(-1, 1) * principal_scale / 5
    gauss_size = int(
        size * 100 * ((cov[0][0] + cov[1][1]) / 2 - abs((cov[0][1] + cov[0][1]) / 2))
    )
    gauss = np.random.multivariate_normal(mean, cov, size=gauss_size)
    hist, _, _ = np.histogram2d(
        gauss[:, 0], gauss[:, 1], bins=size, range=[[0, size], [0, size]]
    )
    return hist


def create_gaussian(size=64, edge_factor=0.1):
    """
    Creates large, mostly circular sources.

    Parameters
    ----------
    size: int
        Size of the simulated image
    edge_factor: float
        Determines the distance (in percent of the size of the image) of the
        source to the edges.
        If 0:  center of source can be on the edge
        If <0: center of source can be outside the image
        If >0: center of source is inside the image

    Returns
    -------
    hist: ndarray
        2d array of size of image containing one gauss-source
    """
    mean = np.random.randint(size * edge_factor, size * (1 - edge_factor), size=2)
    principal_scale = size ** 2 / np.random.uniform(300, 1000)
    cov = np.zeros((2, 2))
    cov[0][0] = np.random.uniform(1, 1.5) * principal_scale
    cov[1][1] = np.random.uniform(1, 1.5) * principal_scale
    cov[0][1] = np.random.uniform(-1, 1) * principal_scale / 5
    cov[1][0] = np.random.uniform(-1, 1) * principal_scale / 5
    gauss_size = int(
        size * 50 * ((cov[0][0] + cov[1][1]) / 2 - abs((cov[0][1] + cov[0][1]) / 2))
    )
    gauss = np.random.multivariate_normal(mean, cov, size=gauss_size)
    hist, _, _ = np.histogram2d(
        gauss[:, 0], gauss[:, 1], bins=size, range=[[0, size], [0, size]]
    )
    return hist


def create_elongated(size=64, edge_factor=0.1):
    """
    Creates large, elongated sources

    Parameters
    ----------
    size: int
        Size of the simulated image
    edge_factor: float
        Determines the distance (in percent of the size of the image) of the
        source to the edges.
        If 0:  center of source can be on the edge
        If <0: center of source can be outside the image
        If >0: center of source is inside the image

    Returns
    -------
    hist: ndarray
        2d array of size of image containing one elongated source
    """
    mean = np.random.randint(size * edge_factor, size * (1 - edge_factor), size=2)
    principal_scale = size ** 2 / np.random.uniform(400, 1500)
    cov = np.zeros((2, 2))
    cov[0][0] = np.random.uniform(1, 1.5) * principal_scale
    cov[1][1] = np.random.uniform(0.2, 0.3) * principal_scale
    cov[0][0], cov[1][1] = flip_num(cov[0][0], cov[1][1])
    cov[0][1] = np.random.uniform(-1, 1) * principal_scale / 2
    cov[1][0] = np.random.uniform(-1, 1) * principal_scale / 2
    gauss_size = int(
        size * 50 * ((cov[0][0] + cov[1][1]) / 2 - abs((cov[0][1] + cov[0][1]) / 2))
    )
    gauss = np.random.multivariate_normal(mean, cov, size=gauss_size)
    hist, _, _ = np.histogram2d(
        gauss[:, 0], gauss[:, 1], bins=size, range=[[0, size], [0, size]]
    )
    return hist


def create_image(size=64, noise=0, edge_factor=0.1):
    """
    Combines the simulated sources and adds noise.

    Parameters
    ----------
    size: int
        Size of the simulated image
    noise: float
        Noise factor
    edge_factor: float
        Determines the distance (in percent of the size of the image) of the
        source to the edges.
        If 0:  center of source can be on the edge
        If <0: center of source can be outside the image
        If >0: center of source is inside the image

    Returns
    -------
    hist: ndarray
        3d array of shape [classes+1, size, size] containing all simulated
        sources and noise
    """
    x0 = np.zeros((1, size, size))
    x1 = np.zeros((1, size, size))
    x2 = np.zeros((1, size, size))
    x_noise = np.zeros((1, size, size))
    for i in range(np.random.randint(5, 10)):
        r = np.random.randint(3)
        if r == 0:  # Class 1: Pointsource
            x0 += create_pointsource(size, edge_factor).reshape(1, size, size)
        if r == 1:  # Class 2: Gaussian
            x1 += create_gaussian(size, edge_factor).reshape(1, size, size)
        if r == 2:  # Class 3: Elongated Gaussian
            x2 += create_elongated(size, edge_factor).reshape(1, size, size)

    x = np.concatenate((x0, x1, x2), axis=0)
    if noise > 0:
        # large values will increase structure in noise
        noise_structure = 4
        size_ratio = size / noise_structure
        size_int = np.int(size_ratio)
        size_rescale = size_ratio / size_int * noise_structure
        size_noise = (1, size_int, size_int)
        max_noise = noise * np.sum(x, axis=0).max()
        # make noise strength random
        max_noise_rnd = np.random.uniform(0, max_noise)
        x_noise = np.random.uniform(0, max_noise_rnd, size=size_noise)
        x_noise = zoom(x_noise, (1, size_rescale, size_rescale))

    x = np.concatenate((x_noise, x0, x1, x2), axis=0)
    return x.astype("int16")


def create_data_bunch(config):
    """
    Creation of data in bunches.

    Parameters
    ----------
    config: .toml configuration
        configurations for the simulation

    Returns
    -------
    x: ndarray
        Data for segmentation
    y: ndarray
        Labels for segmentation task
    """
    n = config["image_options"]["bundle_size"]
    size = config["image_options"]["img_size"]
    noise = config["image_options"]["noise_level"]
    edge_factor = config["image_options"]["edge_factor"]
    classes = config["gaussians"]

    n_to_sim = np.int(np.ceil(n / 8))
    x = np.empty((n_to_sim * 8, len(classes) + 1, size, size), dtype="int16")
    for i in range(n_to_sim):
        x[i*8] = create_image(size, noise, edge_factor)
        x[i*8+1] = np.flip(x[i*8], axis=1)
        x[i*8+2] = np.flip(x[i*8], axis=2)
        x[i*8+3] = np.flip(x[i*8], axis=(1, 2))
        x[i*8+4] = np.rot90(x[i*8], axes=(1, 2))
        x[i*8+5] = np.flip(x[i*8+4], axis=1)
        x[i*8+6] = np.flip(x[i*8+4], axis=2)
        x[i*8+7] = np.flip(x[i*8+4], axis=(1, 2))
    x = x[0:n]
    y = get_y(x, config)
    x = np.sum(x, axis=1, keepdims=True)
    return x, y


def get_y(x, config):
    """
    Generate labels for segmentation task.

    Parameters
    ----------
    x: ndarray
        Data array with sources with shape [classes+1, size, size]
    mode: "noisy" or "clean"
        noisy: Generate labels with underlying noise.
        clean: Generate labels without noise.

    Returns
    -------
    y: ndarray
        Labels for segmentation task
    """
    mode = config["label_options"]["mode"]

    n, n_class, size, _ = np.shape(x)

    epsilon = 1e-5

    x_sum = np.sum(x, axis=1, keepdims=True).repeat(n_class, axis=1)
    x_max = np.max(x_sum, axis=(1, 2, 3)).repeat(size**2).reshape(n, 1, size, size)

    noise = np.delete(x, (1, 2, 3), axis=1)

    x_clean = np.delete(x, 0, axis=1)
    x_clean_sum = np.sum(x_clean, axis=1, keepdims=True)
    x_clean_max = np.max(x_clean_sum, axis=(1, 2, 3)).repeat(size**2).reshape((n, 1, size, size))

    # first part of mask is for high-noise images, second is for low-noise images
    # 0.317 is 1 - first sigma of norm. dist., 0.7 is empirical
    mask = np.logical_and(x_clean_sum > 0.7 * noise, x_clean_sum > 0.317 * x_clean_max)

    y_init = np.zeros_like(x)
    y_init[:, 0] = 1

    if mode == "noisy":
        mask = mask.repeat(n_class, axis=1)
        y = np.where(mask, x / (x_sum + epsilon), y_init)
    elif mode == "clean":
        y = np.where(mask.squeeze(), np.argmax(x_clean, axis=1) + 1, 0)
        y = np.eye(n_class)[y].transpose(0, 3, 1, 2)
    return y
