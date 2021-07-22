import numpy as np
from scipy.ndimage import zoom
import warnings

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

    x = np.concatenate((x0, x1, x2, x_noise), axis=0)
    return x.astype("int16")


def create_data_bunch(config):
    """
    Creation of data bunches.

    # n: number of images for the bunch
    # size: size of the image
    # noise: strength of noise in percent of the max. brightness of the image
    # edge_factor: larger values move the distribution towards the center [0,0.5)
    # x: accumulated data of the different distributions
    # y: label of the most common data in each bin of the 2d hist
    #
    # Memory is allocated and the images are created in bunches.
    """
    n = config["image_options"]["bundle_size"]
    size = config["image_options"]["img_size"]
    noise = config["image_options"]["noise_level"]
    edge_factor = config["image_options"]["edge_factor"]
    classes = config["gaussians"]

    x = np.empty((n, len(classes) + 1, size, size), dtype="int16")
    for i in range(n):
        x[i] = create_image(size, noise, edge_factor)
        if (i + 1) % 100 == 0:
            print("Created {} of {} data bunches".format(i + 1, n))

    y = get_y(x, config)
    print(x.shape)
    x = np.sum(x, axis=1, keepdims=True)
    print(x.shape)
    return x, y


def get_y(x, config):
    """
    Generate labels for segmentation task.

    Parameters
    ----------
    x: ndarray
        Data array with sources with shape [classes+1, size, size]
    smooth: boolean
        True: Generate labels proportional to the occurance of the classes.
        False: Generate labels with the maximum occurance of one class.
    faint: float
        Determines the background strength.

    Returns
    -------
    y: ndarray
        Labels for segmentation task
    """
    smooth = config["label_options"]["smooth"]
    faint = config["label_options"]["faint"]

    n, n_class, size, _ = np.shape(x)
    # delete axis with noise data for the labels, so the labels stay noise free
    x = np.delete(x, n_class - 1, axis=1)
    sum_image = np.sum(x, axis=1, keepdims=True)
    max_of_image = np.max(sum_image, axis=(2, 3)).squeeze()
    x_faint = (np.ones((n, 1, size, size)).T * max_of_image * faint).T

    if smooth:  # the class distribution is captured: y: [n, n_class+1, size, size]
        y = x / sum_image
        y_0 = np.ones((n, 1, size, size))
        y_0 = np.where(sum_image < x_faint, y_0, 0)
        y = np.where(sum_image > x_faint, y, 0)
        y = np.concatenate((y_0, y), axis=1)
        y[np.isnan(y)] = 0
    else:  # takes the largest occurance of one class: y: [n, size, size]
        x = np.concatenate((x_faint, x), axis=1)
        y = np.empty((n, size, size), dtype="int8")
        for i in range(n):
            y[i] = np.argmax(x[i], axis=0)
    return y
