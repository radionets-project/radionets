import gzip
import pickle
import numpy as np
import h5py
from skimage.transform import resize
from simulations.gaussian_simulations import add_noise
from dl_framework.data import save_fft_pair
import os


def open_mnist(path):
    with gzip.open(path, "rb") as f:
        ((train_x, _), (valid_x, _), _) = pickle.load(f, encoding="latin-1")
    return train_x, valid_x


def adjust_outpath(path, option):
    counter = 0
    filename = path + "/fft_bundle_" + option + "{}.h5"
    while os.path.isfile(filename.format(counter)):
        counter += 1
    return filename.format(counter)


def prepare_mnist_bundles(bundle, path, option, noise=False, pixel=63):
    """
    Resize images to specific squared pixel size and calculate fft

    Parameters
    ----------
    image: 2d array
        input image
    pixel: int
        number of pixel in x and y

    Returns
    -------
    x: 2d array
        fft of rescaled input image
    y: 2d array
        rescaled input image
    """
    y = [
        resize(bund, (pixel, pixel), anti_aliasing=True, mode="constant",)
        for bund in bundle
    ]
    y_prep = y.copy()
    if noise:
        y_prep = add_noise(y_prep)
    x = np.fft.fftshift(np.fft.fft2(y_prep))
    path = adjust_outpath(path, option)
    save_fft_pair(path, x, y)


def reshape_img(img, size=64):
    return img.reshape(size, size)


def write_h5(path, x, y, name_x="x_train", name_y="y_train"):
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        hf.close()


def get_h5_data(path, columns):
    """ Load mnist h5 data """
    f = h5py.File(path, "r")
    x = np.array(f[columns[0]])
    y = np.array(f[columns[1]])
    return x, y


def create_mask(ar):
    """ Generating mask with min and max value != inf"""
    val = ar.copy()
    val[np.isinf(val)] = 0
    low = val.min()
    high = val.max()
    mask = (low < ar) & (ar < high)
    return mask


def split_real_imag(array):
    """
    takes a complex array and returns the real and the imaginary part
    """
    return array.real, array.imag


def mean_and_std(array):
    return array.mean(), array.std()


def combine_and_swap_axes(array1, array2):
    """"
    Pair with dstack each element of the arrays with the opposing one,
    like element 1 of array1 with element 1 of array2 and so one.
    Then swap the axis in this way, that one can axes the real part
    with array[:, 0] and the imaginary part with array[:, 1]
    """
    return np.swapaxes(np.dstack((array1, array2)), 1, 2)
