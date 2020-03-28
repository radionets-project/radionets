from torch.utils.data import DataLoader
import torch
import h5py
import numpy as np
from pathlib import Path


def normalize(x, m, s):
    return (x - m) / s


def do_normalisation(x, norm):
    """
    :param x        Object to be normalized
    :param norm     Pandas Dataframe which includes the normalisation factors
    """
    train_mean_real = torch.tensor(norm["train_mean_real"].values[0]).float()
    train_std_real = torch.tensor(norm["train_std_real"].values[0]).float()
    train_mean_imag = torch.tensor(norm["train_mean_imag"].values[0]).float()
    train_std_imag = torch.tensor(norm["train_std_imag"].values[0]).float()
    x[:, 0] = normalize(x[:, 0], train_mean_real, train_std_real)
    x[:, 1] = normalize(x[:, 1], train_mean_imag, train_std_imag)
    assert not torch.isinf(x).any()
    return x


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class h5_dataset:
    def __init__(self, bundle_paths, tar_fourier):
        """
        Save the bundle paths and the number of bundles in one file
        """
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))
        self.tar_fourier = tar_fourier

    def __call__(self):
        return print("This is the h5_dataset class.")

    def __len__(self):
        """
        Return the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        x = self.open_image("x", i)
        y = self.open_image("y", i)
        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def open_image(self, var, i):
        # at the moment all bundles contain 1024 images
        # should be variable in the future
        bundle_i = i // self.num_img
        image_i = i - bundle_i * self.num_img
        bundle = h5py.File(self.bundles[bundle_i], "r")
        data = bundle[var][image_i]
        if var == "x" or self.tar_fourier:
            data_amp, data_phase = split_real_imag(data)
            data_channel = combine_and_swap_axes(data_amp, data_phase).reshape(
                -1, data.shape[0] ** 2
            )
        else:
            data_channel = data.reshape(data.shape[0] ** 2)
        return torch.tensor(data_channel).float()


def combine_and_swap_axes(array1, array2):
    return np.swapaxes(np.dstack((array1, array2)), 2, 0)


def split_real_imag(array):
    """
    takes a complex array and returns the real and the imaginary part
    """
    return array.real, array.imag


def split_amp_phase(array):
    """
    takes a complex array and returns the amplitude and the phase
    """
    amp = np.abs(array)
    phase = np.angle(array)
    return amp, phase


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs, **kwargs),
    )


class DataBunch:
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


def save_bundle(path, bundle, counter, name="gs_bundle"):
    with h5py.File(path + str(counter) + ".h5", "w") as hf:
        hf.create_dataset(name, data=bundle)
        hf.close()


# open and save functions should be generalized in future versions


def open_bundle(path, mnist=False, gaussian=False):
    """
    open radio galaxy bundles created in first analysis step
    """
    f = h5py.File(path, "r")
    if mnist:
        bundle = np.array([f['x'], f['x']])
    if gaussian:
        bundle = np.array(f["gs_bundle"])
    return bundle


def get_bundles(path):
    """
    returns list of bundle paths located in a directory
    """
    data_path = Path(path)
    bundles = np.array([x for x in data_path.iterdir()])
    return bundles


def save_fft_pair(path, x, y, name_x="x", name_y="y"):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        hf.close()


def open_fft_pair(path):
    """
    open fft_pairs which were created in second analysis step
    """
    f = h5py.File(path, "r")
    bundle_x = np.array(f["x"])
    bundle_y = np.array(f["y"])
    return bundle_x, bundle_y


def mean_and_std(array):
    return array.mean(), array.std()
