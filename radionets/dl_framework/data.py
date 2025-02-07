from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from radionets.evaluation.utils import apply_symmetry


class h5_dataset:
    def __init__(self, bundle_paths, tar_fourier):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        if bundle_paths == []:
            raise ValueError("No bundles found! Please check the names of your files.")
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))
        self.tar_fourier = tar_fourier

    def __call__(self):
        return print("This is the h5_dataset class.")

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def __getitem__(self, i):
        x = self.open_image("x", i)

        sym_x = apply_symmetry({"x": x}, overlap=1)["x"]
        di = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(sym_x[0] + 1j * sym_x[1])))
        )
        di_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(di)))

        x = torch.stack([torch.abs(di_fft), torch.angle(di_fft)])
        x[0] **= 0.1
        del sym_x, di, di_fft

        half_image = x.shape[2] // 2
        x = x[:, half_image + 4, :]

        y = self.open_image("y", i)
        sym_y = apply_symmetry({"y": y}, overlap=1)["y"]
        y_di = np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(sym_y[0] + 1j * sym_y[1])))
        )
        y_di_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(y_di)))

        y = torch.stack([torch.abs(y_di_fft), torch.angle(y_di_fft)])
        y[0] **= 0.1
        del sym_y, y_di, y_di_fft

        half_image = y.shape[2] // 2
        y = y[:, half_image + 4, :]

        return x, y

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def open_image(self, var, i):
        if isinstance(i, int):
            i = torch.tensor([i])
        elif isinstance(i, np.ndarray):
            i = torch.tensor(i)
        indices, _ = torch.sort(i)
        bundle = torch.div(indices, self.num_img, rounding_mode="floor")
        image = indices - bundle * self.num_img
        bundle_unique = torch.unique(bundle)
        bundle_paths = [
            h5py.File(self.bundles[bundle], "r") for bundle in bundle_unique
        ]
        bundle_paths_str = list(map(str, bundle_paths))
        data = torch.tensor(
            np.array(
                [
                    bund[var][img]
                    for bund, bund_str in zip(bundle_paths, bundle_paths_str)
                    for img in image[
                        bundle == bundle_unique[bundle_paths_str.index(bund_str)]
                    ]
                ]
            )
        )
        if self.tar_fourier is False and data.shape[1] == 2:
            raise ValueError(
                "Two channeled data is used despite Fourier being False.\
                    Set Fourier to True!"
            )

        if data.shape[0] == 1:
            data = data.squeeze(0)

        return data.float()


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


def get_dls(train_ds, valid_ds, batch_size, **kwargs):
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=batch_size, shuffle=True, **kwargs),
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
    with h5py.File(str(path) + str(counter) + ".h5", "w") as hf:
        hf.create_dataset(name, data=bundle)
        hf.close()


def open_bundle(path):
    """
    open radio galaxy bundles created in first analysis step
    """
    f = h5py.File(path, "r")
    bundle = np.array(f["gs_bundle"])
    return bundle


def open_fft_bundle(path):
    """
    open radio galaxy bundles created in first analysis step
    """
    f = h5py.File(path, "r")
    x = np.array(f["x"])
    y = np.array(f["y"])
    return x, y


def get_bundles(path):
    """
    returns list of bundle paths located in a directory
    """
    data_path = Path(path)
    bundles = np.array(data_path.glob("samp_*.h5"))
    return bundles


def save_fft_pair(path, x, y, z=None, name_x="x", name_y="y", name_z="z"):
    """
    write fft_pairs created in second analysis step to h5 file
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset(name_x, data=x)
        hf.create_dataset(name_y, data=y)
        if z is not None:
            [hf.create_dataset(name_z + str(i), data=z[i]) for i in range(len(z))]
        hf.close()


def open_bundle_pack(path):
    bundle_x = []
    bundle_y = []
    bundle_z = []
    f = h5py.File(path, "r")
    bundle_size = len(f) // 3
    for i in range(bundle_size):
        bundle_x_i = np.array(f["x" + str(i)])
        bundle_x.append(bundle_x_i)
        bundle_y_i = np.array(f["y" + str(i)])
        bundle_y.append(bundle_y_i)
        bundle_z_i = np.array(f["z" + str(i)])
        bundle_z.append(bundle_z_i)
    f.close()
    return np.array(bundle_x), np.array(bundle_y), bundle_z


def load_data(data_path, mode, fourier=False):
    """
    Load data set from a directory and return it as h5_dataset.

    Parameters
    ----------
    data_path: str
        path to data directory
    mode: str
        specify data set type, e.g. test
    fourier: bool
        use Fourier images as target if True, default is False

    Returns
    -------
    test_ds: h5_dataset
        dataset containing x and y images
    """
    data_path = Path(data_path)
    bundle_paths = data_path.glob("samp_*" + mode + "*.h5")
    data = sorted(bundle_paths, key=lambda f: int("".join(filter(str.isdigit, str(f)))))
    ds = h5_dataset(data, tar_fourier=fourier)
    return ds
