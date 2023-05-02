import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import DataLoader


class h5_dataset:
    def __init__(self, bundle_paths, tar_fourier, amp_phase=None, source_list=False):
        """
        Save the bundle paths and the number of bundles in one file.
        """
        if bundle_paths == []:
            raise ValueError("No bundles found! Please check the names of your files.")
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0], "x"))
        self.tar_fourier = tar_fourier
        self.amp_phase = amp_phase
        self.source_list = source_list

    def __call__(self):
        return print("This is the h5_dataset class.")

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
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

        if not self.source_list:
            data = data[:, :65, :]

        return data.float()


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


# open and save functions should be generalized in future versions


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
    bundles = np.array([x for x in data_path.iterdir()])
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


def load_data(data_path, mode, fourier=False, source_list=False):
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
    bundle_paths = get_bundles(data_path)
    data = np.sort(
        [path for path in bundle_paths if re.findall("samp_" + mode, path.name)]
    )
    data = sorted(data, key=lambda f: int("".join(filter(str.isdigit, str(f)))))
    ds = h5_dataset(data, tar_fourier=fourier, source_list=source_list)
    return ds


class MojaveDataset:
    def __init__(
        self,
        data_path: str,
        source: str = None,
        crop_size: int = None,
        scaling: Callable = None,
    ):
        """Dataset class for MOJAVE data.

        Parameters
        ----------
        data_path: str
            path to folder containing the files
        source: str
            select images of one certain source
        crop_size: int
            crop image around center to this size
        scaling: function
            function to be applied for scaling
        """
        self.data_path = data_path
        self.source = source
        self.paths = self._get_path()
        self.source_list = self._get_source_list()
        self.crop_size = crop_size
        self.scaling = scaling

    def __call__(self):
        return print("This is the MojaveDataset class.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.open_image(i)

    def _get_path(self):
        """Get all paths of files in folder. Filtered by source, if provided.

        Returns
        -------
        data_files: list
            folder and filename of each file
        """
        if self.source:
            data_path = self.data_path + self.source
            file_paths = [x for x in Path(data_path).glob("*/*")]
        else:
            data_path = self.data_path
            file_paths = [x for x in Path(data_path).glob("*/*/*")]

        data_files = [
            path for path in file_paths if re.findall(".icn.fits.gz", path.name)
        ]

        print("Number of data files:", len(data_files))
        # print(sorted(data_files)[:5])

        assert data_files, f"No source found in {data_path}"

        return sorted(data_files)

    def _get_source_list(self):
        """Create list of all available source names.

        Returns
        -------
        source_list: list
            list with all source names
        """
        if self.source:
            source_list = [self.source]
        else:
            source_list = sorted(next(os.walk(self.data_path))[1])
            source_list = [elem for elem in source_list if elem[:4].isnumeric()]

        for source in reversed(source_list):
            data_path = self.data_path + source
            file_paths = [x for x in Path(data_path).glob("*/*")]
            data_files = [
                path for path in file_paths if re.findall(".icn.fits.gz", path.name)
            ]
            if not data_files:
                # print("No images for source:", source)
                source_list.remove(source)

        print("Number of sources:", len(source_list))
        return source_list

    def get_header(self, i: int = 0, source_name: str = None):
        """Opening i-th header of fits files.

        Parameters
        ----------
        i: int
            index of image
        source_name: str
            name of the source

        Returns
        -------
        header:
            opened header
        """
        if source_name:
            paths = []
            for path in self.paths:
                if source_name in str(path):
                    paths.append(path)
        else:
            paths = self.paths

        f = fits.open(paths[i])
        header = f[0].header

        try:
            date = datetime.strptime(header["DATE-OBS"], "%d/%m/%y")
            header["DATE-OBS"] = date.isoformat()[:10]
        except ValueError:
            pass

        return header

    def open_image(self, i: int = 0, justify: bool = False):
        """Opening i-th image of fits files.

        Parameters
        ----------
        i: int
            index of image
        justify: bool
            justify the cropping

        Returns
        -------
        img: 2d-array
            opened image
        """
        warnings.filterwarnings("ignore", module="astropy.io.fits")
        f = fits.open(self.paths[i])
        img = f[0].data.astype(np.float32).squeeze()

        if self.crop_size:
            assert (
                img.shape[0] >= self.crop_size
            ), "Image is smaller than the crop_size. Reduce crop_size."
            img = crop_center(
                img, cropx=self.crop_size, cropy=self.crop_size, justify=justify
            )

        if self.scaling:
            img = self.scaling(img)

        return img

    def open_source(self, source_name: str):
        """Opening all images of one source.

        Parameters
        ----------
        source_name: str
            name of the source

        Returns
        -------
        images: 3d-array
            all images of one source
        """
        assert (
            source_name in self.source_list
        ), f"Source name {source_name} not in source_list."

        images = []
        for i, path in enumerate(self.paths):
            if source_name in str(path):
                images.append(self.open_image(i))

        return np.array(images)


def crop_center(img, cropx: int, cropy: int, justify: bool = False, eps: float = 1e-2):
    """Crop out the center of an image

    Parameters
    ----------
    img: 2d-array
        image to be cropped
    cropx: int
        output size of x-axis
    cropy: int
        output size of y-axis
    justify: bool
        wether to justfiy or not, used when cropping MOJAVE data
    eps: float
        threshold for warning
    """
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    img_cropped = img[starty : starty + cropy, startx : startx + cropx]

    if justify:
        img_outer = img.copy()
        # set cropped area to -1, so it does not appear in top values
        img_outer[starty : starty + cropy, startx : startx + cropx] = -1

        # calculate max n values and compare
        n = int(np.sqrt(cropx))
        top_outer = np.partition(img_outer, -n, axis=None)[-n:]
        top_inner = np.partition(img_cropped, -n, axis=None)[-n:]

        mean_ratio = top_outer.mean() / top_inner.mean()
        if mean_ratio > eps:
            print(f"Source might be cut off. Outer to Inner ratio: {mean_ratio:.2f}")

    return img_cropped
