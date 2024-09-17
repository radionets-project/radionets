import re
from datetime import datetime

import h5py
import numpy as np
import torch
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.scan import RIME
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from radionets.dl_framework.data import get_bundles


def scale_uv(uv, max_val=1):
    uv /= max_val
    uv *= 0.5
    return uv


class VisDataset(Dataset):
    def __init__(self, bundle_paths, obs):
        super().__init__()
        self.obs = obs
        self.bundles = bundle_paths
        self.num_img = len(self.open_bundle(self.bundles[0], "y"))
        self.tar_fourier = True

    def __len__(self):
        """
        Returns the total number of pictures in this dataset
        """
        return len(self.bundles) * self.num_img

    def open_bundle(self, bundle_path, var):
        bundle = h5py.File(bundle_path, "r")
        data = bundle[var]
        return data

    def __getitem__(self, i):
        img = Resize(64)(self.open_image("y", i)[None])[0]
        with torch.no_grad():
            uv_coords, vis_sparse = self._prepare_vis(img, self.obs)
        vis_sparse = torch.stack((vis_sparse.real, vis_sparse.imag), dim=-1)
        self.obs.calc_dense_baselines()
        bas_dense = self.obs.dense_baselines_gpu
        uv_dense = torch.stack((bas_dense[2].cpu(), bas_dense[5].cpu()), dim=-1)
        visibilities = self._get_cmpl_true(img, uv_dense)
        return (
            scale_uv(uv_coords.float(), max_val=uv_dense.max()),
            scale_uv(uv_dense.float(), max_val=uv_dense.max()),
            vis_sparse.float(),
            visibilities.flatten(),
            img,
        ), (
            scale_uv(uv_coords.float(), max_val=uv_dense.max()),
            scale_uv(uv_dense.float(), max_val=uv_dense.max()),
            vis_sparse.float(),
            visibilities.flatten(),
            img,
        )

    def _prepare_vis(self, img, obs):
        bas = obs.baselines.get_valid_subset(
            obs.num_baselines,
            obs.device,
        ).get_unique_grid(obs.fov, obs.ref_frequency, obs.img_size, obs.device)

        img = img.unsqueeze(-1)
        stokes = torch.zeros((img.shape[0], img.shape[1], 4), dtype=torch.cdouble)
        stokes[..., 0] = img[..., 0]

        B = torch.zeros((img.shape[0], img.shape[1], 2, 2), dtype=torch.cdouble)  # .to(
        #    torch.device(obs.device)
        # )

        B[:, :, 0, 0] = stokes[:, :, 0] + stokes[:, :, 1]
        B[:, :, 0, 1] = stokes[:, :, 2] + 1j * stokes[:, :, 3]
        B[:, :, 1, 0] = stokes[:, :, 2] + 1j * stokes[:, :, 3]
        B[:, :, 1, 1] = stokes[:, :, 0] - stokes[:, :, 1]

        mask = (img >= obs.sensitivity_cut)[..., 0]
        B = B[mask]
        lm = obs.lm[mask]
        rd = obs.rd[mask]
        int_vals = torch.cat(
            [
                RIME(
                    bas[:][:, p],
                    lm,
                    rd,
                    obs.ra,
                    obs.dec,
                    torch.unique(obs.array.diam),
                    obs.waves_low[0],
                    obs.waves_high[0],
                    corrupted=obs.corrupted,
                )(B)
                for p in torch.arange(bas[:].shape[1]).split(3000)
            ]
        )
        vis_spares = torch.cat(
            [
                0.5 * (int_vals[:, 0, 0] + int_vals[:, 1, 1]),
                0.5 * (int_vals[:, 0, 0].conj() + int_vals[:, 1, 1].conj()),
            ]
        )
        uv_coords = torch.stack(
            (
                torch.cat([bas[2].cpu(), -bas[2].cpu()]),
                torch.cat([bas[5].cpu(), -bas[5].cpu()]),
            ),
            dim=-1,
        )
        return uv_coords, vis_spares

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
        data = torch.from_numpy(
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

    def _get_cmpl_true(self, img, uv_dense):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(img)))


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
    bundle_paths = get_bundles(data_path)
    data = np.sort(
        [path for path in bundle_paths if re.findall("skies_" + mode, path.name)]
    )
    data = sorted(data, key=lambda f: int("".join(filter(str.isdigit, str(f)))))

    obs = Observation(
        src_ra=186.12,
        src_dec=6.47,
        start_time=datetime(2023, 4, 23, 17, 0, 0),
        scan_duration=300,
        scan_separation=1200,
        num_scans=2,
        integration_time=15,
        ref_frequency=1.3e9,
        frequency_offsets=[0],
        bandwidths=[1.67e6],
        fov=1536 / 4,
        image_size=64,
        array_layout="meerkat",
        corrupted=False,
        device="cuda:0",
        dense=False,
        sensitivity_cut=1e-6,
    )

    ds = VisDataset(data, obs)
    return ds
