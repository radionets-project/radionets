from datetime import datetime
from math import pi

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# from matplotlib.colors import LogNorm
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.scan import RIME
from scipy.constants import c
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog
from torch import fft
from torch.nn.functional import max_pool2d_with_indices
from torch.utils.data import Dataset
from torchvision.transforms import Resize


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, num_classes):
        super(Generator, self).__init__()

        self.conv_noise = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.conv_label = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        self.model = nn.Sequential(
            # # input is Z, going into a convolution
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        )

    def forward(self, z, label):
        x = self.conv_noise(z)
        y = self.conv_label(label)
        combined_input = torch.cat([x, y], 1)
        return torch.clamp(self.model(combined_input), -1, 1)


class SourceGenerator:
    def __init__(self, image_size, point_sources=False):
        self.metadata = {
            "ngpu": 1,
            # this is fixed by the training setup and thus by the weights
            "num_classes": 4,
            # Size of images final image dimensions including nc as first input
            "image_shape": [1, 128, 128],
            # Size of z latent vector (i.e. size of generator input)
            "nz": 100,
            # Size of feature maps in generator
            "ngf": 64,
        }
        self.device = torch.device(
            "cuda:0"
            if (torch.cuda.is_available() and self.metadata["ngpu"] > 0)
            else "cpu"
        )
        self.generator_extended = self.create_generator(
            "/home/schmidt3/paper/meerkat/nbs/wgan_saves/\
            generator_epoch_3361_iter_30250_cla0.pt",
        )
        self.generator_compact = self.create_generator(
            "/home/schmidt3/paper/meerkat/nbs/wgan_saves/\
            generator_epoch_3833_iter_34500_cla1.pt"
        )
        self.resizer_24 = Resize(24, antialias=True)
        self.resizer_32 = Resize(32, antialias=True)
        self.img_size = image_size
        self.point_sources = point_sources

    def create_generator(self, path):
        generator = Generator(
            self.metadata["nz"],
            self.metadata["image_shape"][0],
            self.metadata["ngf"],
            self.metadata["num_classes"],
        )

        generator.load_state_dict(
            torch.load(
                path,
                map_location=torch.device("cpu"),
            )
        )
        return generator

    def get_generated_images(self, generator, n_gen_images, nz, device, label_ind):
        tensor_opt = {"dtype": torch.float, "requires_grad": False}
        onehot = torch.zeros(4, 4, **tensor_opt)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1), 1).view(
            4, 4, 1, 1
        )

        noise = torch.randn(n_gen_images, nz, 1, 1, requires_grad=False)
        labels = torch.tensor([label_ind] * n_gen_images, requires_grad=False)
        with torch.no_grad():
            generated_images = (
                ((generator(noise, onehot[labels]) / 2 + 0.5) * 255).int().cpu()
            )
        a = generated_images.squeeze(1)
        while a.sum(axis=1).sum(axis=1).all() == 0:
            print("nochmal")
            noise = torch.randn(n_gen_images, nz, 1, 1, requires_grad=False)
            labels = torch.tensor([label_ind] * n_gen_images, requires_grad=False)
            with torch.no_grad():
                generated_images = (
                    ((generator(noise, onehot[labels]) / 2 + 0.5) * 255).int().cpu()
                )
            a = generated_images.squeeze(1)
            print("return")
        return generated_images.squeeze(1)

    def generate_extended(self, generator):
        extended = self.resizer_32(
            self.get_generated_images(generator, 1, self.metadata["nz"], self.device, 0)
        ).numpy()
        extended = gaussian_filter(extended, sigma=0.75)

        norm = extended.max(axis=1).max(axis=1).reshape(-1, 1, 1)
        extended = np.divide(extended, norm, where=norm != 0) * np.random.normal(
            1, 0.25, size=(extended.shape[0], 1, 1)
        )
        if extended.max() > 0:
            extended /= extended.max()
        extended *= np.abs(np.random.normal(0.001, 0.25))
        extended[np.isnan(extended)] = 1e-8
        return extended

    def generate_compact(self, generator):
        compact = self.resizer_24(
            self.get_generated_images(generator, 1, self.metadata["nz"], self.device, 2)
        ).numpy()
        compact = gaussian_filter(compact, sigma=0.75)

        norm = compact.max(axis=1).max(axis=1).reshape(-1, 1, 1)
        compact = np.divide(compact, norm, where=norm != 0) * np.random.normal(
            1, 0.2, size=(compact.shape[0], 1, 1)
        )
        if compact.max() > 0:
            compact /= compact.max()
        compact *= np.abs(np.random.normal(0.001, 0.25))
        compact[np.isnan(compact)] = 1e-8
        return compact

    def get_chunk(self, size=128, edge_limit=32):
        limit = self.img_size - size - edge_limit
        low_x = np.random.randint(edge_limit, limit)
        high_x = low_x + size
        low_y = np.random.randint(edge_limit, limit)
        high_y = low_y + size
        x, y = np.meshgrid(np.arange(low_x, high_x), np.arange(low_y, high_y))
        return x, y

    def create_sky(
        self,
    ):
        sky_point = np.zeros((self.img_size, self.img_size))
        if self.point_sources:
            sky_point = np.random.choice(
                [True, False], size=self.img_size**2, p=[0.0001, 0.9999]
            ).reshape(self.img_size, self.img_size) * np.random.normal(
                0.0000001, 2, size=(self.img_size, self.img_size)
            )
            sky_point /= sky_point.max()
            sky_point *= 5e-3
            sky_point[sky_point <= 0] = 1e-8

        for i in range(np.random.randint(5, 10)):  # 10 15
            sky_point[self.get_chunk(24)] += self.generate_compact(
                self.generator_compact
            )[0]

        for i in range(np.random.randint(3, 5)):  # 5 10
            sky_point[self.get_chunk(32)] += self.generate_extended(
                self.generator_extended
            )[0]

        sky_point[
            int(self.img_size // 2 - 16) : int(self.img_size // 2 + 16),
            int(self.img_size // 2 - 16) : int(self.img_size // 2 + 16),
        ] += self.generate_extended(self.generator_extended)[0]

        sky_point = gaussian_filter(sky_point, sigma=0.5)
        sky_point /= 5
        sky_point[np.isnan(sky_point)] = 1e-8
        return sky_point


class VisGenerator:
    def __init__(self, obs, img):
        self.obs = obs
        self.bas = obs.baselines.get_valid_subset(
            obs.num_baselines,
            obs.device,
        ).get_unique_grid(
            obs.fov,
            obs.ref_frequency,
            obs.img_size,
            obs.device,
        )
        self.img = self.prepare_img(img)
        self.B = self.create_B()

    def prepare_img(self, img):
        if type(img) == np.ndarray:
            img = torch.from_numpy(img)
        else:
            img = img
        if len(img.shape) == 2:
            img = img.unsqueeze(-1)
        return img

    def create_B(self):
        stokes = torch.zeros(
            (self.img.shape[0], self.img.shape[1], 4), dtype=torch.cdouble
        )
        stokes[..., 0] = self.img[..., 0]

        B = torch.zeros(
            (self.img.shape[0], self.img.shape[1], 2, 2), dtype=torch.cdouble
        )

        B[:, :, 0, 0] = stokes[:, :, 0] + stokes[:, :, 1]
        B[:, :, 0, 1] = stokes[:, :, 2] + 1j * stokes[:, :, 3]
        B[:, :, 1, 0] = stokes[:, :, 2] + 1j * stokes[:, :, 3]
        B[:, :, 1, 1] = stokes[:, :, 0] - stokes[:, :, 1]
        return B

    def calc_vis(self):
        mask = (self.img >= self.obs.sensitivity_cut)[..., 0]
        B = self.B[mask]
        lm = self.obs.lm[mask]
        rd = self.obs.rd[mask]
        int_vals = torch.cat(
            [
                RIME(
                    self.bas[:][:, p],
                    lm,
                    rd,
                    self.obs.ra,
                    self.obs.dec,
                    torch.unique(self.obs.array.diam),
                    self.obs.waves_low[0],
                    self.obs.waves_high[0],
                    corrupted=self.obs.corrupted,
                )(B)
                for p in torch.arange(self.bas[:].shape[1]).split(1000)
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
                torch.cat([-self.bas[2].cpu(), self.bas[2].cpu()]),
                torch.cat([-self.bas[5].cpu(), self.bas[5].cpu()]),
            ),
            dim=-1,
        )
        return uv_coords, vis_spares


class Gridder:
    def __init__(self, uv, vis, freq, fov, img_size):
        self.freq = freq
        self._create_attributes(uv, vis)
        self.fov = fov
        self.img_size = img_size
        self.bins = self.create_bins()
        self.grid_data()
        self.create_di()

    def _create_attributes(self, uv, vis):
        self.u_scaled = uv[:, 0] * self.freq / c
        self.v_scaled = uv[:, 1] * self.freq / c

        self.u = uv[:, 0]
        self.v = uv[:, 1]

        self.samps = torch.cat(
            [
                self.u_scaled[None],
                self.v_scaled[None],
                vis.real[None],
                vis.imag[None],
            ],
            dim=0,
        )

    def create_bins(self):
        N = self.img_size
        fov = self.fov * pi / (3600 * 180)
        delta = (fov) ** (-1)

        bins = (
            torch.arange(start=-(N / 2) * delta, end=(N / 2 + 1) * delta, step=delta)
            - delta / 2
        ).double()
        return bins

    def grid_data(self):
        mask, *_ = torch.histogramdd(
            self.samps[:2].swapaxes(0, 1), bins=[self.bins, self.bins], density=False
        )
        mask[mask == 0] = 1

        mask_real, *_ = torch.histogramdd(
            self.samps[:2].swapaxes(0, 1),
            bins=[self.bins, self.bins],
            weight=self.samps[2],
            density=False,
        )
        mask_imag, *_ = torch.histogramdd(
            self.samps[:2].swapaxes(0, 1),
            bins=[self.bins, self.bins],
            weight=self.samps[3],
            density=False,
        )
        mask_real /= mask
        mask_imag /= mask

        self.mask = mask
        self.mask_real = mask_real
        self.mask_imag = mask_imag
        self.real = mask_real
        self.imag = mask_imag
        return mask, mask_real, mask_imag

    def create_di(self):
        self.dirty_img_cmplx = fft.fftshift(
            fft.ifft2(fft.fftshift(self.mask_real + 1j * self.mask_imag))
        )
        self.dirty_img = torch.abs(self.dirty_img_cmplx).flip(-1)


class VisManipulator:
    def __init__(self, vals, SI=True, n=5):
        if SI:
            if type(vals) == np.ndarray:
                self.img = torch.from_numpy(vals)
            else:
                self.img = vals
            self.vis_full = self.calc_fft(self.img)
            self.mask = None
        else:
            self.vis_full = (vals.real + 1j * vals.imag).flip(0)
            self.img = torch.abs(self.calc_ifft(self.vis_full))
            self.mask = self.vis_full.real
            self.vis_full = self.apply_mask(self.calc_fft(self.img), self.mask)
        self.n = n

    def substract_source(self, di=False):
        # get chunk of brightest source and calc partial fft
        vis_chunks = self.calc_partial_vis(self.img, mask=self.mask, n=self.n)
        # substract partial vis from full vis
        vis_sub = self.substract_vis(self.vis_full, vis_chunks)
        if di:
            return self.calc_ifft(vis_sub)
        else:
            return vis_sub.unsqueeze(0), torch.stack(vis_chunks)

    def substract_vis(self, fft_full, fft_chunks):
        v_real = torch.stack([v.real for v in fft_chunks]).sum(0)
        v_imag = torch.stack([v.imag for v in fft_chunks]).sum(0)
        sub_real = fft_full.real - v_real
        sub_imag = fft_full.imag - v_imag
        return sub_real + 1j * sub_imag

    def calc_partial_vis(self, img, mask=None, n=5):
        x_mid, y_mid = self.get_n_max(img, n=n)
        chunks = []
        for x, y in zip(x_mid, y_mid):
            chunks.append(self.get_chunk(img, x, y))

        vis_chunks = []
        for chunk in chunks:
            vis_chunks.append(self.calc_fft(chunk))
        if mask is not None:
            vis_chunks_masked = []
            for vis_chunk in vis_chunks:
                vis_chunks_masked.append(self.apply_mask(vis_chunk, mask))
            return vis_chunks_masked
        else:
            return vis_chunks

    def get_n_max(self, img, n=5):
        window_maxima = max_pool2d_with_indices(
            img.unsqueeze(0), 10, 1, padding=10 // 2
        )[0].squeeze()
        blobs = blob_dog(window_maxima, threshold=0.0001)[:n]
        return blobs[:, 0], blobs[:, 1]

    def get_chunk(self, img, x_mid, y_mid, size=32):
        low_x = int(x_mid - size // 2)
        high_x = low_x + size
        low_y = int(y_mid - size // 2)
        high_y = low_y + size
        mask = torch.zeros(img.shape).bool()
        mask[low_x:high_x, low_y:high_y] = True
        chunk = img.clone()
        chunk[~mask] = chunk.min()
        return chunk

    def calc_fft(self, img):
        return fft.fftshift(fft.fft2(fft.fftshift(img))).flip(-1)

    def calc_ifft(self, img):
        return fft.fftshift(fft.ifft2(fft.fftshift(img))).flip(-1)

    def apply_mask(self, vis, mask):
        masked = vis.clone()
        m = torch.zeros(vis.shape).bool()
        m[mask != 0] = True
        masked[~m] = 0
        return masked


class PartialData(Dataset):
    def __init__(self, img_size=1024, point_sources=True, len=10000):
        self.len = len
        self.img_size = img_size
        self.point = point_sources
        self.obs = Observation(
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
            fov=3072,
            image_size=self.img_size,
            array_layout="meerkat",
            corrupted=False,
            device="cpu",
            dense=False,
            sensitivity_cut=1e-6,
        )

    def __call__(self):
        print("This is the partial visibility data set class.")

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        x, y = self.create_data()
        half_image = x.shape[2] // 2
        x = x[:, : half_image + 64, :]
        y = y[:, : half_image + 64, :]
        return x, y

    def create_data(self):
        source = SourceGenerator(self.img_size, point_sources=self.point).create_sky()
        uv_coords, vis_sparse = VisGenerator(self.obs, source).calc_vis()
        gridder = Gridder(
            uv_coords, vis_sparse, self.obs.ref_frequency, self.obs.fov, self.img_size
        )

        vis_compl = VisManipulator(source, n=5)
        vis_compl_sub, vis_compl_chunks = vis_compl.substract_source(di=False)
        y = torch.cat([vis_compl_chunks, vis_compl_sub])
        y = torch.cat([y.real, y.imag], dim=0)

        vis_samp = VisManipulator(gridder, SI=False, n=5)
        vis_samp_sub, vis_samp_chunks = vis_samp.substract_source(di=False)
        x = torch.cat([vis_samp_chunks, vis_samp_sub])
        x = torch.cat([x.real, x.imag], dim=0)
        return x.float(), y.float()


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
    ds = PartialData()
    return ds
