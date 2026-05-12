from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.feature import blob_log
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

__all__ = [
    "IntensityRatio",
    "SourceAreaRatio",
    "ViewingAngle",
    "MeanDifference",
    "DynamicRange",
]


class IntensityRatio(Metric):
    """Computes the integrated flux and peak flux intensity ratios
    between prediction and ground truth images.

    Ratios are calculated at a threshold of the maximum true flux (default 5%).

    Parameters
    ----------
    threshold : float, optional
        Threshold at which to compute the metrics. Default: 0.05
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    sum_ratios: list[torch.Tensor]
    peak_ratios: list[torch.Tensor]

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()

        self.add_state("sum_ratios", default=[], dist_reduce_fx="cat")
        self.add_state("peak_ratios", default=[], dist_reduce_fx="cat")

        self.threshold = threshold

    def update(self, ifft_preds: torch.Tensor, ifft_targets: torch.Tensor) -> None:
        """Update state with a new batch of predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Prediction image(s), shape (B, H, W).
        targets : torch.Tensor
            Ground truth image(s), shape (B, H, W).
        """
        if ifft_preds.ndim == 2:
            ifft_preds = ifft_preds.unsqueeze(0)
        if ifft_targets.ndim == 2:
            ifft_targets = ifft_targets.unsqueeze(0)

        threshold = ifft_targets.amax(dim=(-2, -1), keepdim=True) * self.threshold

        source_target = torch.where(
            ifft_targets > threshold,
            ifft_targets,
            torch.zeros_like(ifft_targets),
        )
        source_pred = torch.where(
            ifft_preds > threshold,
            ifft_preds,
            torch.zeros_like(ifft_preds),
        )

        sum_ratio = source_pred.sum(dim=(-2, -1)) / source_target.sum(dim=(-2, -1))
        peak_ratio = source_pred.amax(dim=(-2, -1)) / source_target.amax(dim=(-2, -1))

        self.sum_ratios.append(sum_ratio)
        self.peak_ratios.append(peak_ratio)

    def compute(self) -> dict[str, torch.Tensor]:
        """Concatenate the integrated flux ratios and peak flux ratios
        across all batches.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing both integrated flux ratios ('integrated_flux')
            and peak flux ratios ('peak_flux') for the entire evaluated dataset.
        """
        return {
            "integrated_flux": dim_zero_cat(self.sum_ratios),
            "peak_flux": dim_zero_cat(self.peak_ratios),
        }


class SourceAreaRatio(Metric):
    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    area_ratios: list[torch.Tensor]

    def __init__(
        self,
        threshold: float = 0.05,
        mode: Literal["pixel", "contour"] = "pixel",
    ) -> None:
        super().__init__()

        self.add_state("area_ratios", default=[], dist_reduce_fx="cat")

        self.threshold = threshold

        match mode:
            case "pixel":
                self.get_area_ratios: Callable = self.__pixel_area
            case "contour":
                self.get_area_ratios: Callable = self.__contour_area
            case _:
                raise ValueError(
                    f"Unknown mode '{mode}'. Please use one of 'pixel' or 'contour'."
                )

    def update(self, ifft_preds: torch.Tensor, ifft_targets: torch.Tensor) -> None:
        """Compute area ratio at 5% of the maximum of prediction and target.

        Parameters
        ----------
        ifft_pred : ndarray
            Predicted source image(s).
        ifft_target : ndarray
            Target source image(s).
        level : float, optional
            Percentile level of maximum (true) flux computed on the target.
            Default: 0.05

        Returns
        -------
        list[float] | float
            Ratio of predicted and targeted source areas. If batch size is 1,
            returns ratio as float.
        """
        if ifft_preds.ndim == 2:
            ifft_preds = ifft_preds.unsqueeze(0)
        if ifft_targets.ndim == 2:
            ifft_targets = ifft_targets.unsqueeze(0)

        thresholds = ifft_targets.amax(dim=[-2, -1]) * self.threshold

        area_preds, area_targets = self.get_area_ratios(
            ifft_preds, ifft_targets, thresholds
        )

        self.area_ratios.append(area_preds.sum() / area_targets.sum())

    def __contour_area(
        self,
        ifft_preds: torch.Tensor,
        ifft_targets: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds_np = ifft_preds.detach().cpu().numpy()
        targets_np = ifft_targets.detach().cpu().numpy()

        area_preds = []
        area_targets = []

        # loop over images in batch
        for p, t in zip(preds_np, targets_np):
            level_val = t.max() * self.level

            fig, ax = plt.subplots()
            cs_pred = ax.contour(p, levels=[level_val])
            cs_target = ax.contour(t, levels=[level_val])
            plt.close(fig)

            area_preds.append(
                sum(
                    self.__compute_source_area(path.vertices)
                    for path in cs_pred.get_paths()
                )
            )
            area_targets.append(
                sum(
                    self.__compute_source_area(path.vertices)
                    for path in cs_target.get_paths()
                )
            )

        return torch.as_tensor(area_preds), torch.as_tensor(area_targets)

    def __compute_source_area(self, vertices: np.typing.ArrayLike) -> float:
        """Helper function to compute area of a source
        using the shoelace formula.

        Parameters
        ----------
        vertices : :func:`~numpy.ndarray`, shape (N, 2)
            Polygon (source) vertices as (x, y) coordinates.

        Returns
        -------
        float
            Area of the source.
        """
        x = vertices[:, 0]
        y = vertices[:, 1]

        s1 = np.dot(x, np.roll(y, -1))
        s2 = np.dot(y, np.roll(x, -1))

        return 0.5 * np.abs(s1 - s2)

    def __pixel_area(
        self,
        ifft_preds: torch.Tensor,
        ifft_targets: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        area_preds = (ifft_preds > thresholds[:, None, None]).sum(dim=(-2, -1))
        area_targets = (ifft_targets > thresholds[:, None, None]).sum(dim=(-2, -1))

        return area_preds, area_targets

    def compute(self) -> dict[str, torch.Tensor]:
        """Concatenate the integrated flux ratios and peak flux ratios
        across all batches.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing both integrated flux ratios ('integrated_flux')
            and peak flux ratios ('peak_flux') for the entire evaluated dataset.
        """
        return {
            "source_area": dim_zero_cat(self.area_ratios),
        }


class DynamicRange(Metric):
    """Computes the integrated flux and peak flux intensity ratios
    between prediction and ground truth images.

    Ratios are calculated at a threshold of the maximum true flux (default 5%).

    Parameters
    ----------
    threshold : float, optional
        Threshold at which to compute the metrics. Default: 0.05
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    dr_preds: list[torch.Tensor]
    dr_targets: list[torch.Tensor]

    __BOX_FACTORS: torch.Tensor = torch.tensor([0.3, 0.22, 0.16])

    def __init__(self, sensitivity: float = 1e-6) -> None:
        super().__init__()

        self.add_state("dr_preds", default=[], dist_reduce_fx="cat")
        self.add_state("dr_targets", default=[], dist_reduce_fx="cat")

        self.sensitivity = sensitivity

    def update(self, ifft_preds: torch.Tensor, ifft_targets: torch.Tensor) -> None:
        """
        Calculate dynamic range for ground truth and predicted images.

        The dynamic range is the peak value divided by RMS
        noise in corner (off-)regions (i.e., where no signal is expected).

        Parameters
        ----------
        ifft_pred : :func:`~numpy.ndarray`
            Predicted inverse FFT images (image space), shape (B, H, W).
        ifft_target : :func:`~numpy.ndarray`
            Ground truth inverse FFT images (image space), shape (B, H, W).

        Returns
        -------
        dr_target : :func:`~numpy.ndarray`
            Dynamic range for target.
        dr_preds : :func:`~numpy.ndarray`
            Dynamic range for predictions.
        rms_boxes : np. ndarray
            Number of valid corners per sample.
        corners : :func:`~numpy.ndarray`
            Corner validity mask.
        """
        rms_target, rms_pred = self.__rms(ifft_targets, ifft_preds)

        peak_target = ifft_targets.reshape(len(ifft_targets), -1).amax(dim=1)
        peak_pred = ifft_preds.reshape(len(ifft_preds), -1).amax(dim=1)

        valid_target = rms_target != 0
        valid_pred = rms_pred != 0

        dr_target = peak_target[valid_target] / rms_target[valid_target]
        dr_pred = peak_pred[valid_pred] / rms_pred[valid_pred]

        self.dr_preds.append(dr_pred)
        self.dr_targets.append(dr_target)

    def __rms(self, ifft_truth, ifft_pred) -> tuple:
        """
        Compute RMS values for ground truth and prediction.

        Parameters
        ----------
        ifft_truth : :func:`numpy.ndarray`, shape (B, H, W)
            Ground truth images.
        ifft_pred : :func:`numpy.ndarray`, shape (B, H, W)
            Predicted images.
        sensitivity : float, optional
            Threshold below which corners are considered valid.
            Default: 1e-6.

        Returns
        -------
        rms_truth : :func:`~numpy.ndarray`, shape (B,)
            Averaged RMS for ground truth.
        rms_pred : :func:`~numpy.ndarray`, shape (B,)
            Averaged RMS for predictions.
        """
        _rms_truth_boxes = {}
        _rms_pred_boxes = {}

        for num_corners in [4, 3, 2]:
            size = self.__boxsize(num_corners)
            _rms_truth_boxes[num_corners] = self.__compute_rms(ifft_truth, size)
            _rms_pred_boxes[num_corners] = self.__compute_rms(ifft_pred, size)

        print([b.shape for b in _rms_pred_boxes.values()])

        rms_boxes = self.__select_box(_rms_truth_boxes[4], sensitivity=self.sensitivity)
        current_batch_size = len(ifft_pred)

        # TODO: May need to return this later for plots
        # corners = (_rms_truth_boxes[4] <= self.sensitivity).T.astype(np.float64)

        for num_corners in [3, 2]:
            invalid_mask = _rms_truth_boxes[num_corners] > self.sensitivity
            _rms_pred_boxes[4][invalid_mask] = 0

        rms_truth = torch.zeros(current_batch_size, device=ifft_truth.device)
        rms_pred = torch.zeros(current_batch_size, device=ifft_pred.device)

        for num_corners in [4, 3, 2]:
            mask = rms_boxes == num_corners

            if not mask.any():
                continue

            rms_truth[mask] = (
                torch.abs(_rms_truth_boxes[num_corners][:, mask]).sum(dim=0)
                / num_corners
            )
            rms_pred[mask] = (
                torch.abs(_rms_pred_boxes[num_corners][:, mask]).sum(dim=0)
                / num_corners
            )

        print(f"{rms_truth.shape = }, {rms_truth = }")

        return rms_truth, rms_pred

    def __boxsize(self, num_corners: int, num_pixel: int = 63, box_factors=None) -> int:
        """
        Compute corner box size based on number of corners used.

        Parameters
        ----------
        num_corners : int
            Number of corners to use (2, 3, or 4).
        num_pixel : int, optional
            Image size in pixels. Default: 63

        Returns
        -------
        int
            Box size in pixels.
        """
        if not box_factors:
            box_factors = self.__BOX_FACTORS

        return int(num_pixel * box_factors[num_corners - 2])

    def __select_box(self, rms, sensitivity: float = 1e-6):
        """
        Select valid corner boxes based on RMS threshold.

        Parameters
        ----------
        rms : :func:`~numpy.ndarray`, shape (4, B)
            RMS values for each corner.
        sensitivity : float, optional
            Threshold below which corners are considered valid.
            Default: 1e-6.

        Returns
        -------
        :func:`numpy.ndarray`, shape (B,)
            Number of valid corners per sample.
        """
        valid_corners = rms <= sensitivity

        return valid_corners.sum(dim=0)

    def __compute_rms(self, batch, size: int):
        """
        Compute RMS in all four corner boxes.

        Parameters
        ----------
        batch : :func:`~numpy.ndarray`
            Batch of images, shape (B, H, W).
        size : int
            Corner box size in pixels.

        Returns
        -------
        :func:`numpy.ndarray`
            RMS values for each corner, shape (4, B).
        """
        corners = torch.stack(
            [
                batch[:, :size, :size],  # top left
                batch[:, :size, -size:],  # top right
                batch[:, -size:, :size],  # bottom left
                batch[:, -size:, -size:],  # bottom right
            ]
        )

        return torch.sqrt(
            (corners.reshape(4, len(batch), size * size) ** 2).mean(dim=2)
        )

    def compute(self) -> dict[str, torch.Tensor]:
        """Concatenate the integrated flux ratios and peak flux ratios
        across all batches.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing both integrated flux ratios ('integrated_flux')
            and peak flux ratios ('peak_flux') for the entire evaluated dataset.
        """
        return {
            "dr_preds": dim_zero_cat(self.dr_preds),
            "dr_targets": dim_zero_cat(self.dr_targets),
        }


class MeanDifference(Metric):
    """Computes the integrated flux and peak flux intensity ratios
    between prediction and ground truth images.

    Ratios are calculated at a threshold of the maximum true flux (default 5%).

    Parameters
    ----------
    threshold : float, optional
        Threshold at which to compute the metrics. Default: 0.05
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    mean_flux_preds: list[torch.Tensor]
    mean_flux_targets: list[torch.Tensor]

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__()

        self.add_state("mean_flux_preds", default=[], dist_reduce_fx="cat")
        self.add_state("mean_flux_targets", default=[], dist_reduce_fx="cat")

        self.threshold = threshold

    def update(self, ifft_pred: torch.Tensor, ifft_target: torch.Tensor) -> None:
        """Return cropped images around the first component of the true image.

        Parameters
        ----------
        pred : :func:`~numpy.ndarray`
            Predicted source image.
        truth : :func:`~numpy.ndarray`
            True source image.
        blob_truth : list or tuple
            Coordinates (y, x, r) for the first component.

        Returns
        -------
        flux_pred : :func:`~numpy.ndarray`
            Cropped prediction image.
        flux_truth : :func:`~numpy.ndarray`
            Cropped truth image.
        """
        blobs_target = self.__calc_blobs(ifft_target)[0]
        y, x, r = blobs_target[0], blobs_target[1], blobs_target[2]
        x_coord, y_coord = self.__corners(x, y, r)

        self.mean_flux_preds.append(
            torch.as_tensor(
                ifft_pred[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]
            ).mean(dim=(-2, -1))
        )
        self.mean_flux_targets.append(
            torch.tensor(
                ifft_target[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]
            ).mean(dim=(-2, -1))
        )

    def __calc_blobs(
        self,
        ifft_image: torch.Tensor,
    ):
        """Detect blobs using Laplacian of Gaussian in prediction
        and truth images.

        Parameters
        ----------
        ifft_image : :class:`~torch.Tensor`
            Predicted image (inverse FFT result), shape (N, 3).

        Returns
        -------
        blobs_log : :func:`~numpy.ndarray`
            Detected blobs in the given image, shape (N, 3) with columns [y, x, radius].
        """
        if isinstance(ifft_image, torch.Tensor):
            ifft_image = ifft_image.detach().cpu().numpy()

        threshold = ifft_image.max() * self.threshold
        kwargs = {
            "min_sigma": 1,
            "max_sigma": 10,
            "num_sigma": 100,
            "threshold": threshold,
            "overlap": 0.9,
        }

        blobs_log = blob_log(ifft_image, **kwargs)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

        return blobs_log

    def __corners(
        self,
        x: int | float,
        y: int | float,
        r: int | float,
    ) -> tuple[list[int], list[int]]:
        """Generate coordinate ranges for cropping the first component.

        Parameters
        ----------
        x : int or float
            X coordinate of the component center.
        y : int or float
            Y coordinate of the component center.
        r : int or float
            Radius of the first component.

        Returns
        -------
        x_coord : list of int
            Start and end indices for x-axis cropping.
        y_coord : list of int
            Start and end indices for y-axis cropping.
        """
        r = int(np.round(r))
        x = int(x)
        y = int(y)

        x_coord = [x - r, x + r + 1]
        y_coord = [y - r, y + r + 1]

        return x_coord, y_coord

    def compute(self) -> dict[str, torch.Tensor]:
        """Concatenate the flux values and compute the mean difference.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the mean difference ('mean_diff') in percent
            for the entire evaluated dataset.
        """
        mean_flux_preds = dim_zero_cat(self.mean_flux_preds)
        mean_flux_targets = dim_zero_cat(self.mean_flux_targets)

        mean_diff = (mean_flux_preds - mean_flux_targets) / mean_flux_targets * 100

        return {"mean_diff": mean_diff}


class ViewingAngle(Metric):
    """Computes the integrated flux and peak flux intensity ratios
    between prediction and ground truth images.

    Ratios are calculated at a threshold of the maximum true flux (default 5%).

    Parameters
    ----------
    threshold : float, optional
        Threshold at which to compute the metrics. Default: 0.05
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    pred_angle: list[torch.Tensor]
    pred_m: list[torch.Tensor]
    pred_n: list[torch.Tensor]
    target_angle: list[torch.Tensor]
    target_m: list[torch.Tensor]
    target_n: list[torch.Tensor]

    def __init__(self) -> None:
        super().__init__()

        self.add_state("pred_angle", default=[], dist_reduce_fx="cat")
        self.add_state("pred_m", default=[], dist_reduce_fx="cat")
        self.add_state("pred_n", default=[], dist_reduce_fx="cat")
        self.add_state("target_angle", default=[], dist_reduce_fx="cat")
        self.add_state("target_m", default=[], dist_reduce_fx="cat")
        self.add_state("target_n", default=[], dist_reduce_fx="cat")

    def update(self, ifft_pred: torch.Tensor, ifft_target: torch.Tensor) -> None:
        """Return cropped images around the first component of the true image.

        Parameters
        ----------
        pred : :func:`~numpy.ndarray`
            Predicted source image.
        truth : :func:`~numpy.ndarray`
            True source image.
        blob_truth : list or tuple
            Coordinates (y, x, r) for the first component.

        Returns
        -------
        flux_pred : :func:`~numpy.ndarray`
            Cropped prediction image.
        flux_truth : :func:`~numpy.ndarray`
            Cropped truth image.
        """
        pred_m, pred_n, pred_angle = self.jet_angle(ifft_pred)
        target_m, target_n, target_angle = self.jet_angle(ifft_target)

        self.pred_angle.append(pred_angle)
        self.pred_m.append(pred_m)
        self.pred_n.append(pred_n)
        self.target_angle.append(target_m)
        self.target_m.append(target_m)
        self.target_n.append(target_n)

    def jet_angle(
        self,
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

        _, _, alpha_pca = self.__pca(image)

        # Search for sources with two maxima
        maxima = []
        for img in image:
            a = torch.where(img == img.max())
            if len(a[0]) > 1:
                # if two maxima are found, interpolate to the middle
                # in x and y direction
                mid_x = (a[0][1] - a[0][0]) // 2 + a[0][0]
                mid_y = (a[1][1] - a[1][0]) // 2 + a[1][0]
                maxima.extend([(mid_x, mid_y)])
            else:
                maxima.extend([a])

        vals = torch.tensor(maxima, device=image.device)
        x_mid = vals[:, 0]
        y_mid = vals[:, 1]

        m = torch.tan(torch.tensor(pi / 2, device=image.device) - alpha_pca)
        n = y_mid - m * x_mid
        alpha = torch.rad2deg(alpha_pca)

        return m, n, alpha

    def __pca(
        self,
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
        pix_x, pix_y, image = self.__im2array_value(image)

        image_sum = image.sum(dim=1, keepdim=True)
        cog_x = (pix_x * image).sum(dim=1, keepdim=True) / image_sum
        cog_y = (pix_y * image).sum(dim=1, keepdim=True) / image_sum

        delta_x = pix_x - cog_x
        delta_y = pix_y - cog_y

        inp = torch.stack([delta_x, delta_y], dim=1)

        cov_w = self.__bmul(
            (cog_x - 1 * torch.sum(image * image, dim=1).unsqueeze(-1) / cog_x).squeeze(
                1
            ),
            (torch.matmul(image.unsqueeze(1) * inp, inp.transpose(1, 2))),
        )

        _, eig_vecs_torch = torch.linalg.eigh(cov_w, UPLO="U")
        psi_torch = torch.atan(eig_vecs_torch[:, 1, 1] / eig_vecs_torch[:, 0, 1])

        return cog_x, cog_y, psi_torch

    def __bmul(
        self, vec: torch.Tensor, mat: torch.Tensor, axis: int = 0
    ) -> torch.Tensor:
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

    def __im2array_value(
        self, image: torch.Tensor
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

    def compute(self) -> dict[str, torch.Tensor]:
        """Concatenate the flux values and compute the mean difference.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the mean difference ('mean_diff') in percent
            for the entire evaluated dataset.
        """
        pred_angle = dim_zero_cat(self.pred_angle)
        target_angle = dim_zero_cat(self.target_angle)
        diff = pred_angle - target_angle

        return {
            "diff": diff,
            "pred_angle": pred_angle,
            "pred_m": dim_zero_cat(self.pred_m),
            "pred_n": dim_zero_cat(self.pred_n),
            "target_angle": target_angle,
            "target_m": dim_zero_cat(self.target_m),
            "target_n": dim_zero_cat(self.target_n),
        }
