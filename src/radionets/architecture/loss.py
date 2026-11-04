from abc import ABC, abstractmethod

import torch
from torch import nn

from radionets.evaluation.utils import apply_symmetry


class SplitL1Loss(nn.Module):
    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        inp_amp = pred[:, 0, :]
        inp_phase = pred[:, 1, :]

        tar_amp = target[:, 0, :]
        tar_phase = target[:, 1, :]

        l1 = nn.L1Loss(self.reduction)
        loss_amp = l1(inp_amp, tar_amp)
        loss_phase = l1(inp_phase, tar_phase)
        loss = loss_amp + loss_phase

        return loss


class MaskedSplitL1Loss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        center: list | tuple | None = None,
        radius: int | None = 30,
        **kwargs,
    ) -> None:
        super().__init__()

        self.center = center
        self.radius = radius

        # Assign mask so it can be cached during forward call;
        # None at first, then torch.Tensor after caching
        self._mask: torch.Tensor | None = None

        self._l1 = nn.L1Loss(reduction=reduction)

    def _create_circular_mask(
        self,
        w: int,
        h: int,
        center: list[int] | tuple[int, int] | None = None,
        radius: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if center is None:
            center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        x = torch.arange(w, device=device).view(1, -1)
        y = torch.arange(h, device=device).view(-1, 1)
        dist_from_center = (x - center[0]) ** 2 + (y - center[1]) ** 2

        return dist_from_center <= radius**2

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = apply_symmetry(inputs)
        targets = apply_symmetry(targets)

        *_, h, w = targets.shape

        inp_amp = inputs[:, 0]
        inp_phase = inputs[:, 1]

        tar_amp = targets[:, 0]
        tar_phase = targets[:, 1]

        if self._mask is None or self._mask.device != inputs.device:
            self._mask = self._create_circular_mask(
                w=w,
                h=h,
                center=self.center,
                radius=self.radius,
                device=inputs.device,
            )

        weight = torch.where(self._mask, 1.0, 0.3)

        inp_amp *= weight
        inp_phase *= weight
        tar_amp *= weight
        tar_phase *= weight

        loss_amp = self._l1(inp_amp, tar_amp)
        loss_phase = self._l1(inp_phase, tar_phase)
        loss = loss_amp + loss_phase

        return loss


class _FluxEmoioBase(ABC, nn.Module):
    """
    Abstract base class for all FluxEmoio loss components.

    Subclasses must implement _extract_components to define which tensor
    component (amplitude, phase, real, imaginary) is used for the loss.
    """

    def __init__(self, levels, weights, **kwargs):
        super().__init__()
        # maybe outsource if`s
        if not levels:
            levels = [0.88, 0.75, 0.63, 0.50, 0.38, 0.25, 0.13, 0.05]
        if not weights:
            weights = [2.00, 1.76, 1.50, 1.26, 1.00, 0.76, 0.50, 0.26]
        if not all([isinstance(levels, list), isinstance(weights, list)]):
            raise ValueError(
                "'radii' and 'weights' must be lists, but got "
                f"{type(levels)} and {type(weights)} instead."
            )

        # rewrite!!
        if not len(weights) == len(levels):
            raise ValueError("len(weights) has to be equal to len(levels)")

        self.levels = levels
        self.weights = weights
        self.loss_fn = nn.L1Loss()

    @abstractmethod
    def _extract_components(
        self, pred: torch.Tensor, truth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract the relevant component tensors from prediction and ground truth.

        Must be implemented by each subclass to select the appropriate
        channel from the input tensors.

        Returns (pred_component, truth_component) for the loss.
        """
        ...

    def _masked_loss(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked weighted L1 loss for a single component.

        Partitions the ground truth into regions based on amplitude thresholds
        derived from 'self.levels'. Each region is weighted according to
        'self.weights', giving higher importance to regions with larger values.
        The remaining region below the lowest threshold is computed without
        any weighting.

        Parameters
        ----------
        pred:  Predicted component tensor of shape (B, H, W).
        truth: Ground truth component tensor of shape (B, H, W).

        Returns
        -------
        Scalar loss tensor, sum of weighted masked loss and unweighted
        remainder loss.
        """

        truth_max = truth.amax(dim=(-2, -1), keepdim=True)
        thresholds = [truth_max * level for level in self.levels]
        masks = [torch.abs(truth) > thresholds[0]]

        for i in range(len(thresholds) - 1):
            mask = (torch.abs(truth) <= thresholds[i]) & (
                torch.abs(truth) > thresholds[i + 1]
            )
            masks.append(mask)

        masks.append(torch.abs(truth) <= thresholds[-1])

        loss_parts = [
            self.loss_fn(pred[m] * w, truth[m] * w)
            for m, w in zip(masks[:-1], self.weights)
        ]

        loss_mask = sum(loss_parts)

        loss_rest = self.loss_fn(pred[masks[-1]], truth[masks[-1]])

        loss = loss_rest + loss_mask

        return loss

    def forward(self, x: dict, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the masked weighted L1 loss between prediction and ground truth.

        Applies symmetry to both prediction and ground truth before extracting
        the relevant component and computing the loss.

        Parameters
        ----------
            x:  Dictionary containing the model prediction under the key "pred"
                as a torch.Tensor of shape (B, C, H, W).
            y:  Ground truth tensor of shape (B, C, H, W).

        Returns
        -------
            Scalar loss tensor.
        """

        pred = apply_symmetry({"pred": x["pred"]})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred, truth = self._extract_components(pred, truth)

        loss = self._masked_loss(pred, truth)

        return loss


class Amp(_FluxEmoioBase):
    """Amplitude loss component, extracts channel index 0."""

    def _extract_components(self, pred, truth):
        return pred[:, 0, :].clone(), truth[:, 0, :].clone()


class Phase(_FluxEmoioBase):
    """Phase loss component, extracts channel index 1."""

    def _extract_components(self, pred, truth):
        return pred[:, 1, :].clone(), truth[:, 1, :].clone()


class Real(_FluxEmoioBase):
    """Real loss component, extracts channel index 0."""

    def _extract_components(self, pred, truth):
        return pred[:, 0, :].clone(), truth[:, 0, :].clone()


class Imag(_FluxEmoioBase):
    """Imaginary loss component, extracts channel index 1."""

    def _extract_components(self, pred, truth):
        return pred[:, 1, :].clone(), truth[:, 1, :].clone()


class AmpPhase(Amp, Phase):
    """
    Combined amplitude and phase loss.

    Computes both losses independently and returns their sum. Uses a
    separate Phase instance.
    """

    def __init__(self, levels=None, weights=None, **kwargs):
        super().__init__(levels=levels, weights=weights, **kwargs)
        self._phase_head = Phase(levels=levels, weights=weights, **kwargs)

    def forward(self, x, y):
        loss_amp = Amp.forward(self, x, y)
        loss_phase = self._phase_head(x, y)
        return loss_amp + loss_phase


class RealImag(Real, Imag):
    """
    Combined real and imsginary loss.

    Computes both losses independently and returns their sum. Uses a
    separate Phase instance.
    """

    def __init__(self, levels=None, weights=None, **kwargs):
        super().__init__(levels=levels, weights=weights, **kwargs)
        self._imag_head = Imag(levels=levels, weights=weights, **kwargs)

    def forward(self, x, y):
        loss_real = Real.forward(self, x, y)
        loss_imag = self._imag_head(x, y)
        return loss_real + loss_imag


_MODE_MAP: dict[str, type] = {
    "amp": Amp,
    "phase": Phase,
    "real": Real,
    "imag": Imag,
    "amp_phase": AmpPhase,
    "phase_amp": AmpPhase,
    "real_imag": RealImag,
    "imag_real": RealImag,
}


class FluxEmoio(nn.Module):
    """
    Factory wrapper

    Selects and instantiates the appropriate loss component class based
    on the given mode string.

    Parameters
    ----------
    levels : list of float, optional
        Relative amplitude thresholds as fractions of the maximum flux density used to
        partition the ground truth into masked regions. E.g. 0.88 masks all
        values above 88% of the maximum. Must be in descending order.
    weights : list of float, optional
        Loss weights applied to each masked region defined by levels. Higher
        weights increase the contribution of high flux density regions to the
        total loss.
    mode : str
        Selects the loss component. Valid options:
        'amp', 'phase', 'real', 'imag',
        'amp_phase' / 'phase_amp', 'real_imag' / 'imag_real'.
    **kwargs
        Additional keyword arguments forwarded to the selected component.

    Attributes
    ----------
    loss : _FluxEmoioBase
        The instantiated loss component selected by mode.

    Raises
    ------
    ValueError
        If mode is not a valid key in _MODE_MAP.

    Examples
    --------
        loss = FluxEmoio(levels = None, weights = None, mode="amp")

    """

    def __init__(
        self,
        levels: list | None = None,
        weights: list | None = None,
        mode: str | None = None,
        **kwargs,
    ):
        super().__init__()
        mode = mode.lower().strip()
        cls = _MODE_MAP.get(mode)
        if cls is None:
            valid = ", ".join(_MODE_MAP)
            raise ValueError(f"Unknown mode '{mode}'. Valid options: {valid}")

        self.loss: _FluxEmoioBase = cls(levels=levels, weights=weights, **kwargs)

    def forward(self, x: dict, y: torch.Tensor) -> torch.Tensor:
        return self.loss(x, y)
