import torch
from torch import nn

from radionets.evaluation.utils import apply_symmetry


class SplittedL1Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
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
