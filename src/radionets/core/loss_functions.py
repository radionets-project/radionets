import numpy as np
import torch
from torch import nn

from radionets.core.utils import get_ifft_torch
from radionets.evaluation.utils import apply_symmetry, get_ifft

__all__ = [
    "beta_nll_loss",
    "create_circular_mask",
    "jet_seg",
    "l1",
    "L1Real",
    "RealL1Mask",
    "l1_phase",
    "mse",
    "splitted_L1",
    "splitted_L1_masked",
]

####### L1 normal #######


# pre
def l1(x, y):
    pred = x["pred"]

    l1 = nn.L1Loss()
    loss = l1(pred, y)

    return loss


# Real
class L1Real(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y, **kwargs):
        pred = x["pred"]

        l1 = nn.L1Loss()
        loss = l1(pred[:, 0], y[:, 0])

        return loss


# Imag
class L1Imag(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y, **kwargs):
        pred = x["pred"]

        l1 = nn.L1Loss()
        loss = l1(pred[:, 1], y[:, 1])

        return loss


def l1_phase(x, y):
    pred = x["pred"]

    l1 = nn.L1Loss()
    loss = l1(pred[:, 1], y[:, 1])

    return loss


####### Create mask(s) #######


# pre, one mask
def create_circular_mask(self, h, w, center=None, radius=None, bs=64):
    if center is None:
        center = (int(w / 2), int(h / 2))

    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    return np.repeat([mask], bs, axis=0)


# 2 radii
def create_circular_masks(h, w, radius1, radius2=None, center=None, bs=64):
    if center is None:
        center = (int(w / 2), int(h / 2))

    if radius2 is None:
        if (radius1 + min(h, w) // 4) < min(h - 5, w - 5):
            radius2 = radius1 + min(h, w) // 4

        elif (radius1 + min(h, w) // 8) < min(h - 5, w - 5):
            radius2 = radius1 + min(h, w) // 8

        else:
            radius2 = radius1 + 1

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask1 = dist_from_center <= radius1
    mask2 = (dist_from_center >= radius1) & (dist_from_center <= radius2)
    mask3 = dist_from_center >= radius2

    return (
        np.repeat([mask1], bs, axis=0),
        np.repeat([mask2], bs, axis=0),
        np.repeat([mask3], bs, axis=0),
    )


# N radii
def create_circular_masks2(h, w, radii=None, N_radi=None, center=None, bs=64):
    if radii is None:
        if N_radi == 1:
            radii = [int(min(w, h)) // 2]

        n = int(min(w, h) / N_radi) if h != w else h // 2 // N_radi

        radii = [n]
        for i in range(0, N_radi - 1):
            radii.append(radii[i] + n)

    radii = sorted(radii)

    if center is None:
        center = (int(w / 2), int(h / 2)) if w == h else (int(max(w, h) / 2), min(w, h))

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    masks = [dist_from_center <= radii[0]]

    for i, _ in enumerate(radii[0:-1]):
        mask = (dist_from_center > radii[i]) & (dist_from_center <= radii[i + 1])
        masks.append(mask)

    masks.append(dist_from_center > radii[-1])

    return tuple(np.repeat([mask], bs, axis=0) for mask in masks)


####### Circular mask #######


# amp
def amp_L1_masked(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=10, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.3
    tar_amp[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


# amp + phase
def splitted_L1_masked(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]
    inp_phase = pred[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(create_circular_mask(256, 256, radius=50, bs=y.shape[0]))

    inp_amp[~mask] *= 0.3
    inp_phase[~mask] *= 0.3
    tar_amp[~mask] *= 0.3
    tar_phase[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase

    return loss


# Real
class RealL1Mask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def create_circular_mask(self, h, w, center=None, radius=None, bs=64):
        if center is None:
            center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius

        return np.repeat([mask], bs, axis=0)

    def forward(self, x, y):
        pred = x["pred"]
        pred = apply_symmetry({"pred": pred})["pred"]
        inp_real = pred[:, 0, :]

        y = apply_symmetry({"y": y})["y"]
        tar_real = y[:, 0, :]

        mask = torch.tensor(
            self.create_circular_mask(
                inp_real.shape[-2], inp_real.shape[-1], radius=32, bs=y.shape[0]
            )
        )

        inp_real[~mask] *= 0.3
        tar_real[~mask] *= 0.3

        l1 = nn.L1Loss()
        loss_amp = l1(inp_real, tar_real)
        loss = loss_amp

        return loss


# Imag
class ImagL1Mask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def create_circular_mask(self, h, w, center=None, radius=None, bs=64):
        if center is None:
            center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius

        return np.repeat([mask], bs, axis=0)

    def forward(self, x, y):
        pred = x["pred"]
        pred = apply_symmetry({"pred": pred})["pred"]
        inp_imag = pred[:, 1, :]

        y = apply_symmetry({"y": y})["y"]
        tar_imag = y[:, 1, :]

        mask = torch.tensor(
            self.create_circular_mask(
                inp_imag.shape[-2], inp_imag.shape[-1], radius=32, bs=y.shape[0]
            )
        )

        inp_imag[~mask] *= 0.3
        tar_imag[~mask] *= 0.3

        l1 = nn.L1Loss()
        loss_amp = l1(inp_imag, tar_imag)
        loss = loss_amp

        return loss


####### combined loss #######
# max peak, total sum and circular masks


# combined, all L1, amp
def amp_combined_loss(x, y):
    pred = x["pred"]

    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_amp = pred[:, 0, :].clone()
    truth_phase = truth[:, 1, :].clone()

    pred = torch.cat((pred_amp, truth_phase), dim=1)

    pred_ifft = get_ifft_torch(pred, amp_phase=True)
    truth_ifft = get_ifft_torch(truth, amp_phase=True)

    l1 = nn.L1Loss()

    loss_peak = l1((pred_ifft.max()), torch.as_tensor(truth_ifft.max()))
    loss_total = l1(torch.as_tensor(pred_ifft.sum()), torch.as_tensor(truth_ifft.sum()))

    truth_amp = truth[:, 0, :]

    radii = [5, 10, 15, 20, 30]
    masks = create_circular_masks2(256, 256, radii=radii, bs=y.shape[0])
    # weights = 1 / np.exp(np.linspace(0, 1, len(masks)))
    weights = [10, 5, 2, 1, 0.5, 0.25]

    for mask, weight in zip(masks, weights):
        pred_amp[mask] *= weight
        truth_amp[mask] *= weight

    loss_mask = l1(pred_amp, truth_amp)

    loss = loss_mask + 0.1 * loss_peak + 0.1 * loss_total

    return loss


# combined, all L1, phase
def phase_combined_loss(x, y):
    pred = x["pred"]

    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_ifft = get_ifft(pred, amp_phase=True)
    truth_ifft = get_ifft(truth, amp_phase=True)

    l1 = nn.L1Loss()

    loss_peak = l1(torch.as_tensor(pred_ifft.max()), torch.as_tensor(truth_ifft.max()))
    loss_total = l1(torch.as_tensor(pred_ifft.sum()), torch.as_tensor(truth_ifft.sum()))

    pred = pred[:, 1, :]
    truth = truth[:, 1, :]

    radii = [50, 75, 100, 125]
    masks = create_circular_masks2(256, 256, radii=radii, bs=y.shape[0])
    # weights = 1 / np.exp(np.linspace(0, 1, len(masks)))
    weights = [10, 5, 2, 1, 0.25]

    for mask, weight in zip(masks, weights):
        pred[mask] *= weight
        truth[mask] *= weight

    loss_mask = l1(pred, truth)

    loss = loss_mask + 0.1 * loss_peak + 0.1 * loss_total

    return loss


# code mainly from copilot, all huber, amp
def amp_combined_loss_copilot_huber(x, y):
    """
    Combined loss for amplitude reconstruction with masked weighting.

    Uses Huber loss for peak/total (robust against outliers) and
    L1 loss for masked amplitude comparison.
    """
    pred = x["pred"]

    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_amp = pred[:, 0, :].clone()
    truth_amp = truth[:, 0, :].clone()
    truth_phase = truth[:, 1, :]

    pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

    pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
    truth_ifft = get_ifft_torch(truth, amp_phase=True)

    hu1 = nn.HuberLoss(delta=8.2558)
    hu2 = nn.HuberLoss(delta=0.0443)
    hu3 = nn.HuberLoss(delta=0.008)

    # --- Peak Loss:  Huber (robust gegen Spikes) ---
    pred_peak = pred_ifft.amax(dim=(-2, -1))
    truth_peak = truth_ifft.amax(dim=(-2, -1))
    loss_peak = hu1(pred_peak, truth_peak)

    # --- Total Flux:  Huber (robust + normalisiert) ---
    n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
    pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
    truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
    loss_total = hu2(pred_total, truth_total)

    # --- Masked Amplitude:  L1 (pixelweise, robust) ---
    radii = [5, 10, 15, 20, 30]
    masks = create_circular_masks2(256, 256, radii=radii, bs=y.shape[0])
    weights = [2.0, 1.5, 1, 0.75, 0.5, 0.25]

    weight_map = torch.ones_like(pred_amp)
    for mask, weight in zip(masks, weights):
        mask_tensor = torch.from_numpy(mask).to(pred_amp.device)
        weight_map = torch.where(mask_tensor, weight, weight_map)

    # l1 = nn.L1Loss()

    loss_mask = hu3(pred_amp * weight_map, truth_amp * weight_map)

    loss = loss_mask + 0.1 * loss_peak + 0.1 * loss_total

    return loss


# code mainly from copilot, total & peak huber, mask L1, phase
def phase_combined_loss_copilot(x, y):
    """
    Combined loss for amplitude reconstruction with masked weighting.

    Uses Huber loss for peak/total (robust against outliers) and
    L1 loss for masked amplitude comparison.
    """
    pred = x["pred"]

    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_phase = pred[:, 1, :].clone()
    truth_phase = truth[:, 1, :].clone()
    truth_amp = truth[:, 0, :]

    pred_combined = torch.stack([truth_amp, pred_phase], dim=1)

    pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
    truth_ifft = get_ifft_torch(truth, amp_phase=True)

    hu = nn.HuberLoss(delta=1.0)
    l1 = nn.L1Loss()

    # --- Peak Loss:  Huber (robust gegen Spikes) ---
    pred_peak = pred_ifft.amax(dim=(-2, -1))
    truth_peak = truth_ifft.amax(dim=(-2, -1))
    loss_peak = hu(pred_peak, truth_peak)

    # --- Total Flux:  Huber (robust + normalisiert) ---
    n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
    pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
    truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
    loss_total = hu(pred_total, truth_total)

    # --- Masked Amplitude:  L1 (pixelweise, robust) ---
    radii = [50, 75, 100, 125]
    masks = create_circular_masks2(256, 256, radii=radii, bs=y.shape[0])
    weights = [10, 5, 2, 1, 0.25]

    weight_map = torch.ones_like(pred_phase)
    for mask, weight in zip(masks, weights):
        mask_tensor = torch.from_numpy(mask).to(pred_phase.device)
        weight_map = torch.where(mask_tensor, weight, weight_map)

    loss_mask = l1(pred_phase * weight_map, truth_phase * weight_map)

    loss = loss_mask + 0.1 * loss_peak + 0.1 * loss_total

    return loss


# code mainly from copilot, all huber, amp
# Every Mask On Its Own in huber loss


def amp_combined_loss_emoio(x, y):
    """
    Combined loss for amplitude reconstruction with masked weighting.

    Uses Huber loss for peak/total (robust against outliers) and
    L1 loss for masked amplitude comparison.
    """
    pred = x["pred"]

    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_amp = pred[:, 0, :].clone()
    truth_amp = truth[:, 0, :].clone()
    truth_phase = truth[:, 1, :]

    pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

    pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
    truth_ifft = get_ifft_torch(truth, amp_phase=True)

    hu1 = nn.HuberLoss(delta=8.2558)
    hu2 = nn.HuberLoss(delta=0.0443)
    hu3 = nn.HuberLoss(delta=0.008)

    # --- Peak Loss:  Huber (robust gegen Spikes) ---
    pred_peak = pred_ifft.amax(dim=(-2, -1))
    truth_peak = truth_ifft.amax(dim=(-2, -1))
    loss_peak = hu1(pred_peak, truth_peak)

    # --- Total Flux:  Huber (robust + normalisiert) ---
    n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
    pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
    truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
    loss_total = hu2(pred_total, truth_total)

    # --- Masked Amplitude:  ---
    radii = [15, 25, 50, 75]
    masks = create_circular_masks2(256, 256, radii=radii, bs=y.shape[0])
    weights = [1, 0.80, 0.75, 0.5, 0.25]

    loss = []

    for mask, weight in zip(masks, weights):
        loss.append(hu3(pred_amp[mask], truth_amp[mask]) * weight)

    loss_mask = sum(loss)

    loss = loss_mask + 0.1 * loss_peak + 0.1 * loss_total

    return loss


# code mainly from copilot, all huber, phase
# Every Mask On Its Own in huber loss


def phase_combined_loss_emoio(x, y):
    """
    Combined loss for phase reconstruction with masked weighting.

    Uses Huber loss for peak/total/mask (robust against outliers)
    """
    pred = x["pred"]

    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_phase = pred[:, 1, :].clone()
    truth_phase = truth[:, 1, :].clone()
    truth_amp = truth[:, 0, :]

    pred_combined = torch.stack([pred_phase, truth_amp], dim=1)

    pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
    truth_ifft = get_ifft_torch(truth, amp_phase=True)

    hu1 = nn.HuberLoss(delta=8.2558)
    hu2 = nn.HuberLoss(delta=0.0443)
    hu3 = nn.HuberLoss(delta=0.008)

    # --- Peak Loss:  Huber (robust gegen Spikes) ---
    pred_peak = pred_ifft.amax(dim=(-2, -1))
    truth_peak = truth_ifft.amax(dim=(-2, -1))
    loss_peak = hu1(pred_peak, truth_peak)

    # --- Total Flux:  Huber (robust + normalisiert) ---
    n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
    pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
    truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
    loss_total = hu2(pred_total, truth_total)

    # --- Masked Amplitude:  ---
    radii = [25, 50, 75, 100, 125]
    masks = create_circular_masks2(256, 256, radii=radii, bs=y.shape[0])
    weights = [1, 0.80, 0.75, 0.5, 0.25, 0.05]

    loss = []

    for mask, weight in zip(masks, weights):
        loss.append(hu3(pred_phase[mask], truth_phase[mask]) * weight)

    loss_mask = sum(loss)

    loss = loss_mask + 0.1 * loss_peak + 0.1 * loss_total

    return loss


class RealCombinedEmoio(nn.Module):
    def __init__(self, radii=None, weights=None, **kwargs):
        super().__init__()
        if not radii:
            radii = [15, 25, 50, 75]
        if not weights:
            weights = [1, 0.80, 0.75, 0.5, 0.25]
        if not all([isinstance(radii, list), isinstance(weights, list)]):
            raise ValueError(
                "'radii' and 'weights' must be lists, but got "
                f"{type(radii)} and {type(weights)} instead."
            )
        if not len(weights) == len(radii) + 1:
            raise ValueError("len(weights) has to be equal to len(radii) +1")
        self.radii = radii
        self.weights = weights

    def forward(self, x, y):
        """
        Combined loss for amplitude reconstruction with masked weighting.

        Uses Huber loss for peak/total (robust against outliers) and
        L1 loss for masked amplitude comparison.
        From copilot.
        """
        pred = x["pred"]

        pred = apply_symmetry({"pred": pred})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred_amp = pred[:, 0, :].clone()
        truth_amp = truth[:, 0, :].clone()
        truth_phase = truth[:, 1, :]

        pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

        pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
        truth_ifft = get_ifft_torch(truth, amp_phase=True)

        hu1 = nn.HuberLoss(delta=8.2558)
        hu2 = nn.HuberLoss(delta=0.0443)
        hu3 = nn.HuberLoss(delta=0.008)

        # --- Peak Loss:  Huber (robust gegen Spikes) ---
        pred_peak = pred_ifft.amax(dim=(-2, -1))
        truth_peak = truth_ifft.amax(dim=(-2, -1))
        loss_peak = hu1(pred_peak, truth_peak)

        # --- Total Flux:  Huber (robust + normalisiert) ---
        n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
        pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
        truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
        loss_total = hu2(pred_total, truth_total)

        # --- Masked Amplitude:  ---
        masks = create_circular_masks2(256, 256, radii=self.radii, bs=y.shape[0])

        loss = []

        for mask, weight in zip(masks, self.weights):
            loss.append(hu3(pred_amp[mask], truth_amp[mask]) * weight)

        loss_mask = sum(loss)

        loss = loss_mask + 100 * loss_peak + 10**5 * loss_total

        return loss


# imag
class ImagCombinedEmoio(nn.Module):
    def __init__(self, radii=None, weights=None, **kwargs):
        super().__init__()
        if not radii:
            radii = [15, 25, 50, 75]
        if not weights:
            weights = [1, 0.80, 0.75, 0.5, 0.25]
        if not all([isinstance(radii, list), isinstance(weights, list)]):
            raise ValueError(
                "'radii' and 'weights' must be lists, but got "
                f"{type(radii)} and {type(weights)} instead."
            )
        if not len(weights) == len(radii) + 1:
            raise ValueError("len(weights) has to be equal to len(radii) +1")
        self.radii = radii
        self.weights = weights

    def forward(self, x, y):
        """
        Combined loss for amplitude reconstruction with masked weighting.

        Uses Huber loss for peak/total (robust against outliers) and
        L1 loss for masked amplitude comparison.
        From copilot.
        """
        pred = x["pred"]

        pred = apply_symmetry({"pred": pred})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred_amp = pred[:, 1, :].clone()
        truth_amp = truth[:, 1, :].clone()
        truth_phase = truth[:, 0, :]

        pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

        pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
        truth_ifft = get_ifft_torch(truth, amp_phase=True)

        hu1 = nn.HuberLoss(delta=8.2558)
        hu2 = nn.HuberLoss(delta=0.0443)
        hu3 = nn.HuberLoss(delta=0.008)

        # --- Peak Loss:  Huber (robust gegen Spikes) ---
        pred_peak = pred_ifft.amax(dim=(-2, -1))
        truth_peak = truth_ifft.amax(dim=(-2, -1))
        loss_peak = hu1(pred_peak, truth_peak)

        # --- Total Flux:  Huber (robust + normalisiert) ---
        n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
        pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
        truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
        loss_total = hu2(pred_total, truth_total)

        # --- Masked Amplitude:  ---
        masks = create_circular_masks2(256, 256, radii=self.radii, bs=y.shape[0])

        loss = []

        for mask, weight in zip(masks, self.weights):
            loss.append(hu3(pred_amp[mask], truth_amp[mask]) * weight)

        loss_mask = sum(loss)

        loss = loss_mask + 100 * loss_peak + 10**5 * loss_total

        return loss


####### Masks FLux ######


# just flux mask, amp
def amp_flux_masks(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    truth = apply_symmetry({"y": y})["y"]

    pred_amp = pred[:, 0]
    truth_amp = truth[:, 0]

    # levels = [0.95, 0.90, 0.75, 0.50]
    # weights = [2, 1.5, 1, 0.5, 0.3]

    levels = [0.88, 0.75, 0.63, 0.50, 0.38, 0.25, 0.13, 0.07]
    weights = [2.00, 1.76, 1.50, 1.26, 1.00, 0.76, 0.50, 0.26, 0.14]

    hu = nn.HuberLoss(delta=0.008)

    # Funktioniert für ganzen Batch - jedes Bild bekommt eigenes Max!
    truth_max = truth_amp.amax(dim=(-2, -1), keepdim=True)

    thresholds = [truth_max * level for level in levels]

    masks = [truth_amp > thresholds[0]]

    for i, _ in enumerate(thresholds[0:-1]):
        mask = (truth_amp <= thresholds[i]) & (truth_amp > thresholds[i + 1])
        masks.append(mask)

    masks.append(truth_amp <= thresholds[-1])

    weight_map = torch.ones_like(truth_amp)
    for mask, weight in zip(masks, weights):
        weight_map = torch.where(mask, weight, weight_map)

    loss = hu(pred_amp * weight_map, truth_amp * weight_map)

    return loss


# flux mask, combined with peak & total, emoio, all huber, amp
class AmpCombinedFluxEmoio(nn.Module):
    def __init__(self, levels, weights, **kwargs):
        super().__init__()
        self.levels = levels
        self.weights = weights

    def forward(self, x, y):
        """
        Combined loss for amplitude reconstruction with masked weighting.

        Uses Huber loss for peak/total (robust against outliers) and
        L1 loss for masked amplitude comparison.
        """
        pred = x["pred"]

        pred = apply_symmetry({"pred": pred})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred_amp = pred[:, 0, :].clone()
        truth_amp = truth[:, 0, :].clone()
        truth_phase = truth[:, 1, :]

        pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

        pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
        truth_ifft = get_ifft_torch(truth, amp_phase=True)

        hu1 = nn.HuberLoss(delta=8.2558)
        hu2 = nn.HuberLoss(delta=0.0443)
        hu3 = nn.HuberLoss(delta=0.008)

        # --- Peak Loss:  Huber (robust gegen Spikes) ---
        pred_peak = pred_ifft.amax(dim=(-2, -1))
        truth_peak = truth_ifft.amax(dim=(-2, -1))
        loss_peak = hu1(pred_peak, truth_peak)

        # --- Total Flux:  Huber (robust + normalisiert) ---
        n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
        pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
        truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
        loss_total = hu2(pred_total, truth_total)

        # --- Masked Amplitude:  ---
        truth_max = truth_amp.amax(dim=(-2, -1), keepdim=True)

        thresholds = [truth_max * level for level in self.levels]
        masks = [truth_amp > thresholds[0]]

        for i, _ in enumerate(thresholds[0:-1]):
            mask = (truth_amp <= thresholds[i]) & (truth_amp > thresholds[i + 1])
            masks.append(mask)

        masks.append(truth_amp <= thresholds[-1])

        # --- emoio: ---

        loss = []

        for mask, weight in zip(masks, self.weights):
            loss.append(hu3(pred_amp[mask] * weight, truth_amp[mask] * weight))

        loss_mask = sum(loss)

        loss = loss_mask + 100 * loss_peak + 10**5 * loss_total

        # print(f"{loss_mask = }, {loss_peak = }, {loss_total = }")
        # print(f"{loss_mask = }, {100 * loss_peak = }, {10**5 * loss_total = }")

        return loss


# imag/ phase
class ImagCombinedFluxEmoio(nn.Module):
    def __init__(self, levels, weights, **kwargs):
        super().__init__()
        self.levels = levels
        self.weights = weights

    def forward(self, x, y):
        """
        Combined loss for amplitude reconstruction with masked weighting.

        Uses Huber loss for peak/total (robust against outliers) and
        L1 loss for masked amplitude comparison.
        """
        pred = x["pred"]

        pred = apply_symmetry({"pred": pred})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred_amp = pred[:, 1, :].clone()
        truth_amp = truth[:, 1, :].clone()
        truth_phase = truth[:, 0, :]

        pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

        pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
        truth_ifft = get_ifft_torch(truth, amp_phase=True)

        hu1 = nn.HuberLoss(delta=8.2558)
        hu2 = nn.HuberLoss(delta=0.0443)
        hu3 = nn.HuberLoss(delta=0.008)

        # --- Peak Loss:  Huber (robust gegen Spikes) ---
        pred_peak = pred_ifft.amax(dim=(-2, -1))
        truth_peak = truth_ifft.amax(dim=(-2, -1))
        loss_peak = hu1(pred_peak, truth_peak)

        # --- Total Flux:  Huber (robust + normalisiert) ---
        n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
        pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
        truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
        loss_total = hu2(pred_total, truth_total)

        # --- Masked Amplitude:  ---
        truth_max = truth_amp.amax(dim=(-2, -1), keepdim=True)

        thresholds = [truth_max * level for level in self.levels]
        masks = [truth_amp > thresholds[0]]

        for i, _ in enumerate(thresholds[0:-1]):
            mask = (truth_amp <= thresholds[i]) & (truth_amp > thresholds[i + 1])
            masks.append(mask)

        masks.append(truth_amp <= thresholds[-1])

        # --- emoio: ---

        loss = []

        for mask, weight in zip(masks, self.weights):
            loss.append(hu3(pred_amp[mask] * weight, truth_amp[mask] * weight))

        loss_mask = sum(loss)

        loss = loss_mask + 100 * loss_peak + 10**5 * loss_total

        # print(f"{loss_mask = }, {loss_peak = }, {loss_total = }")
        # print(f"{loss_mask = }, {100 * loss_peak = }, {10**5 * loss_total = }")

        return loss


class ImagCombinedFluxEmoio2(nn.Module):
    def __init__(self, levels, weights, **kwargs):
        super().__init__()
        if not levels:
            levels = [0.88, 0.75, 0.63, 0.50, 0.38, 0.25, 0.13, 0.05]
        if not weights:
            weights = [2.00, 1.76, 1.50, 1.26, 1.00, 0.76, 0.50, 0.26]
        if not all([isinstance(levels, list), isinstance(weights, list)]):
            raise ValueError(
                "'radii' and 'weights' must be lists, but got "
                f"{type(levels)} and {type(weights)} instead."
            )
        if not len(weights) == len(levels):
            raise ValueError("len(weights) has to be equal to len(levels)")
        self.levels = levels
        self.weights = weights

    def forward(self, x, y):
        """
        Combined loss for amplitude reconstruction with masked weighting.

        Uses Huber loss for peak/total (robust against outliers) and
        L1 loss for masked amplitude comparison.
        """
        pred = x["pred"]

        pred = apply_symmetry({"pred": pred})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred_amp = pred[:, 1, :].clone()
        truth_amp = truth[:, 1, :].clone()
        truth_phase = truth[:, 0, :]

        pred_combined = torch.stack([pred_amp, truth_phase], dim=1)

        pred_ifft = get_ifft_torch(pred_combined, amp_phase=True)
        truth_ifft = get_ifft_torch(truth, amp_phase=True)

        hu1 = nn.HuberLoss(delta=8.2558)
        hu2 = nn.HuberLoss(delta=0.0443)
        hu3 = nn.HuberLoss(delta=0.008)
        l1 = nn.L1Loss()

        # --- Peak Loss:  Huber (robust gegen Spikes) ---
        pred_peak = pred_ifft.amax(dim=(-2, -1))
        truth_peak = truth_ifft.amax(dim=(-2, -1))
        loss_peak = hu1(pred_peak, truth_peak)

        # --- Total Flux:  Huber (robust + normalisiert) ---
        n_pixels = pred_ifft.shape[-1] * pred_ifft.shape[-2]
        pred_total = pred_ifft.sum(dim=(-2, -1)) / n_pixels
        truth_total = truth_ifft.sum(dim=(-2, -1)) / n_pixels
        loss_total = hu2(pred_total, truth_total)

        # --- Masked Amplitude:  ---
        truth_max = truth_amp.amax(dim=(-2, -1), keepdim=True)

        thresholds = [truth_max * level for level in self.levels]
        masks = [torch.abs(truth_amp) > thresholds[0]]

        for i, _ in enumerate(thresholds[0:-1]):
            mask = (torch.abs(truth_amp) <= thresholds[i]) & (
                torch.abs(truth_amp) > thresholds[i + 1]
            )
            masks.append(mask)

        masks.append(torch.abs(truth_amp) <= thresholds[-1])

        # --- emoio: ---

        loss = []

        for mask, weight in zip(masks[:-1], self.weights):
            loss.append(hu3(pred_amp[mask] * weight, truth_amp[mask] * weight))

        loss_mask = sum(loss)

        loss_rest = l1(pred_amp[masks[-1]], truth_amp[masks[-1]])

        loss = loss_rest + loss_mask + 100 * loss_peak + 10**5 * loss_total

        print(f"{loss_rest = }, {loss_mask = }, {loss_peak = }, {loss_total = }")
        print(f"{loss_mask = }, {100 * loss_peak = }, {10**5 * loss_total = }")
        print("\n")

        return loss


class ImagFluxEmoio(nn.Module):
    def __init__(self, levels, weights, **kwargs):
        super().__init__()
        if not levels:
            levels = [0.88, 0.75, 0.63, 0.50, 0.38, 0.25, 0.13, 0.05]
        if not weights:
            weights = [2.00, 1.76, 1.50, 1.26, 1.00, 0.76, 0.50, 0.26]
        if not all([isinstance(levels, list), isinstance(weights, list)]):
            raise ValueError(
                "'radii' and 'weights' must be lists, but got "
                f"{type(levels)} and {type(weights)} instead."
            )
        if not len(weights) == len(levels):
            raise ValueError("len(weights) has to be equal to len(levels)")
        self.levels = levels
        self.weights = weights

    def forward(self, x, y):
        """
        Combined loss for amplitude reconstruction with masked weighting.

        Uses Huber loss for peak/total (robust against outliers) and
        L1 loss for masked amplitude comparison.
        """
        pred = x["pred"]

        pred = apply_symmetry({"pred": pred})["pred"]
        truth = apply_symmetry({"y": y})["y"]

        pred_amp = pred[:, 1, :].clone()
        truth_amp = truth[:, 1, :].clone()

        hu3 = nn.HuberLoss(delta=0.008)
        l1 = nn.L1Loss()

        # --- Masked Amplitude:  ---
        truth_max = truth_amp.amax(dim=(-2, -1), keepdim=True)

        thresholds = [truth_max * level for level in self.levels]
        masks = [torch.abs(truth_amp) > thresholds[0]]

        for i, _ in enumerate(thresholds[0:-1]):
            mask = (torch.abs(truth_amp) <= thresholds[i]) & (
                torch.abs(truth_amp) > thresholds[i + 1]
            )
            masks.append(mask)

        masks.append(torch.abs(truth_amp) <= thresholds[-1])

        # --- emoio: ---

        loss = []

        for mask, weight in zip(masks[:-1], self.weights):
            loss.append(hu3(pred_amp[mask] * weight, truth_amp[mask] * weight))

        loss_mask = sum(loss)

        loss_rest = l1(pred_amp[masks[-1]], truth_amp[masks[-1]])

        loss = loss_rest + loss_mask

        print(f"{loss_rest = }, {loss_mask = }")
        print("\n")

        return loss


############################################
############################################
############################################
############################################


def amp_L1_circle_masked(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask1, mask2, mask3 = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius1=50, radius2=100, bs=y.shape[0]
        )
    )

    inp_amp[mask1] *= 10
    tar_amp[mask1] *= 10

    inp_amp[mask2] *= 1.5
    tar_amp[mask2] *= 1.5

    inp_amp[mask3] *= 0.5
    tar_amp[mask3] *= 0.5
    #######

    mean = 0
    std = 0.4

    # gauss = stats.norm.pdf(x, loc=mean, scale=std)

    inp_amp[mask1] *= mean + 3 * std
    tar_amp[mask1] *= mean + 3 * std

    inp_amp[mask2] *= mean + 2 * std
    tar_amp[mask2] *= mean + 2 * std

    inp_amp[mask3] *= mean + 1 * std
    tar_amp[mask3] *= mean + 1 * std

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


################## Amplitude masks #########################################


def amp_L1_masked_r10_w03(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=10, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.3
    tar_amp[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


####


def amp_L1_masked_r16_w01(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=16, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.1
    tar_amp[~mask] *= 0.1

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


def amp_L1_masked_r16_w03(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=16, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.3
    tar_amp[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


def amp_L1_masked_r16_w05(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=16, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.5
    tar_amp[~mask] *= 0.5

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


####


def amp_L1_masked_r32_w01(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=32, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.1
    tar_amp[~mask] *= 0.1

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


def amp_L1_masked_r32_w03(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=32, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.3
    tar_amp[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


def amp_L1_masked_r32_w05(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=32, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.5
    tar_amp[~mask] *= 0.5

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


######


def amp_L1_masked_r64_w01(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=64, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.1
    tar_amp[~mask] *= 0.1

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


def amp_L1_masked_r64_w03(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=64, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.3
    tar_amp[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


def amp_L1_masked_r64_w05(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_amp = pred[:, 0, :]

    y = apply_symmetry({"y": y})["y"]
    tar_amp = y[:, 0, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_amp.shape[-2], inp_amp.shape[-1], radius=64, bs=y.shape[0]
        )
    )

    inp_amp[~mask] *= 0.5
    tar_amp[~mask] *= 0.5

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss = loss_amp

    return loss


############################################################

################## Phase masks #########################################


def phase_L1_masked_r32_w01(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_phase = pred[:, 1, :]

    y = apply_symmetry({"y": y})["y"]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_phase.shape[-2], inp_phase.shape[-1], radius=32, bs=y.shape[0]
        )
    )

    inp_phase[~mask] *= 0.1
    tar_phase[~mask] *= 0.1

    l1 = nn.L1Loss()
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_phase

    return loss


def phase_L1_masked_r32_w03(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_phase = pred[:, 1, :]

    y = apply_symmetry({"y": y})["y"]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_phase.shape[-2], inp_phase.shape[-1], radius=32, bs=y.shape[0]
        )
    )

    inp_phase[~mask] *= 0.3
    tar_phase[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_phase

    return loss


def phase_L1_masked_r32_w05(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_phase = pred[:, 1, :]

    y = apply_symmetry({"y": y})["y"]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_phase.shape[-2], inp_phase.shape[-1], radius=32, bs=y.shape[0]
        )
    )

    inp_phase[~mask] *= 0.5
    tar_phase[~mask] *= 0.5

    l1 = nn.L1Loss()
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_phase

    return loss


#######


def phase_L1_masked_r64_w03(x, y):
    pred = x["pred"]
    pred = apply_symmetry({"pred": pred})["pred"]
    inp_phase = pred[:, 1, :]

    y = apply_symmetry({"y": y})["y"]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(
        create_circular_mask(
            inp_phase.shape[-2], inp_phase.shape[-1], radius=64, bs=y.shape[0]
        )
    )

    inp_phase[~mask] *= 0.3
    tar_phase[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_phase

    return loss


########################### pre, rest #################################


def splitted_L1(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]
    inp_phase = pred[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase

    return loss


def beta_nll_loss(x: torch.tensor, y: torch.tensor, beta: float = 0.5):
    """Compute beta-NLL loss

    Parameters
    ----------
    x : :func:`torch.tensor`
        Prediction of the model.
    y : :func:`torch.tensor`
        Ground truth.
    beta : float
        Parameter from range [0, 1] controlling relative
        weighting between data points, where "0" corresponds to
        high weight on low error points and "1" to an equal weighting.

    Returns
    -------
    float : Loss per batch element of shape B
    """
    pred = x["pred"]
    pred_amp = pred[:, 0, :]
    pred_phase = pred[:, 2, :]
    mean = torch.stack([pred_amp, pred_phase], axis=1)

    unc_amp = pred[:, 1, :]
    unc_phase = pred[:, 3, :]
    variance = torch.stack([unc_amp, unc_phase], axis=1)

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]
    target = torch.stack([tar_amp, tar_phase], axis=1)

    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * variance.detach() ** beta

    return loss.mean()


def mse(x, y):
    pred = x["pred"]
    mse = nn.MSELoss()
    loss = mse(pred, y)

    return loss


def jet_seg(x, y):
    pred = x["pred"]

    # weight components farer outside more
    loss_l1_weighted = 0
    for i in range(pred.shape[1]):
        loss_l1_weighted += l1(pred[:, i], y[:, i]) * (i + 1)

    return loss_l1_weighted
