import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["get_ifft", "apply_symmetry"]


def get_ifft(image, amp_phase=False, scale=False, uncertainty=False):
    """Get inverse FFT of provided image data.

    Returns
    -------
    torch.tensor
        Inverse FFT of provided image data.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    if amp_phase:
        amp = (
            10 ** (10 * image[..., 0, :, :] - 10) - 1e-10
            if scale
            else image[..., 0, :, :]
        )

        index = 2 if uncertainty else 1
        a = amp * torch.cos(image[..., index, :, :])
        b = amp * torch.sin(image[..., index, :, :])

        compl = a + b * 1j
    else:
        compl = image[..., 0, :, :] + image[..., 1, :, :] * 1j

    if compl.shape[0] == 1:
        compl = compl.squeeze(0)

    return torch.abs(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(compl))))


def apply_symmetry(image, uncertainty: bool = False) -> torch.Tensor:
    """Applies symmetry operations on an array.

    This follows Figure 5.3 in http://dx.doi.org/10.17877/DE290R-24834

    Parameters
    ----------
    image : array_like
        Input array of half images.
    uncertainty : bool, optional
        Whether image data contains uncertainty data.
        Default: False

    Returns
    -------
    symmetrical : torch.tensor
        Torch tensor containing the symmetrical image.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    *_, H, W = image.shape

    # Assume images are square; get target height from full width
    # NOTE: This may have to be changed should we allow different
    # aspect ratios in the future
    half_width = W // 2

    # Calculate the overlap from difference of half_image and
    # input height H, so we do not need to pass it anymore
    overlap = H - half_width

    pad_bottom = half_width - overlap
    full_image = F.pad(image, pad=(0, 0, 0, pad_bottom), mode="constant", value=0)

    upper_half = image[..., :half_width, :]

    # flip along image axes W and H to rotate image by 180 deg
    rotated = upper_half.flip(-2, -1)

    # Shift columns to the right by 1 to account for central pixel
    # and drop last row
    lower_half = torch.roll(rotated, shifts=1, dims=-1)
    lower_half = lower_half[..., :-1, :]

    if not uncertainty:
        lower_half[..., 1, :, :] *= -1

    full_image[..., half_width + 1 :, :] = lower_half

    return full_image
