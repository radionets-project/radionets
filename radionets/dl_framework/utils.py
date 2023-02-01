import torch
import numpy as np


def round_odd(x):
    """Rounds a float to the next higher, odd number and returns an int.

    Parameters
    ----------
    x : float
        value to be rounded

    Returns
    -------
    int
        next higher odd value
    """
    return int(np.ceil(x) // 2 * 2 + 1)


def make_padding(kernel_size, stride, dilation):
    """Returns the padding size under the condition, that the image size
    stays the same.

    Parameters
    ----------
    kernel_size : int
        size of the kernel
    stride : int
        stride of the convolution
    dilation : int
        dilation of the convolution

    Returns
    -------
    int
        appropiate padding size for given parameters
    """
    return -((-kernel_size - (kernel_size - 1) * (dilation - 1)) // stride + 1) // 2


def _maybe_item(t):
    t = t.value
    return t.item() if isinstance(t, torch.Tensor) and t.numel() == 1 else t


def get_ifft_torch(array, amp_phase=False, scale=False, uncertainty=False):
    if len(array.shape) == 3:
        array = array.unsqueeze(0)
    if amp_phase:
        if scale:
            amp = 10 ** (10 * array[:, 0] - 10) - 1e-10
        else:
            amp = array[:, 0]
        if uncertainty:
            a = amp * torch.cos(array[:, 2])
            b = amp * torch.sin(array[:, 2])
        else:
            a = amp * torch.cos(array[:, 1])
            b = amp * torch.sin(array[:, 1])
        compl = a + b * 1j
    else:
        compl = array[:, 0] + array[:, 1] * 1j
    if compl.shape[0] == 1:
        compl = compl.squeeze(0)
    return torch.abs(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(compl))))
