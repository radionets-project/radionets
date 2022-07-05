import torch
from typing import Iterable
import numpy as np
from torch import nn


def listify(o):
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def get_batch(dl, learn):
    learn.xb, learn.yb = next(iter(dl))
    for cb in learn.cbs:
        cb.set_runner(learn)
    learn("begin_batch")
    return learn.xb, learn.yb


def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)


class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)  # bool mask
                return [o for m, o in zip(idx, self.items) if m]
            return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del self.items[i]

    def __repr__(self):
        res = f"{self.__class__.__name__} \
                ({len(self)} items)\n{self.items[:10]}"
        if len(self) > 10:
            res = res[:-1] + "...]"
        return res


def children(m):
    "returns the children of m as a list"
    return list(m.children())


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


def get_ifft_torch(array, amp_phase=False, scale=False):
    if len(array.shape) == 3:
        array = array.unsqueeze(0)
    if amp_phase:
        if scale:
            amp = 10 ** (10 * array[:, 0] - 10) - 1e-10

        a = amp * torch.cos(array[:, 1])
        b = amp * torch.sin(array[:, 1])
        compl = a + b * 1j
    else:
        compl = array[:, 0] + array[:, 1] * 1j
    if compl.shape[0] == 1:
        compl = compl.squeeze(0)
    return torch.abs(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(compl))))
