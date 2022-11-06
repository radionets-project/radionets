import torch
from typing import Iterable
import numpy as np
from torch import nn
from scipy.spatial.distance import pdist, squareform


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


def decode_yolo_box(x, stride):
    """Decode one feature map of YOLO model to match input size

    Parameters
    ----------
    x: nd-array
        prediction of model (bs, (1), n, n, 4 or more)
    stride: int
        stride used in model

    Returns
    -------
    x: nd-array
        transformed prediction, same shape as input
    """
    x = torch.clone(x)  # avoid changes of inplace operation
    d = x.device
    ny, nx = x.shape[-3:-1]

    yv, xv = torch.meshgrid(
        [torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing="ij"
    )
    grid = torch.stack((xv, yv), 2).view(1, ny, nx, 2).float()
    x[..., 0:2] = (x[..., 0:2] + grid) * stride.to(d)  # xy
    x[..., 2:4] = torch.exp(x[..., 2:4]) * stride.to(d)  # wh, org. YOLOv6

    return x


def build_target_yolo(y, shape, stride):
    """Building the target for training

    Parameters
    ----------
    y: ndarray
        truth from simulation
        (amplitude, x, y, width, height, comp-rotation, jet-rotation, velocity)
    shape: ndarray
        shape of one output feature map [bs, a, ny, nx, 6]
    stride: int
        stride for this feature map

    Returns
    -------
    target: ndarray
        target for training
    """
    # initialize target
    target = torch.zeros(shape)
    # get indicies for target objectness
    target_idx = (y[..., 1:3] / stride).type(torch.LongTensor)

    for i in range(shape[0]):  # for each batch
        anchors = torch.zeros((shape[2], shape[3])).type(torch.LongTensor)

        for j in range(y.shape[1]):  # for each target component

            if y[i, j, 0] > 0:  # only assign when amplitude is larger 0
                ny = target_idx[i, j, 1]
                nx = target_idx[i, j, 0]
                anchor = anchors[ny, nx]

                if anchor < shape[1]:  # only assign available anchors
                    target[i, anchor, ny, nx, 0:4] = y[i, j, 1:5]
                    target[i, anchor, ny, nx, 4] = 1
                    target[i, anchor, ny, nx, 5] = y[i, j, 5]
                anchors[ny, nx] += 1

    return target


def get_ifft_torch(array, amp_phase=False, scale=False):
    if len(array.shape) == 3:
        array = array.unsqueeze(0)
    if amp_phase:
        if scale:
            amp = 10 ** (10 * array[:, 0] - 10) - 1e-10
        else:
            amp = array[:, 0]

        a = amp * torch.cos(array[:, 1])
        b = amp * torch.sin(array[:, 1])
        compl = a + b * 1j
    else:
        compl = array[:, 0] + array[:, 1] * 1j
    if compl.shape[0] == 1:
        compl = compl.squeeze(0)
    return torch.abs(torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(compl))))


def getAffinityMatrix(coordinates, k: int = 7):
    """Calculate affinity matrix based on input coordinates matrix and the number
    of nearest neighbours.

    Apply local scaling based on the k nearest neighbour.

    Used in spectralClustering (dl_framework/clustering.py).

    Parameters
    ----------
    coordinates: 2d-array
        data points of shape (n, 2)
    k: int
        k nearest neighbour

    Returns
    -------
    affinity_matrix: 2d-array
        affinity matrix of shape (n, n)
    """
    dists = squareform(pdist(coordinates))

    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T

    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale

    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0

    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def normalizeAffinityMatrix(affinity_matrix):
    """Normalization of affinity matrix.

    Used in spectralClustering (dl_framework/clustering.py).

    Parameters
    ----------
    affinity_matrix: 2d-array
        affinity matrix to be normalized

    Returns
    -------
    L: 2d-array
        normalized affinity matrix
    """
    D = np.diag(np.sum(affinity_matrix, axis=1))
    D_inv = np.sqrt(np.linalg.inv(D))
    L = np.dot(D_inv, np.dot(affinity_matrix, D_inv))
    return L
