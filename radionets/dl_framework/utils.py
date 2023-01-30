import math
import numpy as np
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import box
from shapely.ops import unary_union
import torch


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


def getAffinityMatrix(coordinates, k: int = None):
    """Calculate affinity matrix based on input coordinates matrix and the number
    of nearest neighbours.

    Apply local scaling based on the k nearest neighbour.

    Used in spectralClustering (dl_framework/clustering.py).

    Parameters
    ----------
    coordinates: 2d-array
        data points of shape (n, 2)
    k: int
        k nearest neighbour, square root - 1 of number of coordinates it not provided

    Returns
    -------
    affinity_matrix: 2d-array
        affinity matrix of shape (n, n)
    """
    dists = squareform(pdist(coordinates))

    if not k:
        k = int(np.rint(np.sqrt(len(coordinates))))

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


def bbox_iou(box1, box2, xywh=True, iou_type="ciou", eps=1e-7):
    """Returns Intersection over Union (IoU) of box1(n,4) to box2(n,4)
    https://github.com/nirbarazida/YOLOv6/blob/main/yolov6/utils/figure_iou.py
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1, box2
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    if not torch.is_tensor(b1_x1):
        b1_x1 = torch.tensor(b1_x1)
    if not torch.is_tensor(b1_y1):
        b1_y1 = torch.tensor(b1_y1)
    if not torch.is_tensor(b1_x2):
        b1_x2 = torch.tensor(b1_x2)
    if not torch.is_tensor(b1_y2):
        b1_y2 = torch.tensor(b1_y2)
    if not torch.is_tensor(b2_x1):
        b2_x1 = torch.tensor(b2_x1)
    if not torch.is_tensor(b2_y1):
        b2_y1 = torch.tensor(b2_y1)
    if not torch.is_tensor(b2_x2):
        b2_x2 = torch.tensor(b2_x2)
    if not torch.is_tensor(b2_y2):
        b2_y2 = torch.tensor(b2_y2)
    if not torch.is_tensor(w1):
        w1 = torch.tensor(w1)
    if not torch.is_tensor(h1):
        h1 = torch.tensor(h1)
    if not torch.is_tensor(w2):
        w2 = torch.tensor(w2)
    if not torch.is_tensor(h2):
        h2 = torch.tensor(h2)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if iou_type in ["ciou", "diou", "giou"]:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c_area = cw * ch + eps  # convex area
        if iou_type in [
            "diou",
            "ciou",
        ]:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center dist ** 2
            if (
                iou_type == "ciou"
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def overall_iou(boxes1, boxes2):
    """Returns IoU of all boxes of shape [n, 4 (xywh)] for one image"""
    box1_objects = []
    box2_objects = []
    for box1 in xywh2xyxy(boxes1):
        box1_objects.append(box(*box1))
    for box2 in xywh2xyxy(boxes2):
        box2_objects.append(box(*box2))

    box1_union = unary_union(box1_objects)
    box2_union = unary_union(box2_objects)

    union = box1_union.union(box2_union).area
    inter = box1_union.intersection(box2_union).area
    iou = inter / union
    return iou


def xywh2xyxy(x):
    # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
