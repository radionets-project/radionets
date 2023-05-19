import torch

from radionets.dl_framework.utils import overall_iou
from radionets.evaluation.utils import yolo_apply_nms


def iou_YOLOv6(pred, target):
    """Return the overall IoU during YOLOv6 training

    Parameters
    ----------
    pred: list
        list of feature maps, each of shape (bs, a, ny, nx, 6)
    target: ndarray
        Array with target boxes of shape (bs, n_components, 8)
        8: [amplitude, x, y, sx, sy, y_rotation, z_rotation, beta]

    Returns
    -------
    iou: float
        Mean Intersection over Union of all boxes after nms
    """
    if torch.is_tensor(target):
        target = target.clone()
    else:
        target = target.copy()

    bs = target.shape[0]
    # strides_head = torch.tensor([2, 4, 8])
    strides_head = torch.tensor([4, 8, 16])

    target[..., 3:5] *= 2  # increased box sizes (same as in loss function)

    pred_nms = yolo_apply_nms(pred=pred, strides=strides_head)

    amp_threshold = 0.01  # only take components above this amplitude into account
    ious = torch.zeros(bs)
    for i in range(bs):
        target_boxes = target[i, :, 1:5][target[i, :, 0] > amp_threshold]
        if len(pred_nms[i]) == 0:  # no predicted box -> no intersection over union
            ious[i] = 0
        else:
            ious[i] = overall_iou(pred_nms[i][:, :4], target_boxes)
    iou = torch.mean(ious)
    return iou


def binary_accuracy(x, y):
    """Return the accuracy of binary data (classes are 0 or 1).

    Parameters
    ----------
    x: tensor
        tensor of shape (bs, 1)
    y: ndarray
        Array with target boxes of shape (bs, n_components, 8)
        8: [amplitude, x, y, sx, sy, y_rotation, z_rotation, beta]

    Returns
    -------
    accuracy: float
        accuracy of the classes
    """
    threshold = 0.5
    pred = x > threshold

    n_components = y.shape[1]
    amps = y[:, -int((n_components - 1) / 2) :, 0]
    amps_summed = torch.sum(amps, axis=1)
    target = amps_summed > 0

    accuracy = torch.mean((pred == target).float())
    return accuracy
