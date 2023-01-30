import torch

from radionets.dl_framework.utils import (
    overall_iou,
)
from radionets.evaluation.utils import (
    yolo_apply_nms,
)


def iou_YOLOv6(pred, target):
    """Return the overall IoU during YOLOv6 training

    Parameters
    ----------
    pred: list
        The output layers of YOLOv6
    target: ndarray
        Array with target boxes of shape (bs, n_components, 8)
        8: [amplitude, x, y, sx, sy, y_rotation, z_rotation, beta]

    Returns
    -------
    iou: float
        Mean Intersection over Union of all boxes after nms
    """
    bs = target.shape[0]
    strides_head = torch.tensor([4, 8, 16])

    target[..., 3:5] *= 2  # increased box sizes (same as in loss function)

    pred_nms = yolo_apply_nms(pred=pred, strides=strides_head)

    amp_threshold = 0.01  # only take components above this amplitude into account
    ious = torch.zeros(bs)
    for i in range(bs):
        target_boxes = target[i, :, 1:5][target[i, :, 0] > amp_threshold]
        ious[i] = overall_iou(pred_nms[i][:, :4], target_boxes)
    iou = torch.mean(ious)
    return iou
