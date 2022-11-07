import torch

from radionets.dl_framework.utils import (
    decode_yolo_box,
    overall_iou,
)
from radionets.evaluation.utils import (
    non_max_suppression,
    objectness_mapping,
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
    strides_head = torch.tensor([8, 16, 32])

    target[..., 3:5] *= 2  # increased box sizes (same as in loss function)

    # use mean of all objectness map
    obj_map = objectness_mapping(pred)
    # use boxes from prediction of highest resolution
    boxes = decode_yolo_box(pred[0], strides_head[0])

    boxes[..., 4] = torch.tensor(obj_map).to(boxes.device)

    pred_nms = non_max_suppression(
        boxes.reshape(bs, -1, 6), obj_thres=boxes[..., 4].max() / 5
    )

    amp_threshold = 0.02  # only take components above this amplitude into account
    ious = torch.zeros(bs)
    for i in range(bs):
        target_boxes = target[i, :, 1:5][target[i, :, 0] > amp_threshold]
        ious[i] = overall_iou(pred_nms[i][:, :4], target_boxes)
    iou = torch.mean(ious)
    return iou
