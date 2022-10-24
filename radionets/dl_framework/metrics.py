import math
from shapely.geometry import box
from shapely.ops import unary_union
import torch
from radionets.dl_framework.utils import decode_yolo_box
from radionets.evaluation.utils import non_max_suppression, xywh2xyxy


def iou_YOLOv6(pred, target):
    """Return the overall IoU during YOLOv6 training
    
    Parameters
    ----------
    pred: list
        The output layers of YOLOv6
    target: ndarray
        Array with target boxes of shape (bs, n_components, 7)
        7: [amplitude, x, y, sx, sy, z_rotation, beta]

    Returns
    -------
    iou: float
        Mean Intersection over Union of all boxes after nms
    """
    bs = target.shape[0]
    strides_head = torch.tensor([8, 16, 32])

    target[..., 3:5] *= 2   # increased box sizes (same as in loss function)

    preds = []
    for i, pred_map in enumerate(pred):
        # squeeze, because anchors are not implemented in loss yet
        pred_map = pred_map.squeeze(1)

        # somehow the box is already decoded (maybe from an inplace operation in the loss and fast.ai uses it)
        pred_map = decode_yolo_box(pred_map, strides_head[i])
        pred_map[..., 4] = torch.sigmoid(pred_map[..., 4])

        preds.append(pred_map.reshape(bs, -1, 6))
    preds = torch.cat(preds, dim=1)

    pred_nms = non_max_suppression(preds)

    amp_threshold = 0.02    # only take components above this amplitude into account
    ious = torch.zeros(bs)
    for i in range(bs):
        target_boxes = target[i, :, 1:5][target[i, :, 0] > amp_threshold]
        ious[i] = overall_iou(pred_nms[i][:, :4], target_boxes)
    iou = torch.mean(ious)
    return iou


def bbox_iou(box1, box2, xywh=True, iou_type='ciou', eps=1e-7):
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

    if not torch.is_tensor(b1_x1): b1_x1 = torch.tensor(b1_x1)
    if not torch.is_tensor(b1_y1): b1_y1 = torch.tensor(b1_y1)
    if not torch.is_tensor(b1_x2): b1_x2 = torch.tensor(b1_x2)
    if not torch.is_tensor(b1_y2): b1_y2 = torch.tensor(b1_y2)
    if not torch.is_tensor(b2_x1): b2_x1 = torch.tensor(b2_x1)
    if not torch.is_tensor(b2_y1): b2_y1 = torch.tensor(b2_y1)
    if not torch.is_tensor(b2_x2): b2_x2 = torch.tensor(b2_x2)
    if not torch.is_tensor(b2_y2): b2_y2 = torch.tensor(b2_y2)
    if not torch.is_tensor(w1): w1 = torch.tensor(w1)
    if not torch.is_tensor(h1): h1 = torch.tensor(h1)
    if not torch.is_tensor(w2): w2 = torch.tensor(w2)
    if not torch.is_tensor(h2): h2 = torch.tensor(h2)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if iou_type in ['ciou', 'diou', 'giou']:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c_area = cw * ch + eps  # convex area
        if iou_type in ['diou', 'ciou']:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if iou_type == 'ciou':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def overall_iou(boxes1, boxes2):
    """Returns IoU of all boxes of shape [n, 4 (xywh)] for one image
    """
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
