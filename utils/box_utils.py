# coding: utf-8
from typing import Union, List
import cmapy
import math

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from PIL import Image, ImageDraw, ImageFont


def log_sum_exp(x: Tensor):
    """ 计算 log(∑exp(xi))

    Parameters
    ----------
    x: Tensor of shape `(n, n_classes)`
    """
    x_max = x.detach().max()
    out = torch.log(torch.sum(torch.exp(x-x_max), dim=1, keepdim=True))+x_max
    return out


def jaccard_overlap(prior: Tensor, bbox: Tensor):
    """ 计算预测的先验框和边界框真值的交并比，四个坐标为 `(xmin, ymin, xmax, ymax)`

    Parameters
    ----------
    prior: Tensor of shape `(A, 4)`
        先验框

    bbox: Tensor of shape  `(B, 4)`
        边界框真值

    Returns
    -------
    iou: Tensor of shape `(A, B)`
        交并比
    """
    A = prior.size(0)
    B = bbox.size(0)

    # 将先验框和边界框真值的 xmax、ymax 以及 xmin、ymin进行广播使得维度一致，(A, B, 2)
    # 再计算 xmax 和 ymin 较小者、xmin 和 ymin 较大者，W=xmax较小-xmin较大，H=ymax较小-ymin较大
    xy_max = torch.min(prior[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bbox[:, 2:].broadcast_to(A, B, 2))
    xy_min = torch.max(prior[:, :2].unsqueeze(1).expand(A, B, 2),
                       bbox[:, :2].broadcast_to(A, B, 2))

    # 计算交集面积
    inter = (xy_max-xy_min).clamp(min=0)
    inter = inter[:, :, 0]*inter[:, :, 1]

    # 计算每个矩形的面积
    area_prior = ((prior[:, 2]-prior[:, 0]) *
                  (prior[:, 3]-prior[:, 1])).unsqueeze(1).expand(A, B)
    area_bbox = ((bbox[:, 2]-bbox[:, 0]) *
                 (bbox[:, 3]-bbox[:, 1])).broadcast_to(A, B)

    return inter/(area_prior+area_bbox-inter)


def jaccard_overlap_numpy(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和其他边界框的交并比

    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框

    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    # 计算 iou
    iou = inter/(area_box+area_boxes-inter)  # type: np.ndarray
    return iou


def center_to_corner(boxes: Tensor):
    """ 将 `(cx, cy, w, h)` 形式的边界框变换为 `(xmin, ymin, xmax, ymax)` 形式的边界框

    Parameters
    ----------
    boxes: Tensor of shape `(n, 4)`
        边界框
    """
    return torch.cat((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), dim=1)


def corner_to_center(boxes: Tensor):
    """ 将 `(xmin, ymin, xmax, ymax)` 形式的边界框变换为 `(cx, cy, w, h)` 形式的边界框

    Parameters
    ----------
    boxes: Tensor of shape `(n, 4)`
        边界框
    """
    return torch.cat(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]), dim=1)


def encode(prior: Tensor, matched_bbox: Tensor, variance: tuple):
    """ 编码先验框和与边界框之间的偏置量

    Parameters
    ----------
    prior: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    matched_bbox: Tensor of shape `(n_priors, 4)`
        匹配到的边界框，坐标形式为 `(xmin, ymin, xmax, ymax)`

    variance: Tuple[float, float]
        先验框方差

    Returns
    -------
    g: Tensor of shape `(n_priors, 4)`
        编码后的偏置量
    """
    matched_bbox = corner_to_center(matched_bbox)
    g_cxcy = (matched_bbox[:, :2]-prior[:, :2]) / (variance[0]*prior[:, 2:])
    g_wh = torch.log(matched_bbox[:, 2:]/prior[:, 2:]+1e-5) / variance[1]
    return torch.cat((g_cxcy, g_wh), dim=1)


def decode(loc: Tensor, prior: Tensor, variance: tuple):
    """ 根据偏移量和先验框位置解码出边界框的位置

    Parameters
    ----------
    loc: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    prior: Tensor of shape `(n_priors, 4)`
        先验框，坐标形式为 `(cx, cy, w, h)`

    variance: Tuple[float, float]
        先验框方差

    Returns
    -------
    g: Tensor of shape `(n_priors, 4)`
        边界框的位置
    """
    bbox = torch.cat((
        prior[:, :2] + prior[:, 2:] * variance[0] * loc[:, :2],
        prior[:, 2:] * torch.exp(variance[1] * loc[:, 2:])), dim=1)
    bbox = center_to_corner(bbox)
    return bbox


def match(anchors: list, target: List[Tensor], h: int, w: int, n_classes: int, overlap_thresh=0.5):
    """ 匹配先验框和边界框真值

    Parameters
    ----------
    anchors: List[Tuple[float, float]]
        根据特征图的大小进行过缩放的先验框

    target: List[Tensor]
        标签，每个元素的最后一个维度的第一个元素为类别，剩下四个为 `(cx, cy, w, h)`

    h: int
        特征图的高度

    w: int
        特征图的宽度

    n_classes: int
        类别数

    overlap_thresh: float
        忽视样例的 IOU 阈值

    Returns
    -------
    p_mask: Tensor of shape `(N, n_anchors, H, W)`
        正例遮罩

    n_mask: Tensor of shape `(N, n_anchors, H, W)`
        反例遮罩

    t: Tensor of shape `(N, n_anchors, H, W, n_classes+5)`
        标签
    """
    N = len(target)
    n_anchors = len(anchors)

    # 初始化返回值
    p_mask = torch.zeros(N, n_anchors, h, w)
    n_mask = torch.ones(N, n_anchors, h, w)
    t = torch.zeros(N, n_anchors, h, w, n_classes+5)

    # 匹配先验框和边界框
    anchors = np.hstack((np.zeros((n_anchors, 2)), np.array(anchors)))
    for i in range(N):
        for j in range(target[i].size(0)):
            # 获取标签数据
            cx, gw = target[i][j, [1, 3]]*w
            cy, gh = target[i][j, [2, 4]]*h

            # 获取单元格的坐标
            gj, gi = int(cx), int(cy)

            # 计算边界框和先验框的交并比
            bbox = np.array([0, 0, gw, gh])
            iou = jaccard_overlap_numpy(bbox, anchors)

            # 标记出正例和反例
            index = np.argmax(iou)
            p_mask[i, index, gi, gj] = 1
            n_mask[i, iou > overlap_thresh, gi, gj] = 0

            # 计算标签值
            t[i, index, gi, gj, 0] = cx-gj
            t[i, index, gi, gj, 1] = cx-gi
            t[i, index, gi, gj, 2] = math.log(gw/anchors[index, 2]+1e-16)
            t[i, index, gi, gj, 3] = math.log(gh/anchors[index, 3]+1e-16)
            t[i, index, gi, gj, 4] = 1
            t[i, index, gi, gj, 5+target[i][j, 0]] = 1

    return p_mask, n_mask, t


def nms(boxes: Tensor, scores: Tensor, overlap_thresh=0.5, top_k=200):
    """ 非极大值抑制，去除多余的预测框

    Parameters
    ----------
    boxes: Tensor of shape `(n_priors, 4)`
        预测框，坐标形式为 `(xmin, ymin, xmax, ymax)`

    scores: Tensor of shape `(n_priors, )`
        某个类的每个先验框的置信度

    overlap_thresh: float
        IOU 阈值，大于阈值的部分预测框会被移除，值越小保留的框越多

    top_k: int
        保留的预测框个数上限

    Returns
    -------
    indexes: LongTensor of shape `(n, )`
        保留的边界框的索引
    """
    keep = []
    if boxes.numel() == 0:
        return torch.LongTensor(keep)

    # 每个预测框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2-x1)*(y2-y1)

    # 对分数进行降序排序并截取前 top_k 个索引
    _, indexes = scores.sort(dim=0, descending=True)
    indexes = indexes[:top_k]

    while indexes.numel():
        i = indexes[0]
        keep.append(i)

        # 最后一个索引时直接退出循环
        if indexes.numel() == 1:
            break

        # 其他的预测框和当前预测框的交集
        right = x2[indexes].clamp(max=x2[i].item())
        left = x1[indexes].clamp(min=x1[i].item())
        bottom = y2[indexes].clamp(max=y2[i].item())
        top = y1[indexes].clamp(min=y1[i].item())
        inter = ((right-left)*(bottom-top)).clamp(min=0)

        # 计算 iou
        iou = inter/(area[i]+area[indexes]-inter)

        # 保留 iou 小于阈值的边界框，自己和自己的 iou 为 1
        indexes = indexes[iou < overlap_thresh]

    return torch.LongTensor(keep)


def draw(image: Union[ndarray, Image.Image], bbox: ndarray, label: ndarray, conf: ndarray = None) -> Image.Image:
    """ 在图像上绘制边界框和标签

    Parameters
    ----------
    image: `~np.ndarray` of shape `(H, W, 3)` or `~PIL.Image.Image`
        RGB 图像

    bbox: `~np.ndarray` of shape `(n_objects, 4)`
        边界框

    label: Iterable of shape `(n_objects, )`
        标签

    conf: Iterable of shape `(n_objects, )`
        置信度
    """
    bbox = bbox.astype(np.int)

    if isinstance(image, ndarray):
        image = Image.fromarray(image.astype(np.uint8))  # type:Image.Image

    image_draw = ImageDraw.Draw(image, 'RGBA')
    font = ImageFont.truetype('resource/font/msyh.ttc', size=13)

    label_unique = np.unique(label).tolist()
    color_indexes = np.linspace(0, 255, len(label_unique), dtype=int)

    for i in range(bbox.shape[0]):
        x1 = max(0, bbox[i, 0])
        y1 = max(0, bbox[i, 1])
        x2 = min(image.width-1, bbox[i, 2])
        y2 = min(image.height-1, bbox[i, 3])

        # 选择颜色
        class_index = label_unique.index(label[i])
        color = to_hex_color(cmapy.color(
            'viridis', color_indexes[class_index], True))

        # 绘制方框
        image_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 绘制标签
        y1_ = y1 if y1-23 < 0 else y1-23
        y2_ = y1 if y1_ < y1 else y1+23
        text = label[i] if conf is None else f'{label[i]} | {conf[i]:.2f}'
        l = font.getlength(text) + 3
        image_draw.rectangle([x1, y1_, x1+l, y2_],
                             fill=color+'75', outline=color+'DD')
        image_draw.text([x1+2, y1_+2], text=text,
                        font=font, embedded_color=color)

    return image


def to_hex_color(color):
    """ 将颜色转换为 16 进制 """
    color = [hex(c)[2:].zfill(2) for c in color]
    return '#'+''.join(color)
