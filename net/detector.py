# coding:utf-8
from typing import Tuple

import torch
from utils.box_utils import decode, nms


class Detector:
    """ 探测器 """

    def __init__(self, anchors: list, image_size: int, n_classes: int, top_k=100, conf_thresh=0.25, nms_thresh=0.45):
        """
        Parameters
        ----------
        anchors: list of shape `(3, n_anchors, 2)`
            先验框

        image_size: int
            图片尺寸

        n_classes: int
            类别数

        top_k: int
            一张图片中预测框数量的上限

        conf_thresh: float
            置信度阈值

        nms_thresh: float
            nms 操作中 iou 的阈值，越大保留的预测框越多
        """
        self.top_k = top_k
        self.anchors = anchors
        self.n_classes = n_classes
        self.image_size = image_size
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

    def __call__(self, preds: Tuple[torch.Tensor]):
        """ 对神经网络输出的结果进行处理

        Parameters
        ----------
        preds: Tuple[Tensor]
            神经网络输出的三个特征图

        Returns
        -------
        out: Tensor of shape `(N, n_classes, top_k, 5)`
            检测结果，最后一个维度的第一个元素为置信度，后四个元素为边界框 `(cx, cy, w, h)`
        """
        N = preds[0].size(0)

        # 解码
        batch_pred = []
        for pred, anchors in zip(preds, self.anchors):
            pred_ = decode(pred, anchors, self.n_classes, self.image_size)

            # 展平预测框，shape: (N, n_anchors*H*W, 5)
            batch_pred.append(pred_.view(N, -1, self.n_classes+5))

        batch_pred = torch.cat(batch_pred, dim=1)

        # 非极大值抑制
        out = torch.zeros(N, self.n_classes, self.top_k, 5)
        for i in range(N):
            bbox = batch_pred[i, :, :4]
            conf = batch_pred[i, :, 4]

            # 过滤掉置信度过小的预测框（不包含物体）
            mask = conf > self.conf_thresh
            conf = conf[mask]
            boxes = bbox[mask]

            if conf.numel() == 0:
                continue

            # 取出每一个类别的预测框
            for c in range(self.n_classes):
                score = batch_pred[i, :, 5+c][mask]
                indexes = nms(boxes, score, self.nms_thresh, self.top_k)
                out[i, c, :len(indexes)] = torch.cat(
                    (score[indexes].unsqueeze(1), boxes[indexes]), dim=1)

        return out
