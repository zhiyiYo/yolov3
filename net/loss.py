# coding: utf-8
from typing import Tuple, List

import torch
from torch import Tensor, nn
from utils.box_utils import match


class YoloLoss(nn.Module):
    """ 损失函数 """

    def __init__(self, anchors: list, n_classes: int, image_size: int, overlap_thresh=0.5,
                 lambda_box=2.5, lambda_obj=1, lambda_noobj=0.5, lambda_cls=1):
        """
        Parameters
        ----------
        anchors: list of shape `(3, n_anchors, 2)`
            先验框列表

        n_classes: int
            类别数

        image_size: int
            输入神经网络的图片大小

        overlap_thresh: float
            视为忽视样例的 IOU 阈值

        lambda_box, lambda_obj, lambda_noobj, lambda_cls: float
            权重参数
        """
        super().__init__()
        self.anchors = anchors
        self.n_classes = n_classes
        self.image_size = image_size
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.overlap_thresh = overlap_thresh
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, preds: Tuple[Tensor], targets: List[Tensor]):
        """
        Parameters
        ----------
        preds: Tuple[Tensor]
            Yolo 神经网络输出的各个特征图，每个特征图的维度为 `(N, (n_classes+5)*n_anchors, H, W)`

        targets: List[Tensor]
            标签数据，每个标签张量的维度为 `(N, n_objects, 5)`，最后一维的第一个元素为类别，剩下为边界框 `(cx, cy, w, h)`

        Returns
        -------
        loc_loss: Tensor
            定位损失

        conf_loss: Tensor
            置信度损失

        cls_loss: Tensor
            分类损失
        """
        loc_loss = 0
        conf_loss = 0
        cls_loss = 0

        for anchors, pred in zip(self.anchors, preds):
            N, _, img_h, img_w = pred.shape
            n_anchors = len(anchors)

            # 调整特征图尺寸，方便索引
            pred = pred.view(N, n_anchors, self.n_classes+5,
                             img_h, img_w).permute(0, 1, 3, 4, 2).contiguous()

            # 获取特征图最后一个维度的每一部分
            x = pred[..., 0].sigmoid()
            y = pred[..., 1].sigmoid()
            w = pred[..., 2]
            h = pred[..., 3]
            conf = pred[..., 4].sigmoid()
            cls = pred[..., 5:].sigmoid()

            # 匹配边界框
            step_h = self.image_size/img_h
            step_w = self.image_size/img_w
            anchors = [[i/step_w, j/step_h] for i, j in anchors]
            p_mask, n_mask, t, scale = match(
                anchors, targets, img_h, img_w, self.n_classes, self.overlap_thresh)

            p_mask = p_mask.to(pred.device)
            n_mask = n_mask.to(pred.device)
            t = t.to(pred.device)
            scale = scale.to(pred.device)

            # 定位损失
            x_loss = self.mse_loss(x*p_mask*scale, t[..., 0]*p_mask*scale)
            y_loss = self.mse_loss(y*p_mask*scale, t[..., 1]*p_mask*scale)
            w_loss = self.mse_loss(w*p_mask*scale, t[..., 2]*p_mask*scale)
            h_loss = self.mse_loss(h*p_mask*scale, t[..., 3]*p_mask*scale)
            loc_loss += (x_loss + y_loss + w_loss + h_loss)*self.lambda_box

            # 置信度损失
            conf_loss += self.bce_loss(conf*p_mask, p_mask)*self.lambda_obj + \
                self.bce_loss(conf*n_mask, 0*n_mask)*self.lambda_noobj

            # 分类损失
            m = p_mask == 1
            cls_loss += self.bce_loss(cls[m], t[..., 5:][m])*self.lambda_cls

        return loc_loss, conf_loss, cls_loss
