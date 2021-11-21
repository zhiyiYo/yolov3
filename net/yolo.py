# coding:utf-8
import os
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from torch import nn
from PIL import Image

from .detector import Detector
from utils.box_utils import draw
from utils.augmentation_utils import ToTensor


class ConvBlock(nn.Module):
    """ 卷积块 """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.features(x)


class ResidualUnit(nn.Module):
    """ 残差单元 """

    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, in_channels//2, 1, padding=0),
            ConvBlock(in_channels//2, in_channels, 3),
        )

    def forward(self, x):
        y = self.features(x)
        return x+y


class ResidualBlock(nn.Module):
    """ 残差块 """

    def __init__(self, in_channels: int, n_residuals=1):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        n_residuals: int
            残差单元的个数
        """
        super().__init__()
        self.conv = ConvBlock(in_channels, in_channels*2, 3, stride=2)
        self.residual_units = nn.Sequential(*[
            ResidualUnit(2*in_channels) for _ in range(n_residuals)
        ])

    def forward(self, x):
        return self.residual_units(self.conv(x))


class Darknet(nn.Module):
    """ 主干网络 """

    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3, 32, 3)
        self.residuals = nn.ModuleList([
            ResidualBlock(32, 1),
            ResidualBlock(64, 2),
            ResidualBlock(128, 8),
            ResidualBlock(256, 8),
            ResidualBlock(512, 4),
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入图像

        Returns
        -------
        x1: Tensor of shape `(N, 1024, H/32, W/32)`
        x2: Tensor of shape `(N, 512, H/16, W/16)`
        x3: Tensor of shape `(N, 256, H/8, W/8)`
        """
        x3 = self.conv(x)
        for layer in self.residuals[:-2]:
            x3 = layer(x3)

        x2 = self.residuals[-2](x3)
        x1 = self.residuals[-1](x2)
        return x1, x2, x3


class YoloBlock(nn.Module):
    """ Yolo 块 """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.features = nn.Sequential(*[
            ConvBlock(in_channels, out_channels, 1, padding=0),
            ConvBlock(out_channels, out_channels*2, 3, padding=1),
            ConvBlock(out_channels*2, out_channels, 1, padding=0),
            ConvBlock(out_channels, out_channels*2, 3, padding=1),
            ConvBlock(out_channels*2, out_channels, 1, padding=0),
        ])

    def forward(self, x):
        return self.features(x)


class Yolo(nn.Module):
    """ Yolo 神经网络 """

    def __init__(self, n_classes: int, anchors: list, image_size: int):
        """
        Parameters
        ----------
        n_classes: int
            类别数

        anchors: list
            先验框

        image_size: int
            图片尺寸
        """
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size

        self.darknet = Darknet()
        self.yolo1 = YoloBlock(1024, 512)
        self.yolo2 = YoloBlock(768, 256)
        self.yolo3 = YoloBlock(384, 128)
        # YoloBlock 后面的卷积部分
        out_channels = (n_classes+5)*3
        self.conv1 = nn.Sequential(*[
            ConvBlock(512, 1024, 3),
            nn.Conv2d(1024, out_channels, 1)
        ])
        self.conv2 = nn.Sequential(*[
            ConvBlock(256, 512, 3),
            nn.Conv2d(512, out_channels, 1)
        ])
        self.conv3 = nn.Sequential(*[
            ConvBlock(128, 256, 3),
            nn.Conv2d(256, out_channels, 1)
        ])
        # 上采样
        self.upsample1 = nn.Sequential(*[
            nn.Conv2d(512, 256, 1),
            nn.Upsample(scale_factor=2)
        ])
        self.upsample2 = nn.Sequential(*[
            nn.Conv2d(256, 128, 1),
            nn.Upsample(scale_factor=2)
        ])

        # 探测器
        self.detector = Detector(anchors, image_size, n_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入图像

        Returns
        -------
        y1: Tensor of shape `(N, 255, H/32, W/32)`
            最小的特征图

        y2: Tensor of shape `(N, 255, H/16, W/16)`
            中等特征图

        y3: Tensor of shape `(N, 255, H/8, W/8)`
            最大的特征图
        """
        x1, x2, x3 = self.darknet(x)
        x1 = self.yolo1(x1)
        y1 = self.conv1(x1)

        x2 = self.yolo2(torch.cat([self.upsample1(x1), x2], 1))
        y2 = self.conv2(x2)

        x3 = self.yolo3(torch.cat([self.upsample2(x2), x3], 1))
        y3 = self.conv3(x3)

        return y1, y2, y3

    def load(self, model_path: Union[Path, str]):
        """ 载入模型

        Parameters
        ----------
        model_path: str or Path
            模型文件路径
        """
        self.load_state_dict(torch.load(model_path))

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """ 预测结果

        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入图像

        Returns
        -------
        out: Tensor of shape `(N, n_classes, top_k, 5)`
            检测结果，最后一个维度的第一个元素为置信度，后四个元素为边界框 `(cx, cy, w, h)`
        """
        return self.detector(self(x))

    def detect(self, image: Union[str, np.ndarray], classes: List[str], conf_thresh=0.6, use_gpu=True):
        """ 对图片进行目标检测

        Parameters
        ----------
        image: str of `np.ndarray`
            图片路径或者 RGB 图像

        classes: List[str]
            类别列表

        conf_thresh: float
            置信度阈值，舍弃小于这个阈值的预测框

        use_gpu: bool
            是否使用 GPU

        Returns
        -------
        image: `~PIL.Image.Image`
            绘制了边界框、置信度和类别的图像
        """
        if not 0 <= conf_thresh < 1:
            raise ValueError("置信度阈值必须在 [0, 1) 范围内")

        if isinstance(image, str):
            if os.path.exists(image):
                image = np.array(Image.open(image).convert('RGB'))
            else:
                raise FileNotFoundError("图片不存在，请检查图片路径！")

        h, w, channels = image.shape
        if channels != 3:
            raise ValueError('输入的必须是三个通道的 RGB 图像')

        x = ToTensor(self.image_size).transform(image)
        if use_gpu:
            x = x.cuda()

        # 预测边界框和置信度，shape: (n_classes, top_k, 5)
        y = self.predict(x)[0]

        # 筛选出置信度不小于阈值的预测框
        bbox = []
        conf = []
        label = []
        for c in range(y.size(0)):
            mask = y[c, :, 0] >= conf_thresh

            # 将归一化的边界框还原
            boxes = y[c, :, 1:][mask]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            bbox.append(boxes.detach().numpy())

            conf.extend(y[c, :, 0][mask].tolist())
            label.extend([classes[c]] * mask.sum())

        image = draw(image, np.vstack(bbox), label, conf)
        return image
