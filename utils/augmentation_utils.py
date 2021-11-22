# coding:utf-8
from typing import List

import cv2 as cv
import imgaug.augmenters as iaa
from imgaug.augmenters.meta import Augmenter
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from numpy import ndarray
import torch

from .box_utils import center_to_corner_numpy, corner_to_center_numpy


class Transformer:
    """ 数据增强接口 """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 对输入的数据进行增强

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        raise NotImplementedError("图像增强方法必须被重写")


class Compose(Transformer):
    """ 图像增强器流水线 """

    def __init__(self, transformers: List[Transformer]):
        """
        Parameters
        ----------
        transformers: List[Transformer]
            图像增强器列表
        """
        self.transformers = transformers

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        for t in self.transformers:
            image, bbox, label = t.transform(image, bbox, label)

        return image, bbox, label


class BBoxToAbsoluteCoords(Transformer):
    """ 将归一化的边界框还原为原始边界框 """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        h, w, c = image.shape
        bbox[:, [0, 2]] *= w
        bbox[:, [1, 3]] *= h
        return image, bbox, label


class BBoxToPercentCoords(Transformer):
    """ 将未归一化的边界框归一化 """

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        h, w, _ = image.shape
        bbox[:, [0, 2]] /= w
        bbox[:, [1, 3]] /= h
        return image, bbox, label


class Resize(Transformer):
    """ 调整图像大小 """

    def __init__(self, size=(416, 416)):
        super().__init__()
        self.size = size

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 调整图像大小

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            已经归一化的边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        return cv.resize(image, self.size), bbox, label


class ImageAugmenter(Transformer):
    """ 图像增强器 """

    def __init__(self, augmenter: Augmenter):
        super().__init__()
        self.augmenters = augmenter

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        """ 对输入的数据进行增强

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            没有归一化的边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        # 转换坐标形式
        bbox = center_to_corner_numpy(bbox)
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*bbox[i], label=label[i]) for i in range(len(label))],
            shape=image.shape
        )

        # 图像增加
        image, bounding_boxes = self.augmenters(
            image=image, bounding_boxes=bounding_boxes)

        # 将边界框变回坐标矩阵
        bounding_boxes.clip_out_of_image_()
        bbox = corner_to_center_numpy(bounding_boxes.to_xyxy_array())
        label = np.array([box.label for box in bounding_boxes])
        return image, bbox, label


class DefaultAugmenter(ImageAugmenter):
    """ 默认的图像增强器 """

    def __init__(self):
        augmenter = iaa.Sequential([
            iaa.Dropout([0, 0.01]),
            iaa.Sharpen((0, 0.1)),
            iaa.Affine((0.8, 1.5), (-0.1, 0.1), rotate=(-10, 10)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5)
        ])
        super().__init__(augmenter)


class Padding(ImageAugmenter):
    """ 填充图像为正方形 """

    def __init__(self):
        augmenter = iaa.Sequential([iaa.PadToAspectRatio(1)])
        super().__init__(augmenter)


class YoloAugmentation(Transformer):
    """ Yolo 训练时使用的数据集增强器 """

    def __init__(self, image_size: int) -> None:
        """
        Parameters
        ----------
        image_size: int
            图像缩放后的尺寸
        """
        super().__init__()
        self.transformers = Compose([
            BBoxToAbsoluteCoords(),
            DefaultAugmenter(),
            Padding(),
            BBoxToPercentCoords(),
            Resize((image_size, image_size))
        ])

    def transform(self, image: ndarray, bbox: ndarray, label: ndarray):
        return self.transformers.transform(image, bbox, label)


class ToTensor(Transformer):
    """ 将 np.ndarray 图像转换为 Tensor """

    def __init__(self, image_size=416):
        """
        Parameters
        ----------
        image_size: int
            缩放后的图像尺寸

        mean: tuple
            RGB 图像各通道的均值
        """
        super().__init__()
        self.image_size = image_size

    def transform(self, image: ndarray, bbox: ndarray = None, label: ndarray = None):
        """ 将图像进行缩放、中心化并转换为 Tensor

        Parameters
        ----------
        image: `~np.ndarray`
            RGB 图像

        bbox, label: None
            没有用到

        Returns
        -------
        image: Tensor of shape `(1, 3, image_size, image_size)`
            转换后的图像
        """
        size = self.image_size
        x = cv.resize(image, (size, size)).astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x
