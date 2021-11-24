# coding:utf-8
import unittest

import torch
from net.dataset import VOCDataset
from utils.augmentation_utils import (BBoxToAbsoluteCoords, Compose,
                                      YoloAugmentation)
from utils.box_utils import draw


class TestAugmention(unittest.TestCase):
    """ 测试数据增强 """

    def __init__(self, methodName) -> None:
        super().__init__(methodName)
        root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
        self.dataset = VOCDataset(root, 'trainval')

    def test_voc_augmenter(self):
        """ 测试 VOC 图像增强器 """
        self.dataset.transformer = Compose(
            [YoloAugmentation(416), BBoxToAbsoluteCoords()])
        image, target = self.dataset[4]
        self.draw(image, target)

    def draw(self, image: torch.Tensor, target):
        """ 绘制图像 """
        image = image.permute(1, 2, 0).numpy()*255
        label = [self.dataset.classes[int(i)] for i in target[:, 0]]

        # 绘制边界框和标签
        image = draw(image, target[:, 1:], label)
        image.show()
