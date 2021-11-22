# coding:utf-8
import unittest

from net.dataset import VOCDataset, collate_fn
from torch.utils.data import DataLoader
from utils.augmentation_utils import YoloAugmentation


class TestDataset(unittest.TestCase):
    """ 测试数据集 """

    def __init__(self, methodName) -> None:
        super().__init__(methodName=methodName)
        root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
        self.dataset = VOCDataset(root, 'trainval', YoloAugmentation(416))
        self.dataloader = DataLoader(
            self.dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    def test_data_loader(self):
        """ 测试数据加载 """
        for images, targets in self.dataloader:
            pass