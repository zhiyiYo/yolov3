# coding:utf-8
import unittest

import torch
from net import Yolo


class TestYolo(unittest.TestCase):
    """ 测试 Yolo 模型 """

    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        self.model = Yolo(80, 256).cuda()

    def test_forward(self):
        """ 测试前馈 """
        x = torch.rand(2, 3, 256, 256).cuda()
        y1, y2, y3 = self.model(x)
        self.assertEqual(y1.size(), torch.Size((2, 255, 8, 8)))
        self.assertEqual(y2.size(), torch.Size((2, 255, 16, 16)))
        self.assertEqual(y3.size(), torch.Size((2, 255, 32, 32)))

    def test_predict(self):
        """ 测试推理 """
        x = torch.rand(2, 3, 256, 256).cuda()
        out = self.model.predict(x)
        print('\n预测结果：', out[0, 0, :5])
