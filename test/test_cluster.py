# coding:utf-8
import unittest
from time import time

from utils.cluster_utils import AnchorKmeans


class TestCluster(unittest.TestCase):
    """ 测试先验框聚类 """

    def test_voc_cluster(self):
        """ 测试 VOC 数据集聚类 """
        root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
        model = AnchorKmeans(root)

        t0 = time()
        clusters = model.get_cluster(9)
        t1 = time()

        print(f'耗时: {t1-t0} s')
        print('聚类结果:\n', clusters*416)
        print('平均 IOU:', model.average_iou(clusters))
