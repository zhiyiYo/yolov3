# coding:utf-8
from unittest import TestCase

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.plot_utils import *

mpl.rc_file('resource/theme/matlab.mplstyle')


class TestPlot(TestCase):
    """ 测试绘制 """

    def test_plot_loss(self):
        """ 测试损失绘制 """
        fig, ax = plot_loss('log/2021-11-22_22-03-27/train_losses_5.json')
        plt.show()

    def test_plot_PR(self):
        """ 测试 PR 曲线绘制 """
        fig, ax = plot_PR('eval/SSD_AP.json', 'cat')
        plt.show()

    def test_plot_AP(self):
        """ 测试 AP 柱状图绘制 """
        fig, ax = plot_AP('eval/SSD_AP.json')
        plt.show()
