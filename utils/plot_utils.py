# coding:utf-8
import json

import numpy as np
import matplotlib.pyplot as plt

from .log_utils import LossLogger


def plot_loss(log_file: str):
    """ 绘制损失曲线

    Parameters
    ----------
    log_file: str
        损失日志文件路径
    """
    logger = LossLogger(None, log_file)
    epoch = np.arange(0, len(logger.losses))+1

    fig = plt.figure(num='损失曲线', tight_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 总损失
    ax1 = fig.add_subplot(gs[0, :])  # type:plt.Axes
    ax1.plot(epoch, logger.losses)
    ax1.set(xlabel='epoch', title='Total Loss')

    # 定位损失
    ax2 = fig.add_subplot(gs[1, 0])  # type:plt.Axes
    ax2.plot(epoch, logger.loc_losses)
    ax2.set(xlabel='epoch', title='Location Loss')

    # 置信度损失
    ax3 = fig.add_subplot(gs[1, 1])  # type:plt.Axes
    ax3.plot(epoch, logger.conf_losses)
    ax3.set(xlabel='epoch', title='Confidence Loss')

    # 分类损失
    ax4 = fig.add_subplot(gs[1, 2])  # type:plt.Axes
    ax4.plot(epoch, logger.cls_losses)
    ax4.set(xlabel='epoch', title='Classification Loss')

    axes = [ax1, ax2, ax3, ax4]

    return fig, axes


def plot_PR(file_path: str, class_name: str):
    """ 绘制 PR 曲线

    Parameters
    ----------
    file_path: str
        由 eval.py 生成的测试结果文件路径

    class_name: str
        类别名
    """
    with open(file_path, encoding='utf-8') as f:
        result = json.load(f)[class_name]

    fig, ax = plt.subplots(1, 1, num='PR 曲线')
    ax.plot(result['recall'], result['precision'])
    ax.set(xlabel='recall', ylabel='precision', title='PR curve')
    return fig, ax


def plot_AP(file_path: str):
    """ 绘制 AP 柱状图 """
    with open(file_path, encoding='utf-8') as f:
        result = json.load(f)

    AP = []
    classes = []
    for k, v in result.items():
        AP.append(v['AP'])
        classes.append(k)

    fig, ax = plt.subplots(1, 1, num='AP 柱状图')
    ax.barh(range(len(AP)), AP, height=0.6, tick_label=classes)
    ax.set(xlabel='AP')

    return fig, ax
