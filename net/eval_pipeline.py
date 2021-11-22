# coding:utf-8
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image
from utils.box_utils import jaccard_overlap_numpy, center_to_corner_numpy
from utils.augmentation_utils import ToTensor

from .dataset import VOCDataset
from .yolo import Yolo


class EvalPipeline:
    """ 测试模型流水线 """

    def __init__(self, model_path: str, dataset: VOCDataset, image_size=416, anchors: list = None,
                 conf_thresh=0.05, overlap_thresh=0.5, save_dir='eval', use_07_metric=False, use_gpu=True):
        """
        Parameters
        ----------
        model_path: str
            模型文件路径

        dataset: VOCDataset
            数据集

        image_size: int
            图像尺寸

        anchors: list
            输入神经网络的图像大小为 416 时对应的先验框

        conf_thresh: float
            置信度阈值

        overlap_thresh: float
            IOU 阈值

        save_dir: str
            测试结果和预测结果文件的保存目录

        use_07_metric: bool
            是否使用 VOC2007 的 AP 计算方法

        use_gpu: bool
            是否使用 GPU
        """
        self.use_gpu = use_gpu
        self.dataset = dataset
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.overlap_thresh = overlap_thresh
        self.use_07_metric = use_07_metric
        self.save_dir = Path(save_dir)

        self.model_path = Path(model_path)
        self.device = 'cuda:0' if use_gpu else 'cpu'
        self.model = Yolo(self.dataset.n_classes, image_size, anchors)
        self.model = self.model.to(self.device)
        self.model.load(model_path)
        self.model.eval()

    @torch.no_grad()
    def eval(self):
        """ 测试模型，获取 mAP """
        self._predict()
        self._get_ground_truth()
        self._get_mAP()

    def _predict(self):
        """ 预测每一种类存在于哪些图片中 """
        suffix = f'_conf{self.conf_thresh}_iou{self.overlap_thresh}_pred.json'
        p = self.save_dir/(self.model_path.stem + suffix)
        if p.exists():
            print(f'🛸 从 {p} 中取得预测数据')
            with open(p, encoding='utf-8') as f:
                self.preds = json.load(f)
                return

        self.preds = {c: {} for c in self.dataset.classes}
        transformer = ToTensor(self.image_size)

        print('🛸 正在预测中...')
        size = self.image_size
        for i, (image_path, image_name) in enumerate(zip(self.dataset.image_paths, self.dataset.image_names)):
            print(f'\r当前进度：{i/len(self.dataset):.0%}', end='')

            # 读入图片
            image = np.array(Image.open(image_path).convert('RGB'))
            h, w, _ = image.shape

            # 预测
            x = transformer.transform(image).to(self.device)
            out = self.model.predict(x)[0].cpu().numpy()

            for i, c in enumerate(self.dataset.classes):
                y = out[i]
                mask = y[:, 0] > self.conf_thresh

                # 如果没有一个边界框的置信度大于阈值就纸条跳过这个类
                if not mask.any():
                    continue

                # 筛选出满足阈值条件的边界框
                conf = y[:, 0][mask]  # type:np.ndarray
                bbox = y[:, 1:][mask]  # type:np.ndarray
                bbox[:, [0, 2]] *= (w/self.image_size)
                bbox[:, [1, 3]] *= (h/self.image_size)
                bbox[:, [0, 2]] += 1
                bbox = center_to_corner_numpy(bbox)

                # 保存预测结果
                self.preds[c][image_name] = {
                    "bbox": bbox.tolist(),
                    "conf": conf.tolist()
                }

        # 保存预测数据
        self.save_dir.mkdir(exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(self.preds, f)

    def _get_ground_truth(self):
        """ 获取 ground truth 中每一种类存在于哪些图片中 """
        self.ground_truths = {c: {} for c in self.dataset.classes}
        self.n_positives = {c: 0 for c in self.dataset.classes}

        print('\n\n🧩 正在获取标签中...')
        for i, (anno_path, img_name) in enumerate(zip(self.dataset.annotation_paths, self.dataset.image_names)):
            print(f'\r当前进度：{i/len(self.dataset):.0%}', end='')

            root = ET.parse(anno_path).getroot()

            for obj in root.iter('object'):
                # 获取标签含有的的类和边界框
                c = obj.find('name').text.lower().strip()
                difficult = int(obj.find('difficult').text)
                bbox = obj.find('bndbox')
                bbox = [
                    int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                ]

                if not self.ground_truths[c].get(img_name):
                    self.ground_truths[c][img_name] = {
                        "bbox": [],
                        "detected": [],
                        "difficult": []
                    }

                # 添加一条 ground truth 记录
                self.ground_truths[c][img_name]['bbox'].append(bbox)
                self.ground_truths[c][img_name]['detected'].append(False)
                self.ground_truths[c][img_name]['difficult'].append(difficult)
                self.n_positives[c] += (1-difficult)

    def _get_mAP(self):
        """ 计算 mAP """
        result = {}

        print('\n\n🧪 正在计算 AP 中...')
        mAP = 0
        for c in self.dataset.classes:
            ap, precision, recall = self._get_AP(c)
            result[c] = {
                'AP': ap,
                'precision': precision,
                'recall': recall
            }
            mAP += ap
            print(f'class {c}: AP={ap:.2%}')

        mAP /= len(self.dataset.classes)
        print(f'mAP of {self.model_path}: {mAP:.2%}')

        # 保存评估结果
        self.save_dir.mkdir(exist_ok=True)
        p = self.save_dir / (self.model_path.stem + '_AP.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(result, f)

    def _get_AP(self, c: str):
        """ 计算一个类的 AP

        Parameters
        ----------
        c: str
            类别名

        Returns
        -------
        ap: float
            AP，没有预测出这个类就返回 0

        precision: list
            查准率

        recall: list
            查全率
        """
        pred = self.preds[c]
        ground_truth = self.ground_truths[c]
        bbox = []
        conf = []
        image_names = []

        # 将 bbox 拼接为二维矩阵，每一行为一个预测框
        for image_name, v in pred.items():
            image_names.extend([image_name]*len(v['conf']))
            bbox.append(v['bbox'])
            conf.append(v['conf'])

        # 没有在任何一张图片中预测出这个类
        if not bbox:
            return 0, 0, 0

        bbox = np.vstack(bbox)  # type:np.ndarray
        conf = np.hstack(conf)  # type:np.ndarray
        image_names = np.array(image_names)

        # 根据置信度降序排序预测框
        index = np.argsort(-conf)
        bbox = bbox[index]
        conf = conf[index]
        image_names = image_names[index]

        # 计算 TP 和 FP
        tp = np.zeros(len(image_names))  # type:np.ndarray
        fp = np.zeros(len(image_names))  # type:np.ndarray
        for i, image_name in enumerate(image_names):
            # 获取一张图片中关于这个类的 ground truth
            record = ground_truth.get(image_name)

            # 这张图片的 ground_truth 中没有这个类就将 fp+1
            if not record:
                fp[i] = 1
                continue

            bbox_pred = bbox[i]  # shape:(4, )
            bbox_gt = np.array(record['bbox'])  # shape:(n, 4)
            difficult = np.array(record['difficult'], np.bool)  # shape:(n, )

            # 计算交并比
            iou = jaccard_overlap_numpy(bbox_pred, bbox_gt)
            iou_max = iou.max()
            iou_max_index = iou.argmax()

            if iou_max < self.overlap_thresh:
                fp[i] = 1
            elif not record['difficult'][iou_max_index]:
                # 已经匹配了预测框的边界框不能再匹配预测框
                if not record['detected'][iou_max_index]:
                    tp[i] = 1
                    record['detected'][iou_max_index] = True
                else:
                    fp[i] = 1

        # 查全率和查准率
        tp = tp.cumsum()
        fp = fp.cumsum()
        n_positives = self.n_positives[c]
        recall = tp / n_positives  # type:np.ndarray
        precision = tp / (tp + fp)  # type:np.ndarray

        # 计算 AP
        if not self.use_07_metric:
            rec = np.concatenate(([0.], recall, [1.]))
            prec = np.concatenate(([0.], precision, [0.]))

            # 计算 PR 曲线的包络线
            for i in range(prec.size-1, 0, -1):
                prec[i - 1] = np.maximum(prec[i - 1], prec[i])

            # 找出 recall 变化时的索引
            i = np.where(rec[1:] != rec[:-1])[0]

            # 用recall的间隔对精度作加权平均
            ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
        else:
            ap = 0
            for r in np.arange(0, 1.1, 0.1):
                if np.any(recall >= r):
                    ap += np.max(precision[recall >= r])/11

        return ap, precision.tolist(), recall.tolist()
