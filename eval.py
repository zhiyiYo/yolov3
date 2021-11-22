# coding:utf-8
from net import EvalPipeline, VOCDataset

# load dataset
root = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'test')

model_path = 'model/2021-11-22_12-18-59/Yolo_5.pth'
eval_pipeline = EvalPipeline(model_path, dataset, conf_thresh=0)
eval_pipeline.eval()