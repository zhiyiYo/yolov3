# coding:utf-8
from net import VOCDataset
from utils.detection_utils import image_detect

# 模型文件和图片路径
model_path = 'model/2021-11-22_12-18-59/Yolo_5.pth'
image_path = 'resource/image/飞机.jpg'

# 检测目标
image = image_detect(model_path, image_path, VOCDataset.classes, conf_thresh=0.1, nms_thresh=0.3)
image.show()