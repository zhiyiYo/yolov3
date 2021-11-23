# coding:utf-8
from net import TrainPipeline, VOCDataset
from utils.augmentation_utils import YoloAugmentation


# train config
config = {
    "n_classes": 20,
    "image_size": 320,
    "anchors": [
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]],
    ],
    "darknet_path": "model/darknet53.pth",
    "batch_size": 8
}

# load dataset
root = 'data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
dataset = VOCDataset(
    root,
    'trainval',
    YoloAugmentation(config['image_size']),
    keep_difficult=True
)


# train
train_pipeline = TrainPipeline(dataset=dataset, **config)
train_pipeline.train()
