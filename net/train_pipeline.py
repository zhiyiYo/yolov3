# coding:utf-8
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
from torch import optim, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.log_utils import LossLogger
from utils.datetime_utils import time_delta

from .dataset import collate_fn
from .loss import YoloLoss
from .yolo import Yolo


def exception_handler(train_func):
    """ 处理训练过程中发生的异常并保存模型 """
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            return train_func(train_pipeline, *args, **kwargs)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()

            train_pipeline.save()

            # 清空 GPU 缓存
            cuda.empty_cache()

            exit()

    return wrapper


class TrainPipeline:
    """ 训练模型流水线 """

    def __init__(self, n_classes: int, image_size: int, anchors: list, dataset: Dataset, darknet_path: str = None,
                 yolo_path: str = None, lr=0.01, backbone_lr=1e-3, momentum=0.9, weight_decay=4e-5, lr_step_size=20,
                 batch_size=16, start_epoch=0, max_epoch=60, save_frequency=5, use_gpu=True, save_dir='model',
                 log_file: str = None, log_dir='log'):
        """
        Parameters
        ----------
        n_classes: int
            类别数

        image_size: int
            输入 Yolo 神经网络的图片大小

        anchors: list of shape `(3, n_anchors, 2)`
            输入神经网络的图片尺寸为 416 时的先验框尺寸

        dataset: Dataset
            训练数据集

        darknet_path: str
            预训练的 darknet53 模型文件路径

        yolo_path: Union[str, None]
            Yolo 模型文件路径，有以下两种选择:
            * 如果不为 `None`，将使用模型文件中的参数初始化 `Yolo`
            * 如果为 `None`，将随机初始化 darknet53 之后的各层参数

        lr: float
            学习率

        backbone_lr: float
            主干网络学习率

        momentum: float
            冲量

        weight_decay: float
            权重衰减

        lr_step_size: int
            学习率退火的步长

        batch_size: int
            训练集 batch 大小

        start_epoch: int
            Yolo 模型文件包含的参数是训练了多少个 epoch 的结果

        max_epoch: int
            最多迭代多少个 epoch

        save_frequency: int
            迭代多少个 epoch 保存一次模型

        use_gpu: bool
            是否使用 GPU 加速训练

        save_dir: str
            保存 SSD 模型的文件夹

        log_file: str
            训练损失数据历史记录文件，要求是 json 文件

        save_dir: str
            训练损失数据保存的文件夹
        """
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.use_gpu = use_gpu
        self.save_frequency = save_frequency
        self.batch_size = batch_size

        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch

        if use_gpu and cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # 创建模型
        self.model = Yolo(n_classes, image_size, anchors).to(self.device)
        if yolo_path:
            self.model.load(yolo_path)
            print('🧪 成功载入 Yolo 模型：' + yolo_path)
        elif darknet_path:
            self.model.darknet.load(darknet_path)
            print('🧪 成功载入 Darknet53 模型：' + darknet_path)
        else:
            raise ValueError("必须指定预训练的 Darknet53 模型文件路径")

        # 创建优化器和损失函数
        self.criterion = YoloLoss(anchors, n_classes, image_size)

        darknet_params = self.model.darknet.parameters()
        other_params = [i for i in self.model.parameters()
                        if i not in darknet_params]
        self.optimizer = optim.SGD(
            [
                {"params": darknet_params, 'initial_lr': backbone_lr, 'lr': backbone_lr},
                {'params': other_params, 'initial_lr': lr, 'lr': lr}
            ],
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.lr_schedule = optim.lr_scheduler.StepLR(
            self.optimizer, lr_step_size, 0.1, last_epoch=start_epoch)

        # 训练损失记录器
        self.n_batches = len(self.dataset)//self.batch_size
        self.logger = LossLogger(self.n_batches, log_file, log_dir)

    def save(self):
        """ 保存模型和训练损失数据 """
        self.pbar.close()
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # 保存模型
        self.model.eval()
        path = self.save_dir/f'Yolo_{self.current_epoch+1}.pth'
        torch.save(self.model.state_dict(), path)

        # 保存训练损失数据
        self.logger.save(f'train_losses_{self.current_epoch+1}')

        print(f'\n🎉 已将当前模型保存到 {path.absolute()}\n')

    @exception_handler
    def train(self):
        """ 训练模型 """
        t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        self.save_dir = self.save_dir/t
        self.logger.save_dir = self.logger.save_dir/t

        # 数据迭代器
        data_loader = DataLoader(
            self.dataset, self.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        print('🚀 开始训练！')

        for e in range(self.start_epoch, self.max_epoch):
            self.current_epoch = e

            self.model.train()

            # 创建进度条
            self.pbar = tqdm(total=self.n_batches, bar_format=bar_format)
            self.pbar.set_description(f"\33[36m💫 Epoch {(e+1):5d}")
            start_time = datetime.now()

            for images, targets in data_loader:
                # 预测边界框、置信度和条件类别概率
                preds = self.model(images.to(self.device))

                # 误差反向传播
                self.optimizer.zero_grad()
                loc_loss, conf_loss, cls_loss = self.criterion(preds, targets)
                loss = loc_loss + conf_loss + cls_loss
                loss.backward()
                self.optimizer.step()

                # 记录误差
                self.logger.update(
                    loc_loss.item(), conf_loss.item(), cls_loss.item())

                # 更新进度条
                cost_time = time_delta(start_time)
                self.pbar.set_postfix_str(
                    f'loss: {loss.item():.5f}, loc_loss: {loc_loss.item():.5f}, conf_loss: {conf_loss.item():.5f}, cls_loss: {cls_loss.item():.5f}, 执行时间: {cost_time}\33[0m')
                self.pbar.update()

            # 关闭进度条
            self.pbar.close()

            # 学习率退火
            self.lr_schedule.step()

            # 定期保存模型
            if e > self.start_epoch and (e+1-self.start_epoch) % self.save_frequency == 0:
                self.save()

        self.save()
