import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
import torchvision


def get_pretrained_resnet34(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有10个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 10))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = True
    return finetune_net


def get_pretrained_resnet50(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(pretrained=True)
    # 定义一个新的输出网络，共有10个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 10))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = True
    return finetune_net


class mlp:
    pass


class Residual_blk(nn.Module):  # 定义残差块
    def __init__(self, in_channels, out_channels,
                 need_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=strides, padding=1)
        if need_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        Y = self.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return self.relu(Y)


class multi_Residual_blks(nn.Module):
    def __init__(self, num_blks, in_channels, out_channels,
                 need_1x1conv=False, strides=1):
        super().__init__()
        self.blks = []
        for _ in range(num_blks):
            self.blks.append(nn.Sequential(
                Residual_blk(in_channels, out_channels,
                             need_1x1conv, strides=strides)))
            strides = 1
            need_1x1conv = False
            in_channels = out_channels

    def forward(self, x):
        return self.blks(x)


class ResNet34(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64,
                                             kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2, padding=1)
                                   )
        self.res_blks1 = multi_Residual_blks(3, 64, 64)
        self.res_blks2 = multi_Residual_blks(4, 64, 128,
                                             need_1x1conv=True, strides=2)
        self.res_blks3 = multi_Residual_blks(6, 128, 256,
                                             need_1x1conv=True, strides=2)
        self.res_blks4 = multi_Residual_blks(3, 256, 512,
                                             need_1x1conv=True, strides=2)
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1, 1),
                                     nn.Flatten(),
                                     nn.Linear(512, num_classes)
                                     )

    def forward(self, x):
        Y = self.conv1(x)
        Y = self.res_blks1(Y)
        Y = self.res_blks2(Y)
        Y = self.res_blks3(Y)
        Y = self.res_blks4(Y)
        return self.pooling(Y)


class transformer(nn.Module):
    pass
