import torch
from torch import nn
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


class ResNet:
    pass


class transformer:
    pass
