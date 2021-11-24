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


'''定义一个ResNet'''


class Residual_blk(nn.Module):  # 定义残差块
    def __init__(self, in_channels, out_channels,
                 need_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if need_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
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
            self.blks.append(Residual_blk(in_channels, out_channels,
                                          need_1x1conv, strides=strides))
            strides = 1
            need_1x1conv = False
            in_channels = out_channels
        self.residual_blks = nn.Sequential(*self.blks)

    def forward(self, x):
        Y = self.residual_blks(x)
        return Y


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
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
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


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n):  # n为控制总层数的变量
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True)
                                   )
        self.res_blks1 = multi_Residual_blks(n, 16, 16)  # 32 x 32
        self.res_blks2 = multi_Residual_blks(n, 16, 32,
                                             need_1x1conv=True, strides=2)  # 16 x16
        self.res_blks3 = multi_Residual_blks(n, 32, 64,
                                             need_1x1conv=True, strides=2)  # 8 x 8
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Flatten(),
                                     nn.Linear(64, num_classes)
                                     )

    def forward(self, x):
        Y = self.conv1(x)
        Y = self.res_blks1(Y)
        Y = self.res_blks2(Y)
        Y = self.res_blks3(Y)
        return self.pooling(Y)


'''定义一个vision transformer'''


class Pic_Embedding(nn.Module):  #
    def __init__(self, **kwargs):
        super(Pic_Embedding).__init__(**kwargs)

    def forward(self, x):
        pass


class Position_Encoding(nn.Module):  # 位置编码
    def __init__(self, **kwargs):
        super(Position_Encoding).__init__(**kwargs)

    def forward(self, x):
        pass


class PositionWiseFFN(nn.Module):  # Feed-Forward neural network
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output,
                 **kwargs):
        super(PositionWiseFFN).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU(inplace=True)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, x):
        Y = self.dense2(self.relu(self.dense1(x)))
        return Y


class AddNorm(nn.Module):
    def __init__(self, **kwargs):
        super(AddNorm).__init__(**kwargs)

    def forward(self, x, y):
        pass


class EncoderBlk(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderBlk).__init__(**kwargs)

    def forward(self, x):
        pass


class Transformer_Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer_Encoder).__init__(**kwargs)

    def forward(self, x):
        pass


class DecoderBlk(nn.Module):
    def __init__(self, **kwargs):
        super(DecoderBlk).__init__(**kwargs)

    def forward(self, x):
        pass


class Transformer_Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Transformer_Decoder).__init__(**kwargs)

    def forward(self, x):
        pass


class transformer(nn.Module):
    def __init__(self, **kwargs):
        super(transformer).__init__(**kwargs)

    def forward(self, x):
        pass
