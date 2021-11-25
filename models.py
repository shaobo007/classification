from tkinter.constants import NO
import torch
import math
from torch import nn
from torch._C import dtype
from torch.nn.modules import dropout
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


class Pic_Embedding(nn.Module):  # 将图片转化为token做embedding
    def __init__(self, **kwargs):
        super(Pic_Embedding).__init__(**kwargs)

    def forward(self, x):
        pass


class Position_Encoding(nn.Module):  # 位置编码
    def __init__(self, num_hiddens, dropout, max_len=1000, **kwargs):
        super(Position_Encoding).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arrange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arrange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)  # 从第0个开始，间隔为2
        self.P[:, :, 1::2] = torch.cos(X)  # 从第一个开始，间隔为2

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


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
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    if valid_lens is not None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):  # q, k, v做点积
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens=None):
        d = q.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v)


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MuiltiHead_Attention(nn.Module):  # 多头注意力机制实现
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MuiltiHead_Attention).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_q(keys), self.num_heads)
        values = transpose_qkv(self.W_q(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


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
