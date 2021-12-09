import torch
import math
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


def Pic_Embedding(input_Pic, patch_size):
    # assert input_Pic % patch_size == 0
    num_patches = input_Pic.shape[2] // patch_size
    # tokens = torch.zeros((num_patches ** 2, input_Pic.shape[0], input_Pic.shape[1], patch_size, patch_size))
    tokens = {}
    for i in range(num_patches):
        for j in range(num_patches):
            tokens[i*num_patches + j] = input_Pic[:, :, i*patch_size:(i+1)*patch_size,
                                                  j*patch_size:(j+1)*patch_size]
    return tokens


class Picture_Embedding(nn.Module):
    # 将图片转化为token做embedding
    # 32x32 -->  4x4 patch
    # total 8x8 patches
    def __init__(self, picture_size, patch_size, **kwargs):
        super(Picture_Embedding).__init__(**kwargs)
        self.flatten = nn.Flatten()
        self.patch_size = patch_size
    def forward(self, x):
        tokens = Pic_Embedding(x, self.patch_size)



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


class Encoder(nn.Module):
    def __init__(self,  **kwargs):
        super(Encoder).__init__(**kwargs)

    def forward(self, x):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError


class Encoder_Decoder(nn.Module):
    def __init__(self,  **kwargs):
        super(Encoder_Decoder).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, dec_state)


class EncoderBlk(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, used_bias=False, **kwargs):
        super(EncoderBlk).__init__(**kwargs)
        self.attention = MuiltiHead_Attention(key_size, query_size, value_size,
                                              num_hiddens, num_heads, dropout,
                                              used_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, x, valid_lens):
        Y = self.addnorm1(x, self.attention(x, x, x, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class Transformer_Encoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(Transformer_Encoder).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = Position_Encoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlk(key_size, query_size, value_size,
                                            num_hiddens, norm_shape,
                                            ffn_num_input, ffn_num_hiddens,
                                            num_heads, dropout, use_bias))

    def forward(self, x, valid_lens, *args):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blks(x, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return x


class DecoderBlk(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlk).__init__(**kwargs)
        self.i = i
        self.attention1 = MuiltiHead_Attention(key_size, query_size, value_size,
                                               num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MuiltiHead_Attention(key_size, query_size, value_size,
                                               num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, x, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[2][self.i], x), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(
                1, num_steps+1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        x2 = self.attention1(x, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(x, x2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class Transformer_Decoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(Transformer_Decoder).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = Position_Encoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(self.num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlk(key_size, query_size, value_size,
                                            num_hiddens, norm_shape,
                                            ffn_num_input, ffn_num_hiddens,
                                            num_heads, dropout))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, x, state):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self._attention_weight = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)
            self._attention_weight[0][i] = blk.attention1.attention.attention_weights
            self._attention_weight[1][i] = blk.attention2.attention.attention_weights
        return self.dense(x), state

    @property
    def attention_weights(self):
        return self._attention_weights


class transformer(nn.Module):
    def __init__(self, **kwargs):
        super(transformer).__init__(**kwargs)

    def forward(self, x):
        pass


def main():
    x = torch.randn((1, 3, 32, 32))
    tokens = Pic_Embedding(x, 4)
    print(tokens[0])

if __name__ == '__main__':
    main()

