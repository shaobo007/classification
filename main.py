import torch
import torchvision

from torch import nn
from d2l import torch as d2l
from dataSet import DATA_DIR
from dataSet import load_data_cifar10
from train_model import train
from models import get_pretrained_resnet34, get_pretrained_resnet50


def try_all_gpus():
    """返回所有可用的GPU，如果没有就使用cpu"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


BATCH_SIZE = 256
VAL_RATIO = 0.1
DEVICES = try_all_gpus()
train_iter, val_iter, test_iter, train_val_iter, num_test, classes = load_data_cifar10(
    DATA_DIR, VAL_RATIO, BATCH_SIZE)  # 载入数据集

#net_1 = get_pretrained_resnet34(DEVICES)
#net_1 = get_pretrained_resnet50(DEVICES)
#net_2 = get_pretrained_resnet34(DEVICES)
net_2 = get_pretrained_resnet34(DEVICES)
num_epochs, lr, wd = 20, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9

# train(net_1, num_epochs, lr, wd, lr_period, lr_decay, DEVICES,
#      train_iter, val_iter, save_train_process_to_csv='train_process_resnet50.csv')  # 只在训练样本上训练

train(net_2, num_epochs, lr, wd, lr_period, lr_decay, DEVICES,
      train_val_iter, save_train_process_to_csv='train_process_resnet34_final.csv')  # 在训练与验证样本上训练

torch.save(net_2.state_dict(), 'cifar10-pretrained_resnet34_final.params')
