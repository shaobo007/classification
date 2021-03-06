import torch
# import torchvision

# from torch import nn
# from d2l import torch as d2l
from dataSet import DATA_DIR
from dataSet import load_data_cifar10
from train_model import train_notPretrained_model
# from models import get_pretrained_resnet34, get_pretrained_resnet50, ResNet34, ResNet, ViT
from VITmodel import ViT
from lr_scheduler import CosineScheduler


def try_all_gpus():
    """返回所有可用的GPU，如果没有就使用cpu"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


BATCH_SIZE = 128
VAL_RATIO = 0.1
DEVICES = try_all_gpus()
input_channels = 3
num_classes = 10
# n = 3  # 层数为110的resNet

train_iter, val_iter, test_iter, train_val_iter, num_test, classes = load_data_cifar10(
    DATA_DIR, VAL_RATIO, BATCH_SIZE)  # 载入数据集
#net_1 = get_pretrained_resnet34(DEVICES)
#net_1 = get_pretrained_resnet50(DEVICES)
#net_2 = get_pretrained_resnet34(DEVICES)
#ResNet = ResNet34(input_channels, num_classes)
# model_ResNet = ResNet(input_channels, num_classes, n)  # 层数为110的resNet
model_vit = ViT(64, 8, num_classes, 192, 12, 12, 1024, dropout=0.1)  # ViT
# x = torch.zeros(1, 3, 32, 32)
# print(model_ResNet(x).shape)
num_epochs, lr, wd = 20, 2e-4, 5e-4
#num_epochs, lr, wd = 180, 0.1, 1e-4
lr_period, lr_decay = 4, 0.9
scheduler = CosineScheduler(50, base_lr=4e-4, final_lr=3e-5, warmup_steps=8, warmup_begin_lr=lr)
train_notPretrained_model(model_vit, num_epochs, lr, wd, lr_period, lr_decay, DEVICES,
                          train_iter, val_iter, scheduler=scheduler,
                          save_train_process_to_csv='train_process_ViT_1.csv')  # 只在训练样本上训练

# train(ResNet, num_epochs, lr, wd, lr_period, lr_decay, DEVICES,
#     train_val_iter, save_train_process_to_csv='train_process_resnet34_final.csv')  # 在训练与验证样本上训练

torch.save(model_vit.state_dict(), 'cifar10-ViT_1.pth')
