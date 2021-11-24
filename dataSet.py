import os
import pandas as pd
import shutil
import collections
import math
import torch
from torch.utils import data
import torchvision

DATA_DIR = '../dataSet/cifar-10/'


def read_csv_labels(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict((name, label) for name, label in tokens)


def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_val(data_dir, labels, val_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_val_per_label = max(1, math.floor(n * val_ratio))
    label_count = {}  # 每个label的文件计数
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]  # 得到每个图片的label
        f_name = os.path.join(data_dir, 'train', train_file)
        # 讲所有训练数据复制到新文件夹中，并通过label分类到各个文件夹
        copyfile(f_name, os.path.join(
            data_dir, 'train_val_test', 'train_val', label))
        if label not in label_count or label_count[label] < n_val_per_label:
            copyfile(f_name, os.path.join(
                data_dir, 'train_val_test', 'val', label))
            # 每复制一个图片到相应valid数据文件夹中，该label_count + 1
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(f_name, os.path.join(
                data_dir, 'train_val_test', 'train', label))
    return n_val_per_label


def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        f_name = os.path.join(data_dir, 'test', test_file)
        copyfile(f_name, os.path.join(
            data_dir, 'train_val_test', 'test', 'unknown'))


def reorg_data(data_dir, val_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_val(data_dir, labels, val_ratio)
    reorg_test(data_dir)


def load_data_cifar10(data_dir, val_ratio, batch_size, enable_dataAugment=True):
    transform_train = None
    transform_test = None
    if enable_dataAugment is True:  # 如果需要数据增强，则给数据增强规则赋值
        transform_train = torchvision.transforms.Compose([torchvision.transforms.Resize(40),
                                                          torchvision.transforms.RandomResizedCrop(
                                                          32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
                                                          torchvision.transforms.RandomHorizontalFlip(),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                           [0.2023, 0.1994, 0.2010])])

        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                          [0.2023, 0.1994, 0.2010])])
    # 如果不存在train_val_test文件夹，则将原数据重新整理
    if os.path.exists(os.path.join(data_dir, 'train_val_test', 'train')) is False:
        reorg_data(data_dir, val_ratio)
    # 读取数据集
    train_ds, train_val_ds = [torchvision.datasets.ImageFolder(os.path.join(
        data_dir, 'train_val_test', folder),
        transform=transform_train) for folder in ['train', 'train_val']]
    val_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(
        data_dir, 'train_val_test', folder),
        transform=transform_test) for folder in ['val', 'test']]
    train_iter, train_val_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True)
        for dataset in (train_ds, train_val_ds)]
    val_iter = torch.utils.data.DataLoader(
        val_ds, batch_size, shuffle=False, drop_last=True)
    test_iter = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False, drop_last=False)
    return train_iter, val_iter, test_iter, train_val_iter, len(test_ds), train_ds.classes
