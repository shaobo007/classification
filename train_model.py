import os
import torch
from torch import nn
from d2l import torch as d2l
from torch._C import device


loss = nn.CrossEntropyLoss(reduction='none')


def train_pretrained_model(net, num_epochs, lr, wd, lr_period, lr_decay, devices,
                           train_iter, val_iter=None, save_train_process_to_csv='train_process_resnet34.csv'):
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()  # 定义一个计时器
    train_process = open(os.path.join('./', save_train_process_to_csv), 'w')
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):  # 训练一个epoch
        net.train()
        metric = d2l.Accumulator(3)  # 定义一个累加器
        for i, (features, labels) in enumerate(train_iter):
            # 训练一个batch
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            # 计算train loss, train acc, val acc
            train_loss = l.sum()
            train_acc = d2l.accuracy(output, labels)
            metric.add(train_loss, train_acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 4) == 0 or i == num_batches - 1:
                train_loss_txt = f'train_loss: {metric[0]/metric[2]:.4f}, '
                train_acc_txt = f'train_acc: {metric[1]/metric[2]:.4f}\n'
                train_process.write(f'epoch {epoch + (i + 1) / num_batches:.2f}: ' +
                                    train_loss_txt + train_acc_txt)  # 将训练结果写入文件
                print(f'epoch {epoch + (i + 1) / num_batches:.2f}',
                      train_loss_txt, train_acc_txt)
        if val_iter is not None:
            val_acc = d2l.evaluate_accuracy_gpu(
                net, val_iter)  # 定义一个计算网络在验证集的准确度
            val_acc_txt = f'val_acc:{val_acc:4f}\n'
            train_process.write(val_acc_txt)
            print(val_acc_txt)
        scheduler.step()
    measure = (f'train_loss: {metric[0]/metric[2]:.4f}, '
               f'train_acc: {metric[1]/metric[2]:.4f}')
    if val_iter is not None:
        measure += f', val_acc: {val_acc:.4f}'
    train_process.write(measure + f'\n{metric[2] * num_epochs / timer.sum():.1f}' +
                        f' examples/sec on {str(devices)}\n' +
                        f'total exacutive time: {timer.sum() / 60} mins\n')
    print(measure + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}\n'
          f'total exacutive time: {timer.sum() / 60:.1f} mins\n')


def train_notPretrained_model(net, num_epochs, lr, wd, lr_period, lr_decay, devices,
                              train_iter, val_iter=None, save_train_process_to_csv='train_process_resnet.csv'):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        trainer, lr_period, lr_decay)  # 每过lr_period个epoch更新一次lr
#    scheduler = torch.optim.lr_scheduler.MultiStepLR(
#        trainer, milestones=[90,136], gamma=0.1, last_epoch=-1)
    num_batches, timer = len(train_iter), d2l.Timer()  # 定义一个计时器

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    train_process = open(os.path.join('./', save_train_process_to_csv), 'w')
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):  # 训练一个epoch
        net.train()
        metric = d2l.Accumulator(3)  # 定义一个累加器
        for i, (features, labels) in enumerate(train_iter):
            # 训练一个batch
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            # 计算train loss, train acc, val acc
            train_loss = l.sum()
            train_acc = d2l.accuracy(output, labels)
            metric.add(train_loss, train_acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 4) == 0 or i == num_batches - 1:
                train_loss_txt = f'train_loss: {metric[0]/metric[2]:.4f}, '
                train_acc_txt = f'train_acc: {metric[1]/metric[2]:.4f}\n'
                train_process.write(f'epoch {epoch + (i + 1) / num_batches:.2f}: ' +
                                    train_loss_txt + train_acc_txt)  # 将训练结果写入文件
                print(f'epoch {epoch + (i + 1) / num_batches:.2f}',
                      train_loss_txt, train_acc_txt)
        if val_iter is not None:
            val_acc = d2l.evaluate_accuracy_gpu(
                net, val_iter)  # 定义一个计算网络在验证集的准确度
            val_acc_txt = f'val_acc:{val_acc:4f}\n'
            train_process.write(val_acc_txt)
            print(val_acc_txt)
        scheduler.step()
    measure = (f'train_loss: {metric[0]/metric[2]:.4f}, '
               f'train_acc: {metric[1]/metric[2]:.4f}')
    if val_iter is not None:
        measure += f', val_acc: {val_acc:.4f}'
    train_process.write(measure + f'\n{metric[2] * num_epochs / timer.sum():.1f}' +
                        f' examples/sec on {str(devices)}\n' +
                        f'total exacutive time: {timer.sum() / 60} mins\n')
    print(measure + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}\n'
          f'total exacutive time: {timer.sum() / 60:.1f} mins\n')
