import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import math
from lr_scheduler import CosineScheduler

initial_lr = 2e-4
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()

optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
scheduler_1 = CosineScheduler(60, warmup_steps=8, base_lr=4e-4, final_lr=3e-5, warmup_begin_lr=2e-4)
# scheduler_1 = torch.optim.lr_scheduler.LambdaLR(optimizer_1, lr_lambda=lambda epoch:  1 / (0.22 * epoch + 1) if epoch < 15 else 1 / (0.1 * epoch + 2.4))
# scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, 15, eta_min=5e-5)
# optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
# # scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1/(epoch+1))
# scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: epoch+1 if epoch < 8 else 10/math.sqrt(epoch-7))

print("初始化的学习率：", optimizer_1.defaults['lr'])

for epoch in range(1, 100):
    # train
    optimizer_1.zero_grad()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    if scheduler_1:
        if scheduler_1.__module__ == torch.optim.lr_scheduler.__name__:
            # Using PyTorch In-Built scheduler
            scheduler_1.step()
        else:
            # Using custom defined scheduler
            for param_group in optimizer_1.param_groups:
                param_group['lr'] = scheduler_1(epoch)
