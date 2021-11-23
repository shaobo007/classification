import torch
import pandas as pd
import torchvision
from models import get_pretrained_resnet34, get_pretrained_resnet50
from dataSet import DATA_DIR, load_data_cifar10
from PIL import Image


def try_all_gpus():
    """返回所有可用的GPU，如果没有就使用cpu"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


DEVICES = try_all_gpus()
net_predict = get_pretrained_resnet34(DEVICES)
net_predict.load_state_dict(torch.load(
    'cifar10-pretrained_resnet34-train.params'))
net_predict.eval()
BATCH_SIZE = 256
VAL_RATIO = 0.1
_, _, test_iter, _, num_test, classes = load_data_cifar10(
    DATA_DIR, VAL_RATIO, BATCH_SIZE)
'''
# 提交submission文件
preds = []
for X, _ in test_iter:
    y_hat = net_predict(X.to(DEVICES[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, num_test + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: classes[x])
df.to_csv('submission-resnet50_final1.csv', index=False)
'''

file_picture = '254000.png'
input_picture = Image.open(file_picture)
transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                                  [0.2023, 0.1994, 0.2010])])

input = transform_test(input_picture)
input = input.unsqueeze(0)
pred = net_predict(input.to(DEVICES[0]))
pred = pred.argmax(dim=1).type(torch.int32).cpu().numpy()
print('图片' + file_picture + '是'
      + classes[int(pred)])
