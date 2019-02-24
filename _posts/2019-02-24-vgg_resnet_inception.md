---
layout: default
title: 比较 VGG, resnet和inception的图像分类效果
tags: cv
---

## 简介

VGG, resnet和inception是3种典型的卷积神经网络结构。

* VGG采用了3*3的卷积核，逐步扩大通道数量
* resnet中，每两层卷积增加一个旁路
* inception实现了卷积核的并联，然后把各自通道拼接到一起

简单起见，直接使用了[1]的代码来测试 resnet，然后用[2],[4]的代码替换[1]中的model，改了改通道，测 VGG 和 inception。

GPU是gtx1050，主板开始是 x79，后来坏了，换成 x470，GPU占比提高很多。
CPU占比始终100%

## 实验结果

超参数：epochs=80,lr=0.001,optim=Adam
数据集：cifar10

|_|参数个数(k)|训练时间(m)|精度(%)|GPU内存(M)|GPU占比(%)|
|-|-|-|-|-|-|
|resnet|195|22|88|607|99|
|vgg_bn|207|17|84|535|60|
|inception|107|19|80|613|98|

结论：条条道路通罗马。

## 附加实验

因为方便，注释掉 Batch Normalization，以及 Data Augmentation 又试了两次。

|_|参数个数(k)|训练时间(m)|精度(%)|GPU内存(M)|GPU占比(%)|
|-|-|-|-|-|-|
|resnet|195|22|88|607|99|
|resnet-BN|195|19|86|553|99|
|resnet-DA|195|22|64|607|99|

结论：Data Augmentation很重要

## 代码改动

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

print('# generator parameters:', sum(param.numel() for param in model.parameters()))
```
```python
class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(9 * 8 * 8, 64),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(64, num_classes),
        )

def vgg_bn(**kwargs):

    cfg = [16, 16, 'M', 32, 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M']
    model = VGG(make_layers(cfg, batch_norm=True), **kwargs)

```
```python
class Inception_v1(nn.Module):
    def __init__(self, num_classes=10):
        super(Inception_v1, self).__init__()

        #conv2d0
        self.conv1 = conv3x3(3, 6)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = nn.BatchNorm2d(6)

        self.inception_3a = Inception_base(1, 6, [[16], [16,32], [8, 16], [3, 16]]) #3a
        self.inception_3b = Inception_base(1, 80, [[40], [32,48], [12, 16], [3, 16]]) #3b
        self.max_pool_inc3= nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.inception_5a = Inception_base(1, 120, [[40], [32,48], [12, 16], [3, 16]]) #5a
        self.inception_5b = Inception_base(1, 120, [[40], [32,48], [12, 16], [3, 16]]) #5b
        self.avg_pool5 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)

        self.dropout_layer = nn.Dropout(0.4)
        self.fc = nn.Linear(120*9, num_classes)

```
## 引用
1. https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/02-intermediate/deep_residual_network/main.py
2. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
3. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
4.https://github.com/antspy/inception_v1.pytorch/blob/master/inception_v1.py
