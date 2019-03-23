---
layout: default
title: 验证resneXt，densenet，mobilenet和SENet的特色结构
---
## 简介

图像分类对网络结构的要求，一个是精度，另一个是速度。这两个需求推动了网络结构的发展。
* resneXt：分组卷积，降低了网络参数个数。
* densenet：密集的跳连接。
* mobilenet：标准卷积分解成深度卷积和逐点卷积，即深度分离卷积。
* SENet：注意力机制。

简单起见，使用了[1]的代码，注释掉 layer4，作为基本框架resnet14。然后改变局部结构，验证分类效果。

## 实验结果
GPU：gtx1070
超参数：epochs=80,lr=0.001,optim=Adam
数据集：cifar10，batch_size=100

### 分组卷积

```python
# 3x3 convolution with grouping
def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False,groups=groups)
```
|_|参数个数(k)|GPU内存(M)|训练时间(s)|测试时间(s)|精度(%)|
|-|-|-|-|-|-|
|resnet14|195|617|665|0.34|87|
|分组=2|99|615|727|0.40|85|
|分组=4|50|615|834|0.50|81|

结论：卷积分组降低了参数个数，同时也降低了速度和精度。

### 密集连接
```python
    def forward(self, x): # basic block
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        out = self.layer1(x)
        out = self.relu(out)
        out2 = self.layer2(out)
        out2 = self.relu(out2)
        out3 = torch.cat([out,out2],1)
        out = self.layer3(out3)
        out4 = self.relu(out)
        out5 = torch.cat([out3,out4],1)
        out = self.layer4(out5) # back to the specified channels
        return out
```
|_|参数个数(k)|GPU内存(M)|训练时间(s)|测试时间(s)|精度(%)|
|-|-|-|-|-|-|
|resnet14|195|617|665|0.34|87|
|密集连接|341|679|703|0.43|88|

结论：参数个数和精度有所增加，速度下降一点点。

### 深度分离卷积
```python
def Conv2d(in_channels, out_channels,kernel_size=1,padding=0,stride=1):
    return nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels,kernel_size,stride=stride,padding=padding,groups=in_channels,bias=False),
            nn.Conv2d(in_channels, out_channels,1,bias=False),
        ])
```
|_|参数个数(k)|GPU内存(M)|训练时间(s)|测试时间(s)|精度(%)|
|-|-|-|-|-|-|
|resnet14|195|617|665|0.34|87|
|分组=2|99|615|727|0.40|85|
|分组=4|50|615|834|0.50|81|
|深度分离卷积|27|665|788|0.40|84|

结论：深度分离卷积降低了参数个数，同时也降低了速度和精度。与分组卷积（分组=4）相比，精度要高一点。
### 注意力机制
利用[2]的代码，修正通道个数
```python
    def forward(self, x): # BasicBlock
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        # attention
        original_out = out
        out = F.avg_pool2d(out,out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out
        out += residual
        out = self.relu(out)
        return out
```
|_|参数个数(k)|GPU内存(M)|训练时间(s)|测试时间(s)|精度(%)|
|-|-|-|-|-|-|
|resnet14|195|617|665|0.34|87|
|注意力|201|641|838|0.51|87|

结论：参数个数和精度变动不大，速度降低比较明显。

## 引用
[1] https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/02-intermediate/deep_residual_network/main.py
[2] https://github.com/miraclewkf/SENet-PyTorch/blob/master/se_resnet.py

## 参考文献
*   Chollet, François. Xception: Deep Learning with Depthwise Separable Convolutions[J]. 2016.
*   Xie S , Girshick R , Dollár, Piotr, et al. Aggregated Residual Transformations for Deep Neural Networks[J]. 2016.
*   Huang G, Liu Z, Laurens V D M, et al. Densely Connected Convolutional Networks[J]. 2016.
*   Howard A G , Zhu M , Chen B , et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications[J]. 2017.
*   Hu J , Shen L , Albanie S , et al. Squeeze-and-Excitation Networks[J]. 2017.
*   https://www.cnblogs.com/liaohuiqiang/p/9691458.html
