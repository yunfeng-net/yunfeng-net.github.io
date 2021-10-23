---
layout: default
title: 比较语义分割的几种结构：FCN，UNET，SegNet，PSPNet和Deeplab
tags: cv
---

## 简介

语义分割：给图像的每个像素点标注类别。通常认为这个类别与邻近像素类别有关，同时也和这个像素点归属的整体类别有关。利用图像分类的网络结构,可以利用不同层次的特征向量来满足判定需求。现有算法的主要区别是如何提高这些向量的分辨率，以及如何组合这些向量。

## 几种结构

* 全卷积网络FCN：上采样提高分割精度，不同特征向量相加。[3]
* UNET：拼接特征向量；编码-解码结构；采用弹性形变的方式，进行数据增广；用边界加权的损失函数分离接触的细胞。[4]
* SegNet：记录池化的位置，反池化时恢复。[3]
* PSPNet：多尺度池化特征向量，上采样后拼接[3]
* Deeplab：池化跨度为1，然后接带孔卷积。
* ICNet：多分辨图像输入，综合不同网络生成结果。

## 实验设计

### 测试平台
* 采用[1]的代码，去掉one_hot，把损失函数改成交叉熵。
* 在验证过程引入pixel accuray和mIOU，代码见[2]
* 用颜色代码替换标签的类别代码，这样visdom可以显示多类别标签

### 数据集
* [1]自带数据集Bag，二分类，图像800*800，代码中转换到160*160。
    * 这个数据集很容易收敛，可以忽略优化器的影响，用来估计网络结构的性能上限。
* CamVid,代码见[2]，从视频中截取的，图像很相似。图像尺寸960*720。
* PASCAL VOC 2007/2012，代码参照[3]，图像差别大。

### 测试计划
* 在github上收集能成功运行的模型
* 在同等条件下比较技术细节：vgg16为基础结构
    * 比较单层特征向量进行转置卷积、上采样或者反池化后的效果
    * 比较特征向量的拼接和线性组合
    * 比较多尺度输入的网络组合

## 实验结果

超参数：epochs=50,lr=0.001,optim=SGD,momentum=0.7u 
数据集：Bag，resize(160,160)，batch_size=4
注意vgg16正确的层号，每层最后一个是池化。
```python
feats = list(models.vgg16(pretrained=True).features.children())
self.feat1 = nn.Sequential(*feats[0:5])
self.feat2 = nn.Sequential(*feats[5:10])
self.feat3 = nn.Sequential(*feats[10:17])
self.feat4 = nn.Sequential(*feats[17:24])
self.feat5 = nn.Sequential(*feats[24:31])
```

### 单层特征向量

1*1卷积+标签收缩（到对应层尺寸）

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5|8|82|90|1.1|
|4|8|86|93|1.0|
|3|6|80|90|1.0|

1*1卷积+上采样（2倍）+标签收缩

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5->4|8|72|85|1.1|
|4->3|6|80|90|1.0|
|3->2|5|78|88|1.0|

1*1卷积+转置卷积（2倍）+标签收缩

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5->4|8|79|89|1.1|
|4->3|6|84|92|1.0|
|3->2|5|80|90|1.0|

反池化（2倍）+1*1卷积+标签收缩

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5->4|8|84|92|1.1|
|4->3|7|87|94|1.1|
|3->2|5|84|91|1.0|

池化（stride=1）+2*2卷积（stride=1,padding=1）+标签收缩

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5->4|8|84|92|1.1|
|4->3|7|89|95|1.0|
|3->2|7|80|90|1.1|

### 多层特征向量组合
* 理论上，求和是拼接+1*1卷积的一个特例。

上采样（逐层，直到原始尺寸）+1*1卷积+求和（FCN）

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5|8|82|91|1.2|
|5+4|8|88|94|1.2|
|5+4+3|9|88|94|1.2|

上采样（逐层，直到原始尺寸）+1*1卷积+拼接（UNET'）

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5|8|82|91|1.2
|5+4|9|87|93|1.2
|5+4+3|9|89|94|1.1

上采样（直接达到原始尺寸）+1*1卷积+拼接（PSPNET'）

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5|8|84|92|1.2
|5+4|9|87|93|1.2
|5+4+3|8|88|94|1.2

反池化（逐层）+1*1卷积+上采样（SegNet'）

|网络层|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|5|8|82|91|1.1
|5->4|8|88|94|1.1
|5->4->3|9|89|95|1.1

### 附加实验
epochs=100，lr=3e-3

|网络|单epoch时间(s)|mIOU(%)|pixel-acc(%)|GPU(G)|
|-|-|-|-|-|
|PSPNET(反池化)|8|91|96|1.1
|PSPNET(池化,stride=1)|9|91|96|1.2

## 引用
1. https://github.com/bat67/pytorch-FCN-easiest-demo
2. https://github.com/pochih/FCN-pytorch
3. https://github.com/bodokaiser/piwise
4. https://github.com/jaxony/unet-pytorch/

## 参考文献
*  Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
*  Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]// International Conference on Medical Image Computing & Computer-assisted Intervention. 2015.
*  Zhao H , Shi J , Qi X , et al. Pyramid Scene Parsing Network[J]. 2016.
*   Chen L C , Papandreou G , Schroff F , et al. Rethinking Atrous Convolution for Semantic Image Segmentation[J]. 2017.    
*   Zhao H, Qi X, Shen X, et al. ICNet for Real-Time Semantic Segmentation on High-Resolution Images[J]. 2017.
