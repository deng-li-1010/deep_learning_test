# d2l NiN 代码逐行完整讲解 + 直观尺寸例子
原文代码位置：https://zh.d2l.ai/chapter_convolutional-modern/nin.html
框架：PyTorch，先拆分 `nin_block` 模块，再搭完整NiN网络，再训练。

## 一、导入依赖（前置代码）
```python
import torch
from torch import nn
from d2l import torch as d2l
```
- `torch`：pytorch主库
- `nn`：神经网络层（Conv2d、ReLU、MaxPool2d、Dropout、AvgPool2d）
- `d2l`：本书工具包，数据加载、训练、绘图

# 二、核心：nin_block 1×1卷积块 逐行拆解
## 完整块代码
```python
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        # 两层1×1卷积，等价每个像素内部两层MLP
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )
```
### 逐行解释
1. `def nin_block(in_channels, out_channels, kernel_size, strides, padding):`
   函数入参：
- `in_channels`：输入特征图通道数
- `out_channels`：本块最终输出通道
- `kernel_size`：第一层空间卷积核大小（一般3或5）
- `strides`：第一层卷积步幅
- `padding`：第一层填充

2. `return nn.Sequential(...)`
   Sequential：把层按顺序打包，数据从上到下依次走。

3. 第一层：空间卷积
```python
nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
nn.ReLU(),
```
作用：**提取图像空间局部特征**（边缘、纹理），改变长宽、调整通道到`out_channels`。

4. 第二层：第一个1×1卷积
```python
nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
```
核1×1，stride默认1，padding0：
- 特征图 **长宽完全不变**
- 只在通道维度做线性变换：对每个像素的所有通道加权融合
- 等价：每个像素单独一层全连接

5. 第三层：第二个1×1卷积
```python
nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
```
再一层1×1+激活，叠加两层非线性。
> 这就是NiN精髓：每个像素自带两层小型MLP，增强特征组合能力。

### nin_block 尺寸演算例子
输入：`batch=1, C=3, H=224, W=224`（224×224原图RGB）
调用：`nin_block(3, 192, kernel_size=5, strides=1, padding=2)`
- 5×5卷积，padding=2 → 长宽不变仍224，通道变为192
- 两次1×1卷积：H=224,W=224,C=192 全程不变
  输出shape：`(1, 192, 224, 224)`

# 三、搭建完整NiN网络 逐行拆解
```python
net = nn.Sequential(
    # 第1个NiN块
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    # 第2个NiN块
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    # 第3个NiN块
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 关键：最后一块输出通道 = 类别数10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 全局平均池化替代全连接
    nn.AdaptiveAvgPool2d((1, 1)),
    # 把 (batch,10,1,1) 压成 (batch,10)
    nn.Flatten()
)
```
## 逐层走一遍 + 尺寸变化示例（输入 1×1×224×224 Fashion-MNIST灰度图）
输入shape：`(1, 1, 224, 224)`

### 1）第一块 + 最大池化
```python
nin_block(1, 96, kernel_size=11, strides=4, padding=0),
nn.MaxPool2d(3, stride=2),
```
- 11×11 conv stride4：224 → 54，通道96 → (1,96,54,54)
- MaxPool 3×3 stride2：54→26 → (1,96,26,26)

### 2）第二块 + 最大池化
```python
nin_block(96, 256, kernel_size=5, strides=1, padding=2),
nn.MaxPool2d(3, stride=2),
```
- 5×5 padding2 长宽不变26，通道256 → (1,256,26,26)
- MaxPool3 stride2：26→12 → (1,256,12,12)

### 3）第三块 + 最大池化 + Dropout
```python
nin_block(256, 384, kernel_size=3, strides=1, padding=1),
nn.MaxPool2d(3, stride=2),
nn.Dropout(0.5),
```
- 3×3 padding1 长宽不变12，通道384 → (1,384,12,12)
- MaxPool3 stride2：12→5 → (1,384,5,5)
- Dropout(0.5)：训练时随机一半通道置0，防过拟合，尺寸不变

### 4）最后关键NiN块（通道=分类数10）
```python
nin_block(384, 10, kernel_size=3, strides=1, padding=1),
```
3×3 padding1，长宽保持5不变，通道压缩为10
输出shape：`(1, 10, 5, 5)`
> 重点：10个通道分别对应10个分类，每个通道是该类的响应热力图

### 5）全局自适应平均池化（替代全连接核心）
```python
nn.AdaptiveAvgPool2d((1, 1)),
```
`AdaptiveAvgPool2d((1,1))` = 全局平均池化GAP
不管输入H、W是多少，强制输出高宽=1：
对每个通道单独求整张特征图所有像素平均值
输入(1,10,5,5) → 输出(1,10,1,1)
- 通道0均值 = 类别0得分
- 通道1均值 = 类别1得分
  ……
  **完全不需要巨型全连接层！**

### 6）Flatten展平
```python
nn.Flatten()
```
把 `(batch,10,1,1)` 变成 `(batch,10)`，直接喂交叉熵损失做分类。

## 四、测试网络尺寸的演示代码（书上自带）
```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
运行后会逐层打印每一层输出维度，直观看到尺寸收缩、通道变化，验证上面的演算。

## 五、读取数据 & 训练代码简单说明
```python
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.1, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
1. `load_data_fashion_mnist(..., resize=224)`：把28×28手写服饰图放大到224×224适配网络输入
2. `train_ch6`：d2l封装训练函数，包含前向传播、损失反向、SGD优化、测试集评估、绘图

# 六、从反向传播角度：网络是怎么把通道和类别绑定的？

前向：图片进来，算出 10 个通道均值分数；
和真实标签（比如标签 1 = 裤子）算 loss；
反向梯度更新所有卷积权重：
增大能让通道 1（裤子通道）在裤子图片上变亮的权重；
压低通道 0/2~9 在裤子图片上的激活；
反复迭代后，网络自动学会分工：
每个通道只对自己对应的那一类物体产生大面积高激活，对其他物体几乎不响应。


