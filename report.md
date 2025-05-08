# 基于FixMatch的Fashion-MNIST弱监督图像分类报告

本报告介绍了采用FixMatch半监督学习算法处理Fashion-MNIST数据集弱监督图像分类任务的过程。此任务中，有标签数据量较少，而无标签数据量较大。改进的ResNet18被用作特征提取网络，并结合FixMatch策略来有效利用无标签数据。最终，该模型在验证集上取得了0.9076的Macro F1分数，表现优于仅使用有标签数据训练的传统监督学习模型。

## 1. 问题理解与分析

### 1.1 任务描述
本次任务是使用Fashion-MNIST数据集进行图像分类。该数据集包含28x28像素的灰度服装图像，共10个类别。
任务的特点是弱监督性质：有标签训练数据包含12,000张图片及其对应标签；无标签训练数据包含48,000张图片，这部分数据需要被充分利用；测试数据包含10,000张图片，需要对这些图片进行分类预测。评估指标为Macro F1 Score。

### 1.2 问题分析
主要挑战在于如何有效地利用大量的无标签数据来提升模型在少量有标签数据基础上的性能。仅使用有标签图片进行监督学习，模型的泛化能力会受限。引入无标签数据使得半监督学习算法成为合适的选择。

## 2. 算法动机与背景介绍

### 2.1 算法动机
面对大量无标签数据和少量有标签数据的场景，半监督学习（SSL）策略被采用。SSL算法旨在同时从有标签和无标签数据中学习，以期获得更好的性能。

在众多SSL算法中，FixMatch被选用，因其相对简洁且效果显著。FixMatch的核心思想是结合伪标签和一致性正则化。它首先利用模型对无标签样本（经过弱增强）的预测生成伪标签，然后只保留那些置信度较高的伪标签，用这些高质量的伪标签去监督模型对同一无标签样本（经过强增强）的预测。

### 2.2 背景介绍: FixMatch
FixMatch主要流程首先是对有标签数据使用标准的交叉熵损失进行训练。其次，对于无标签数据，先对无标签样本进行弱增强（如随机翻转和移位），然后将弱增强后的样本输入模型获得预测概率。如果最大预测概率超过一个预设的阈值τ，则将对应的类别作为该无标签样本的伪标签。接着，对这同一个无标签样本进行强增强（如RandAugment或Cutout）。最后，使用之前生成的伪标签，计算模型在强增强样本上的预测与该伪标签之间的交叉熵损失，以此实现一致性正则化。总损失函数是带标签数据的有监督损失和无标签数据的一致性损失的加权和。

## 3. 算法技术细节

### 3.1 模型架构
CNN 被采用作为核心模型。

特征提取器采用 ResNet 结构，具体为修改版的 resnet18。针对Fashion-MNIST的单通道灰度图 (1x28x28)，resnet18 的第一个卷积层 self.resnet.conv1 被修改为适应单通道输入：
```python
self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
```
同时，原始 resnet18 的 maxpool 层被替换为 nn.Identity()：
```python
self.resnet.maxpool = nn.Identity()
```

分类头采用一个简单的多层感知机 (MLP) MLPClassifier。它包含一个从 feature_dim 到 hidden_dim=256 的全连接层，接一个ReLU激活函数，最后是一个从 hidden_dim 到 num_classes=10 的全连接层。

### 3.2 数据处理与增强
数据加载由 ImageDataset 类处理。图像被转换为 (1, 28, 28) 的张量：
```python
image = self.images[idx].reshape(1, 28, 28)  # 转为1x28x28
image = torch.from_numpy(image)
```
对于FixMatch算法的数据增强，有标签数据应用了标准的弱数据增强，如随机水平翻转。无标签数据则区分弱增强和强增强：弱增强用于生成伪标签，主要使用了随机水平翻转；强增强用于一致性正则化，采用了RandAugment等策略。

### 3.3 训练配置与FixMatch实现要点
训练配置的关键实现代码如下：
```python
class FixMatchLoss(nn.Module):
    def forward(self, logits_w, logits_s):
        probs_w = F.softmax(logits_w.detach(), dim=-1)
        max_probs, pseudo_labels = torch.max(probs_w, dim=-1)
        mask = max_probs.ge(self.threshold).float()
        loss = F.cross_entropy(logits_s, pseudo_labels, reduction='none')
        return (loss * mask).mean()

def weak_augment():  # 弱增强
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=28, padding=4),
        transforms.Normalize((0.5,), (0.5,))
    ])

def strong_augment():  # 强增强
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=28, padding=4),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Normalize((0.5,), (0.5,))
    ])
```
训练参数设置为：batch_size为512，mu（无标签数据批次大小与有标签数据批次大小的比例）为7，epochs为100，学习率lr为0.0005，伪标签置信度阈值τ为0.95，无监督损失权重lambda_u为1.0，学习率或lambda_u的预热期epoch数rampup_length为80。

损失函数:
总损失 $L = L_s + \lambda_u L_u$。

$L_s$ 是有标签数据的监督损失（交叉熵损失）：
$L_s = \frac{1}{B} \sum_{i=1}^{B} H(y_i, p_m(x_i^w))$
其中 $B$ 是有标签数据的批次大小，$x_i^w$ 是弱增强后的有标签样本，$y_i$ 是其真实标签。

$L_u$ 是无标签数据的一致性损失（交叉熵损失）：
$L_u = \frac{1}{\mu B} \sum_{j=1}^{\mu B} \mathbb{I}(\max(p_m(u_j^w)) \ge \tau) H(\hat{y}_j, p_m(u_j^s))$
其中 $\hat{y}_j = \text{argmax}(p_m(u_j^w))$ 是伪标签。



## 4. 性能描述与分析

### 4.1 实验结果
采用ResNet18和FixMatch算法进行训练，在验证集上取得的最佳Macro F1分数为 0.9076。这个结果是在训练到第97个epoch时达成的。训练过程中Macro F1从初始值逐步提升至0.9076，显示了模型通过FixMatch策略从有标签和无标签数据中有效学习的过程。

### 4.2 与baseline对比

| 算法                |  Macro F1 |
|---------------------|--------------------------------|
| Decision Tree       | 0.7457                         |
| MLP                 | 0.7986                         |
| GBDT                | 0.8468                         |
| Random Forest       | 0.8472                         |
| **FixMatch + ResNet18** | **0.9076**                         |

0.9076的Macro F1分数在Fashion-MNIST这类任务上是一个具有竞争力的结果，尤其是在弱监督的设定下。

