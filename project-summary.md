# 项目总结 - 基于SimCLR的Few-shot Learning实现

## 任务概述
本任务是使用Fashion-MNIST数据集进行弱监督图像分类。数据集包含：
- 12,000个有标签训练样本(每个类别1,200个)
- 48,000个无标签训练样本(每个类别4,800个)
- 10,000个测试样本

任务要求：
1. 实现学习算法并输出预测结果
2. 编写技术报告
3. 提交预测文件、报告和源代码

评估指标：Macro F1 Score

## 当前实现

### 方法选择
采用SimCLR(Simple Contrastive Learning of Representations)对比学习方法：
- 利用无标签数据进行自监督预训练
- 学习对数据增强不变的表示
- 后续可进行有监督微调

### 模型架构
```python
SimCLR(
  (encoder): CNNFeatureExtractor(
    (resnet): ResNet18(修改输入通道为1)
  )
  (projector): Sequential(
    (0): Linear(in_features=512, out_features=512)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=128)
  )
)
```

### 数据增强
针对Fashion-MNIST优化的增强策略：
1. 随机裁剪和缩放(70%-100%)
2. 随机水平翻转(50%概率)
3. 随机旋转(±10度, 80%概率)
4. 亮度和对比度调整(80%概率)
5. 高斯噪声(50%概率)

### 训练配置
- 批量大小: 512
- 学习率: 0.001
- 优化器: Adam
- 权重衰减: 1e-4
- 温度参数: 0.5
- 特征维度: 512
- 投影维度: 128

### 损失函数
NT-Xent(Normalized Temperature-scaled Cross Entropy)损失：
```math
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i,z_k)/\tau)}
```

## 当前状态
已完成SimCLR预训练阶段：
- 预训练模型保存在: `exp/20250507_23/experiment_001/`
  - `simclr-pretrained-best.pth`: 最佳模型
  - `simclr-pretrained-last.pth`: 最后模型
  - `slm-clr-train.log`: 训练日志

## 后续计划
使用有标签数据对预训练模型进行微调