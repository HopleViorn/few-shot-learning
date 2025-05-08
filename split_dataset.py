import pandas as pd
import numpy as np
import os

# 确保输出目录存在
os.makedirs('data', exist_ok=True)

# 读取标记数据和标签
labeled_images = pd.read_csv('data/train_image_labeled.csv')
labels = pd.read_csv('data/train_label.csv')

# 获取数据总量
total_samples = len(labeled_images)
validation_size = int(total_samples * 0.1)  # 10%的数据用于验证集

# 随机打乱索引
indices = np.arange(total_samples)
np.random.seed(42)  # 设定随机种子以确保结果可复现
np.random.shuffle(indices)

# 分割为训练集和验证集的索引
val_indices = indices[:validation_size]
train_indices = indices[validation_size:]

# 创建新的训练集和验证集
train_images = labeled_images.iloc[train_indices].reset_index(drop=True)
val_images = labeled_images.iloc[val_indices].reset_index(drop=True)

# 按照对应索引分割标签
train_labels = labels.iloc[train_indices].reset_index(drop=True)
val_labels = labels.iloc[val_indices].reset_index(drop=True)

# 保存到文件
train_images.to_csv('data/train_image_labeled_new.csv', index=False)
val_images.to_csv('data/val_image_labeled.csv', index=False)
train_labels.to_csv('data/train_label_new.csv', index=False)
val_labels.to_csv('data/val_label.csv', index=False)

# 打印数据集统计信息
print(f"总样本数: {total_samples}")
print(f"训练集样本数: {len(train_images)} ({len(train_images)/total_samples*100:.2f}%)")
print(f"验证集样本数: {len(val_images)} ({len(val_images)/total_samples*100:.2f}%)") 