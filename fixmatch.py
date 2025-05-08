import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """高斯模糊变换"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
        
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class FixMatchLoss(nn.Module):
    """FixMatch一致性损失"""
    def __init__(self, threshold=0.95):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, logits_w, logits_s):
        """
        Args:
            logits_w: 弱增强图像的模型输出 [batch_size, num_classes]
            logits_s: 强增强图像的模型输出 [batch_size, num_classes]
        Returns:
            一致性损失值
        """
        # 生成伪标签
        probs_w = F.softmax(logits_w.detach(), dim=-1)
        max_probs, pseudo_labels = torch.max(probs_w, dim=-1)
        
        # 筛选高置信度样本
        mask = max_probs.ge(self.threshold).float()
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits_s, pseudo_labels, reduction='none')
        loss = (loss * mask).mean()
        
        return loss

def weak_augment():
    """弱数据增强 - 适用于28x28灰阶图像"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=28, padding=4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5,), (0.5,))
    ])

def strong_augment():
    """强数据增强 - 适用于28x28灰阶图像"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=28, padding=4),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.RandomApply([
            transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, random.uniform(0.5, 2.0)))
        ], p=0.8),
        transforms.RandomApply([
            transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, random.uniform(0.8, 1.2)))
        ], p=0.8),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_fixmatch_transforms():
    """获取FixMatch数据增强组合"""
    return {
        'weak': weak_augment(),
        'strong': strong_augment()
    }

def update_rampup_weight(epoch, rampup_length=80):
    """渐进式权重调整"""
    if rampup_length == 0:
        return 1.0
    epoch = np.clip(epoch, 0.0, rampup_length)
    phase = 1.0 - epoch / rampup_length
    return float(np.exp(-5.0 * phase * phase))