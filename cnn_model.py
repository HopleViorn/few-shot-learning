import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        # 使用ResNet18作为基础模型
        self.resnet = resnet18(pretrained=False)
        
        # 修改输入通道数为1(灰度图)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改全连接层输出特征维度
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)
        
    def forward(self, x):
        # 输入x形状: (batch_size, 1, 28, 28)
        # ResNet18会自动处理输入尺寸
        features = self.resnet(x)
        return features

# 示例用法
if __name__ == "__main__":
    model = CNNFeatureExtractor()
    dummy_input = torch.randn(1, 1, 28, 28)  # 模拟Fashion-MNIST输入
    features = model(dummy_input)
    print(f"输出特征向量维度: {features.shape}")  # 应为 [1, 512]

class LinearClassifier(nn.Module):
    """线性分类器用于评估预训练特征"""
    def __init__(self, feature_dim=512, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)