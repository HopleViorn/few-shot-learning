import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(ResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.resnet.maxpool = nn.Identity()
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)
        
    def forward(self, x):
        features = self.resnet(x)
        return features

class MLPClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=10, hidden_dim=256):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 简化CNN模型，更适合28x28的小图像
class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=512):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 14x14
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 7x7
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 3x3
        x = x.view(-1, 128 * 3 * 3)
        features = F.relu(self.fc1(x))
        return features

class CNNClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=10, model_type='resnet'):
        super(CNNClassifier, self).__init__()
        if model_type == 'resnet':
            self.feature_extractor = ResNet(feature_dim)
        elif model_type == 'simple':
            self.feature_extractor = SimpleCNN(feature_dim)
            
        self.classifier = MLPClassifier(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

