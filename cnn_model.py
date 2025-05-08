import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        self.resnet = resnet18(pretrained=False)
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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

class CNNClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(feature_dim)
        self.classifier = MLPClassifier(feature_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

