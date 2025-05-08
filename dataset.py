import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """支持有标签和无标签数据的Dataset类"""
    def __init__(self, image_file, label_file=None, transform=None):
        """
        Args:
            image_file (str): 图像数据CSV文件路径
            label_file (str, optional): 标签数据CSV文件路径
            transform (callable, optional): 数据增强函数
        """
        self.images = pd.read_csv(image_file).values.astype(np.float32)
        self.has_labels = label_file is not None
        if self.has_labels:
            self.labels = pd.read_csv(label_file).values.astype(np.int64)
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, 28, 28)  # 转为1x28x28
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
            
        if self.has_labels:
            label = torch.from_numpy(self.labels[idx]).squeeze()
            return image, label
        return image