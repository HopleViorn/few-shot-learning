import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from simclr import SimCLR
from cnn_model import CNNFeatureExtractor

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
            self.labels = pd.read_csv(label_file).values.astype(np.long)
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, 28, 28)  # 转为1x28x28
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
            
        if self.has_labels:
            label = self.labels[idx]
            return image, label
        return image

class SLM_CLR_Trainer:
    """SimCLR训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = SimCLR(
            feature_dim=config['feature_dim'],
            projection_dim=config['projection_dim'],
            temperature=config['temperature']
        ).to(self.device)
        
        # 优化器
        if config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        else:  # SGD
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['lr'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
            
        # 数据加载器
        self.train_loader = self._get_data_loader(
            image_file=config['train_image_file'],
            label_file=config.get('train_label_file'),
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        # 实验目录管理
        self.exp_dir = config.get('exp_dir', 'exp')
        self.experiment_dir = self._get_experiment_dir()
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 日志记录
        self.log_file = os.path.join(
            self.experiment_dir,
            "slm-clr-train.log"
        )
        
    def _get_data_loader(self, image_file, label_file=None, batch_size=32, shuffle=False):
        """创建数据加载器"""
        dataset = ImageDataset(
            image_file=image_file,
            label_file=label_file,
            transform=None  # SimCLR内部已包含数据增强
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=6,
            pin_memory=True
        )
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}', unit='batch') as pbar:
            for batch_idx, images in enumerate(pbar):
                images = images.to(self.device)
                
                # 前向传播
                z1, z2 = self.model(images)
                loss = self.model.nt_xent_loss(z1, z2)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                if batch_idx % self.config['log_interval'] == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    log_msg = (
                        f"Train Epoch: {epoch} [{batch_idx * len(images)}/{len(self.train_loader.dataset)} "
                        f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {avg_loss:.6f}"
                    )
                    pbar.set_postfix({'loss': avg_loss})
                    self._log_to_file(log_msg)
                    
        return total_loss / len(self.train_loader)
    
    def _log_to_file(self, message):
        """记录日志到文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            
    def save_model(self, path, epoch=None):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'config': self.config
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
        self._log_to_file(f"Model saved to {path}")
        
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        print(f"Model loaded from {path}")
        self._log_to_file(f"Model loaded from {path}")
        return epoch
        
    def _get_experiment_dir(self):
        """获取当前实验目录"""
        # 按小时创建父目录
        hour_dir = os.path.join(
            self.exp_dir,
            datetime.now().strftime('%Y%m%d_%H')
        )
        os.makedirs(hour_dir, exist_ok=True)
        
        # 查找当前实验编号
        exp_num = 1
        while os.path.exists(os.path.join(hour_dir, f"experiment_{exp_num:03d}")):
            exp_num += 1
            
        return os.path.join(hour_dir, f"experiment_{exp_num:03d}")
        
    def train(self, epochs):
        """完整训练流程"""
        best_loss = float('inf')
        
        with tqdm(range(1, epochs + 1), desc='Training', unit='epoch') as pbar:
            for epoch in pbar:
                avg_loss = self.train_epoch(epoch)
                
                # 保存last模型
                last_path = os.path.join(
                    self.experiment_dir,
                    f"{self.config.get('model_type', 'simclr')}-"
                    f"{self.config.get('train_phase', 'pretrained')}-last.pth"
                )
                self.save_model(last_path, epoch)
                
                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(
                        self.experiment_dir,
                        f"{self.config.get('model_type', 'simclr')}-"
                        f"{self.config.get('train_phase', 'pretrained')}-best.pth"
                    )
                    self.save_model(best_path, epoch)
                    
                # 更新进度条
                log_msg = f"Epoch {epoch} completed. Average Loss: {avg_loss:.6f}"
                pbar.set_postfix({'loss': avg_loss})
                self._log_to_file(log_msg)

# 默认配置
DEFAULT_CONFIG = {
    'feature_dim': 512,
    'projection_dim': 128,
    'temperature': 0.5,
    'optimizer': 'Adam',  # 或'SGD'
    'lr': 0.001,
    'weight_decay': 1e-4,
    'momentum': 0.9,  # SGD使用
    'batch_size': 256,
    'log_interval': 10,
    'train_image_file': 'data/train_image_unlabeled.csv',  # 无监督学习使用无标签数据
    'log_dir': 'logs'
}

if __name__ == "__main__":
    # 示例训练流程
    trainer = Trainer(DEFAULT_CONFIG)
    trainer.train(epochs=50)