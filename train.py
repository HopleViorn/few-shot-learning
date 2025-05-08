import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import numpy as np
import os
import time
from datetime import datetime

from dataset import ImageDataset
from cnn_model import CNNClassifier
from fixmatch import FixMatchLoss, get_fixmatch_transforms, update_rampup_weight

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = CNNClassifier(num_classes=10).to(self.device)
        
        # 损失函数
        self.supervised_loss = nn.CrossEntropyLoss()
        self.unsupervised_loss = FixMatchLoss(threshold=config['threshold'])
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # 数据增强
        self.transforms = get_fixmatch_transforms()
        
        # 创建数据加载器
        self._create_dataloaders()
        
        # 创建实验目录和日志
        timestamp = datetime.now()
        hour_dir = os.path.join('exp', timestamp.strftime('%Y%m%d_%H'))
        self.exp_dir = os.path.join(hour_dir, timestamp.strftime('%M%S'))
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 日志记录
        self.writer = SummaryWriter(self.exp_dir)
        
        # 创建训练日志文件
        self.log_file = os.path.join(self.exp_dir, 'training.log')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {timestamp}\n")
            f.write(f"Config: {config}\n\n")
        
    def _create_dataloaders(self):
        """创建有标签和无标签数据的数据加载器"""
        # 有标签数据
        labeled_train_dataset = ImageDataset(
            'data/train_image_labeled_new.csv',
            'data/train_label_new.csv',
            transform=self.transforms['weak']  # 有标签数据使用弱增强
        )
        self.labeled_train_loader = DataLoader(
            labeled_train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=6
        )
        
        unlabeled_dataset = ImageDataset(
            'data/train_image_unlabeled.csv',
            transform=None 
        )
        self.unlabeled_train_loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.config['batch_size'] * self.config['mu'],
            shuffle=True,
            num_workers=6
        )
        
        val_dataset = ImageDataset(
            'data/val_image_labeled.csv',
            'data/val_label.csv',
            transform=None 
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=6
        )
    
    def train(self):
        """训练循环"""
        best_f1 = 0.0
        
        for epoch in range(self.config['epochs']):
            train_metrics = self._train_epoch(epoch)
            
            val_metrics = self.validate()
            
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self._save_model(epoch, best_f1)
                
            log_msg = (f"Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f} | "
                      f"Best F1: {best_f1:.4f}")
            print(log_msg)
            with open(self.log_file, 'a') as f:
                f.write(log_msg + '\n')
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        labeled_iter = iter(self.labeled_train_loader)
        
        unsup_weight = self.config['lambda_u'] * update_rampup_weight(epoch, self.config['rampup_length'])
        
        for batch_idx, unlabeled_images in enumerate(self.unlabeled_train_loader):
            try:
                labeled_images, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_train_loader)
                labeled_images, labels = next(labeled_iter)
            
            labeled_images = labeled_images.to(self.device)
            labels = labels.to(self.device)
            unlabeled_images = unlabeled_images.to(self.device)
            
            # 对无标签数据同时生成弱增强和强增强版本
            with torch.no_grad():
                weak_augmented = self.transforms['weak'](unlabeled_images)
            strong_augmented = self.transforms['strong'](unlabeled_images.clone())
            
            # 前向传播 - 有标签数据
            outputs_labeled = self.model(labeled_images)
            sup_loss = self.supervised_loss(outputs_labeled, labels)
            
            # 前向传播 - 无标签数据
            with torch.no_grad():
                # 弱增强版本用于生成伪标签
                outputs_weak = self.model(weak_augmented)
            
            # 强增强版本用于一致性正则化
            outputs_strong = self.model(strong_augmented)
            
            # 计算无监督损失
            unsup_loss = self.unsupervised_loss(outputs_weak, outputs_strong)
            
            # 总损失
            loss = sup_loss + unsup_weight * unsup_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'loss': total_loss / len(self.unlabeled_train_loader)
        }
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # 计算Macro F1 Score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'f1': f1
        }
    
    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """记录指标到TensorBoard"""
        self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
    
    def _save_model(self, epoch, f1):
        """保存模型"""
        # 删除旧的checkpoint
        for old_file in os.listdir(self.exp_dir):
            if old_file.startswith('model_epoch') and old_file.endswith('.pth'):
                os.remove(os.path.join(self.exp_dir, old_file))
        
        # 保存新的best checkpoint
        save_path = os.path.join(self.exp_dir, f'model_epoch{epoch}_f1{f1:.4f}.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存到 {save_path}")
        with open(self.log_file, 'a') as f:
            f.write(f"Saved model checkpoint: {save_path}\n")

if __name__ == '__main__':
    config = {
        'batch_size': 64,
        'mu': 7,  # 无标签数据batch size倍数
        'epochs': 100,
        'lr': 0.0005,
        'threshold': 0.95,  # 伪标签阈值
        'lambda_u': 1.0,  # 无监督损失权重
        'rampup_length': 80  # 权重渐进长度
    }
    
    trainer = Trainer(config)
    trainer.train()