import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from cnn_model import CNNClassifier
from dataset import ImageDataset
from fixmatch import get_fixmatch_transforms

def evaluate_model(model_path, test_csv, label_csv=None):
    """评估模型性能"""
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = CNNClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 数据加载
    transforms = get_fixmatch_transforms()['weak']
    test_dataset = ImageDataset(test_csv, label_csv, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 预测
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 计算指标
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds))
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    
    return macro_f1

if __name__ == '__main__':
    # 配置参数
    model_path = "checkpoints/best_model.pth"  # 请替换为实际模型路径
    test_csv = "data/test_image.csv"
    label_csv = "data/train_label.csv"  # 测试集标签
    
    # 评估模型
    print("开始模型评估...")
    f1_score = evaluate_model(model_path, test_csv, label_csv)
    print(f"评估完成，Macro F1 Score: {f1_score:.4f}")