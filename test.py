
import torch
import numpy as np
from dataset import ImageDataset
from cnn_model import CNNClassifier
from torch.utils.data import DataLoader

def main():
    # 加载最佳模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNClassifier(num_classes=10).to(device)
    model.load_state_dict(torch.load('exp/20250508_19/0438/model_epoch96_f10.9076.pth'))
    model.eval()

    # 创建测试数据集
    test_dataset = ImageDataset('data/test_image.csv', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=6)

    # 生成预测
    all_preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # 输出预测结果到文件
    with open('502024330034.txt', 'w') as f:
        for pred in all_preds:
            f.write(f"{pred}\n")

    print(f"预测结果已保存到502024330034.txt，共{len(all_preds)}条预测")

if __name__ == '__main__':
    main()