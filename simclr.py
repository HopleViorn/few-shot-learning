import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from cnn_model import CNNFeatureExtractor

class SimCLR(nn.Module):
    def __init__(self, feature_dim=512, projection_dim=128, temperature=0.5):
        super(SimCLR, self).__init__()
        self.encoder = CNNFeatureExtractor(feature_dim=feature_dim)
        self.temperature = temperature
        
        # 投影头 (g→h)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
        # 数据增强 (针对Fashion-MNIST的特点优化)
        def augment(x):
            # x形状: [batch_size, 1, 28, 28]
            batch_size = x.size(0)
            augmented = []
            
            for i in range(batch_size):
                img = x[i]  # [1, 28, 28]
                
                # 随机裁剪和缩放 (更保守的裁剪范围: 70%-100%)
                crop_size = int(28 * (0.7 + 0.3 * torch.rand(1).item()))
                top = int((28 - crop_size) * torch.rand(1).item())
                left = int((28 - crop_size) * torch.rand(1).item())
                img = img[:, top:top+crop_size, left:left+crop_size]
                img = F.interpolate(img.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False).squeeze(0)
            
                # 随机水平翻转 (50%概率)
                if torch.rand(1).item() < 0.5:
                    img = torch.flip(img, [2])
                
                # 随机小角度旋转 (±10度, 80%概率)
                if torch.rand(1).item() < 0.8:
                    angle = 20 * torch.rand(1).item() - 10  # -10到+10度
                    # 创建旋转矩阵
                    rad = torch.tensor(angle) * math.pi / 180
                    device = x.device  # 获取输入张量的设备
                    theta = torch.tensor([
                        [torch.cos(rad), -torch.sin(rad), 0],
                        [torch.sin(rad), torch.cos(rad), 0]
                    ], dtype=torch.float, device=device)
                    grid = F.affine_grid(theta.unsqueeze(0), torch.Size((1, 1, 28, 28)), align_corners=False).to(device)
                    img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False).squeeze(0)
                
                # 随机亮度和对比度调整 (80%概率)
                if torch.rand(1).item() < 0.8:
                    # 亮度调整: ±0.1
                    brightness = 0.2 * torch.rand(1).item() - 0.1
                    img = torch.clamp(img + brightness, 0, 1)
                    
                    # 对比度调整: 0.8-1.2
                    contrast = 0.8 + 0.4 * torch.rand(1).item()  # 0.8-1.2
                    img = torch.clamp((img - 0.5) * contrast + 0.5, 0, 1)
                
                # 随机高斯噪声 (50%概率)
                if torch.rand(1).item() < 0.5:
                    noise = torch.randn_like(img) * 0.05  # 更小的噪声幅度
                    img = torch.clamp(img + noise, 0, 1)
                
                # 标准化到[-1,1]
                img = (img - 0.5) / 0.5
                
                augmented.append(img)
            
            return torch.stack(augmented)  # [batch_size, 1, 28, 28]
            
        self.transform = augment
    
    def forward(self, x):
        # 生成两个增强视图
        x1 = self.transform(x)
        x2 = self.transform(x)
        
        # 获取特征表示
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # 投影到对比空间
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        return z1, z2
    
    def nt_xent_loss(self, z1, z2):
        # 归一化投影向量
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        batch_size = z1.size(0)
        # 合并所有投影向量
        z = torch.cat([z1, z2], dim=0)
        
        # 计算相似度矩阵
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temperature)
        
        # 创建正样本对掩码
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = mask.repeat(2, 2)
        
        # 排除对角线元素
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)
        
        # 计算正样本相似度
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        # 计算NT-Xent损失
        loss = -torch.log(pos_sim / sim_matrix.sum(dim=1))
        return loss.mean()

# 示例用法
if __name__ == "__main__":
    model = SimCLR()
    dummy_input = torch.randn(1, 1, 28, 28)  # 模拟Fashion-MNIST输入
    z1, z2 = model(dummy_input)
    loss = model.nt_xent_loss(z1, z2)
    print(f"对比损失: {loss.item():.4f}")