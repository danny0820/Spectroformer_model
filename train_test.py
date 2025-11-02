import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from torchvision import transforms
from model import Spectroformer

# 簡單的 L1 損失開始測試
class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        return self.l1(pred, target)

# 診斷函數
def diagnose_output(output, target, name=""):
    print(f"\n=== 診斷 {name} ===")
    print(f"輸出範圍: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"輸出平均: {output.mean().item():.3f}")
    print(f"輸出標準差: {output.std().item():.3f}")
    print(f"目標範圍: [{target.min().item():.3f}, {target.max().item():.3f}]")
    print(f"是否有 NaN: {torch.isnan(output).any().item()}")
    print(f"是否有 Inf: {torch.isinf(output).any().item()}")

# 簡化的資料集
class SimpleDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.path = os.path.join(path, mode)
        self.input_path = os.path.join(self.path, 'input')
        self.gt_path = os.path.join(self.path, 'gt')
        self.files = sorted(os.listdir(self.input_path))[:10]  # 只用10張測試
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        name = self.files[idx]
        
        # 讀取圖片
        input_img = Image.open(os.path.join(self.input_path, name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_path, name)).convert('RGB')
        
        # 轉換為張量 - 不做任何正規化
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()  # 只轉換為 [0, 1] 範圍
        ])
        
        input_tensor = transform(input_img)
        gt_tensor = transform(gt_img)
        
        return input_tensor, gt_tensor, name

def train_debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用最小的模型配置
    model = Spectroformer(
        dim=16,  # 極小的維度
        num_blocks=[1, 1, 1, 1],  # 最少的區塊
        num_refinement_blocks=1,
        num_heads=[1, 1, 1, 1],
        ffn_expansion_factor=2.0,
        bias=True  # 使用 bias
    ).to(device)
    
    # 簡單的初始化
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # 小的初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    print(f"模型參數: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 資料載入
    dataset = SimpleDataset('/danny/Spectroformer_model/LSUI_test', 'train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 優化器 - 使用很小的學習率
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # 只使用 L1 損失
    criterion = nn.L1Loss()
    
    # 訓練
    model.train()
    for epoch in range(10):
        total_loss = 0
        
        for i, (input_img, target_img, name) in enumerate(dataloader):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            
            # 前向傳播
            output = model(input_img)
            
            # 診斷第一個 batch
            if epoch == 0 and i == 0:
                diagnose_output(output, target_img, f"Epoch {epoch} Batch {i}")
            
            # 計算損失
            loss = criterion(output, target_img)
            
            # 檢查損失
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"異常損失: {loss.item()}")
                continue
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每 5 個 batch 輸出一次
            if i % 5 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} 平均損失: {avg_loss:.4f}")
        
        # 保存一個輸出樣本
        if epoch % 2 == 0:
            with torch.no_grad():
                model.eval()
                input_img, target_img, _ = next(iter(dataloader))
                input_img = input_img.to(device)
                output = model(input_img)
                
                # 保存圖片
                save_comparison(input_img[0], output[0], target_img[0], f"debug_epoch_{epoch}.jpg")
                model.train()

def save_comparison(input_img, output, target, filename):
    """保存對比圖"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 轉換為 numpy
    input_np = input_img.cpu().permute(1, 2, 0).numpy()
    output_np = output.cpu().detach().permute(1, 2, 0).numpy()
    target_np = target.cpu().permute(1, 2, 0).numpy()
    
    # 裁剪到 [0, 1]
    output_np = np.clip(output_np, 0, 1)
    
    axes[0].imshow(input_np)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(output_np)
    axes[1].set_title(f'Output')
    axes[1].axis('off')
    
    axes[2].imshow(target_np)
    axes[2].set_title('Target')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"已保存對比圖: {filename}")

# 執行診斷訓練
if __name__ == "__main__":
    train_debug()