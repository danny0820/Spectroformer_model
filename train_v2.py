# 引入必要的函式庫
from __future__ import print_function
import argparse
import os
import math
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from model_v2 import Spectroformer 
import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import gc

# 解析命令列參數
parser = argparse.ArgumentParser(description='Spectroformer 訓練腳本')
# --- 資料集與批次大小 ---
parser.add_argument('--dataset_path', default='/danny/Spectroformer_model/LSUI_test', help='LSUI 資料集的路徑')
parser.add_argument('--batch_size', type=int, default=2, help='訓練批次大小')
parser.add_argument('--test_batch_size', type=int, default=1, help='測試批次大小')
parser.add_argument('--threads', type=int, default=4, help='資料載入器使用的執行緒數量')

# --- 訓練週期與學習率 ---
parser.add_argument('--niter', type=int, default=200, help='訓練週期數')
parser.add_argument('--lr', type=float, default=2e-4, help='初始學習率')
parser.add_argument('--lr_policy', type=str, default='cosine', help='學習率策略')
parser.add_argument('--epoch_count', type=int, default=1, help='起始的 epoch 計數')

# --- 優化器與模型儲存 ---
parser.add_argument('--beta1', type=float, default=0.9, help='Adam 優化器的 beta1 參數')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='儲存模型權重的資料夾')
parser.add_argument('--output_dir', type=str, default='./images', help='儲存訓練/測試影像的資料夾')
parser.add_argument('--resume', type=str, default='', help='恢復訓練的檢查點路徑')
parser.add_argument('--save_freq', type=int, default=5, help='每幾個 epoch 儲存一次模型')

# --- 環境設定 ---
parser.add_argument('--cuda', action='store_true', default=True, help='是否使用 CUDA')
parser.add_argument('--seed', type=int, default=123, help='隨機種子')
parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累積步數')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("未找到 GPU，請在沒有 --cuda 的情況下執行")

# 設置隨機種子
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

cudnn.benchmark = True
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# GPU 記憶體優化設定
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    print(f"GPU 記憶體優化已啟用")

# 清理函數
def clear_cuda_cache():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

# 實例化 Spectroformer 模型 - 使用較小的配置
net_g = Spectroformer(
    dim=32,  # 減小維度以節省記憶體
    num_blocks=[2, 3, 3, 4],  # 減少區塊數量
    num_refinement_blocks=2,  # 減少 refinement blocks
    num_heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False
)

# 多卡訓練設置
if torch.cuda.device_count() > 1:
    print(f"檢測到 {torch.cuda.device_count()} 張 GPU")
    net_g = torch.nn.DataParallel(net_g)
    print(f"使用 {torch.cuda.device_count()} 張 GPU 進行訓練")
else:
    print("只檢測到 1 張 GPU，使用單卡訓練")

net_g = net_g.to(device)

# 改進的權重初始化
def init_weights_improved(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

net_g.apply(init_weights_improved)
print('模型參數總量:', sum(p.numel() for p in net_g.parameters() if p.requires_grad))

# 計算有效批次大小和調整學習率
effective_batch_size = opt.batch_size * opt.accumulation_steps
adjusted_lr = opt.lr * (effective_batch_size / 8)  # 基準批次大小為 8
print(f"有效批次大小: {effective_batch_size}, 調整後學習率: {adjusted_lr:.6f}")

# 優化器設置 - 使用 AdamW
optimizer_g = optim.AdamW(
    net_g.parameters(), 
    lr=adjusted_lr,
    betas=(opt.beta1, 0.999),
    weight_decay=0.02,
    eps=1e-8
)

# 恢復訓練檢查點
start_epoch = opt.epoch_count
if opt.resume:
    if os.path.isfile(opt.resume):
        print(f"載入檢查點 '{opt.resume}'")
        checkpoint = torch.load(opt.resume, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            if isinstance(net_g, torch.nn.DataParallel):
                net_g.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                net_g.load_state_dict(checkpoint['model_state_dict'])
        else:
            if isinstance(net_g, torch.nn.DataParallel):
                net_g.module.load_state_dict(checkpoint)
            else:
                net_g.load_state_dict(checkpoint)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        
        print(f"從第 {start_epoch} 個 epoch 恢復訓練")

class LSUITrainingDataset(Dataset):
    """LSUI 訓練資料集"""
    def __init__(self, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'train')
        self.degraded_path = os.path.join(self.data_path, 'input')
        self.gt_path = os.path.join(self.data_path, 'gt')
        
        if not os.path.exists(self.degraded_path):
            raise ValueError(f"找不到路徑: {self.degraded_path}")
        if not os.path.exists(self.gt_path):
            raise ValueError(f"找不到路徑: {self.gt_path}")
            
        self.image_files = sorted(os.listdir(self.degraded_path))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        degraded_img = Image.open(os.path.join(self.degraded_path, img_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_path, img_name)).convert('RGB')
        
        if self.transform:
            # 對兩張圖片應用相同的隨機變換
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            degraded_img = self.transform(degraded_img)
            torch.manual_seed(seed)
            gt_img = self.transform(gt_img)
            
        return degraded_img, gt_img, img_name

class LSUITestDataset(Dataset):
    """LSUI 測試資料集"""
    def __init__(self, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'test')
        self.degraded_path = os.path.join(self.data_path, 'input')
        self.gt_path = os.path.join(self.data_path, 'gt')
        
        if not os.path.exists(self.degraded_path):
            raise ValueError(f"找不到路徑: {self.degraded_path}")
        if not os.path.exists(self.gt_path):
            raise ValueError(f"找不到路徑: {self.gt_path}")
            
        self.image_files = sorted(os.listdir(self.degraded_path))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        degraded_img = Image.open(os.path.join(self.degraded_path, img_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_path, img_name)).convert('RGB')

        if self.transform:
            degraded_img = self.transform(degraded_img)
            gt_img = self.transform(gt_img)
            
        return degraded_img, gt_img, img_name

print('===> 載入資料集')

# 簡化的資料增強 - 不使用 Normalize
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),  # 不使用 Normalize
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_set = LSUITrainingDataset(opt.dataset_path, transform=train_transform)
test_set = LSUITestDataset(opt.dataset_path, transform=test_transform)

training_data_loader = DataLoader(
    dataset=train_set, 
    num_workers=opt.threads, 
    batch_size=opt.batch_size, 
    shuffle=True, 
    pin_memory=True,
    drop_last=True
)

testing_data_loader = DataLoader(
    dataset=test_set, 
    num_workers=opt.threads, 
    batch_size=opt.test_batch_size, 
    shuffle=False, 
    pin_memory=True
)

print(f"訓練資料數量: {len(train_set)}")
print(f"測試資料數量: {len(test_set)}")

# 損失函數定義
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        self.sobel_x = nn.Parameter(self.sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(self.sobel_y, requires_grad=False)

    def forward(self, x, y):
        grad_x_x = F.conv2d(x, self.sobel_x.to(x.device), padding=1, groups=3)
        grad_x_y = F.conv2d(x, self.sobel_y.to(x.device), padding=1, groups=3)
        
        grad_y_x = F.conv2d(y, self.sobel_x.to(y.device), padding=1, groups=3)
        grad_y_y = F.conv2d(y, self.sobel_y.to(y.device), padding=1, groups=3)
        
        return F.l1_loss(grad_x_x, grad_y_x) + F.l1_loss(grad_x_y, grad_y_y)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(weights='DEFAULT').features[:16].eval()  # 使用較淺的層
        self.vgg = nn.Sequential(*vgg)
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))

# 初始化所有損失函數
Charbonnier_loss = CharbonnierLoss().to(device)
Gradient_Loss = GradientLoss().to(device)
L_per = VGGPerceptualLoss().to(device)
MS_SSIM_loss = MS_SSIM(
    win_size=7,
    win_sigma=1.5, 
    data_range=1, 
    size_average=True, 
    channel=3
).to(device)

# 學習率調度器 - 使用 warmup
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 計算總步數和 warmup 步數
total_steps = len(training_data_loader) * opt.niter // opt.accumulation_steps
warmup_steps = min(500, total_steps // 20)  # 5% warmup 或 500 步
scheduler = get_cosine_schedule_with_warmup(optimizer_g, warmup_steps, total_steps)

def save_img(img_tensor, filename):
    """將張量保存為圖片"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if len(img_tensor.shape) == 3:
        img = transforms.ToPILImage()(img_tensor.clamp(0, 1))
        img.save(filename)

# 創建輸出目錄
os.makedirs(opt.output_dir, exist_ok=True)
os.makedirs(opt.checkpoint_dir, exist_ok=True)
os.makedirs('./test/', exist_ok=True)

# 混合精度訓練
scaler = GradScaler()

# 訓練記錄
best_psnr = 0
best_ssim = 0
global_step = 0

print("開始訓練...")
print(f"梯度累積步數: {opt.accumulation_steps}")
print(f"有效批次大小: {effective_batch_size}")

for epoch in range(start_epoch, opt.niter + 1):
    net_g.train()
    epoch_loss = 0
    accumulation_loss = 0
    
    pbar = tqdm(training_data_loader, desc=f'Epoch {epoch}/{opt.niter}')
    
    optimizer_g.zero_grad()
    
    for iteration, batch in enumerate(pbar, 1):
        rgb, tar, indx = batch[0].to(device), batch[1].to(device), batch[2]
        
        # 混合精度訓練
        with autocast():
            fake_b = net_g(rgb)
            
            # 計算損失 - 調整權重
            loss_char = Charbonnier_loss(tar, fake_b)
            loss_perc = L_per(fake_b, tar)
            loss_grad = Gradient_Loss(fake_b, tar)
            loss_msssim = 1 - MS_SSIM_loss(fake_b, tar)
            
            # 使用更保守的權重
            loss_g = (1.0 * loss_char +  # 主要損失
                      0.1 * loss_perc +   # 感知損失
                      0.05 * loss_grad +  # 梯度損失
                      0.1 * loss_msssim)  # MS-SSIM 損失
            
            # 梯度累積
            loss_g = loss_g / opt.accumulation_steps
        
        # 反向傳播
        scaler.scale(loss_g).backward()
        accumulation_loss += loss_g.item()
        
        # 梯度累積更新
        if (iteration) % opt.accumulation_steps == 0 or iteration == len(training_data_loader):
            # 梯度裁剪
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=0.5)
            
            # 優化器步驟
            scaler.step(optimizer_g)
            scaler.update()
            optimizer_g.zero_grad()
            
            # 更新學習率
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += accumulation_loss * opt.accumulation_steps
            accumulation_loss = 0
            global_step += 1
            
            # 更新進度條
            current_lr = optimizer_g.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss_g.item() * opt.accumulation_steps:.4f}',
                'LR': f'{current_lr:.6f}',
                'L_char': f'{loss_char.item():.4f}',
                'L_perc': f'{loss_perc.item():.4f}'
            })
        
        # 定期保存訓練樣本
        if global_step % 50 == 0 and iteration % opt.accumulation_steps == 0:
            with torch.no_grad():
                out_image = torch.cat((rgb[0:1], fake_b[0:1], tar[0:1]), 3)
                save_img(out_image[0].cpu(), os.path.join(opt.output_dir, f'train_epoch_{epoch}_step_{global_step}.jpg'))
        
        # 定期清理記憶體
        if iteration % 100 == 0:
            clear_cuda_cache()
    
    # 計算平均損失
    avg_loss = epoch_loss / len(training_data_loader)
    print(f"\nEpoch [{epoch}] 平均訓練損失: {avg_loss:.4f}")
    
    # 測試
    if epoch % 2 == 0:  # 每 2 個 epoch 測試一次
        net_g.eval()
        total_psnr = 0
        total_ssim = 0
        total_images = 0
        
        print("開始測試...")
        with torch.no_grad():
            for test_iter, batch in enumerate(testing_data_loader, 1):
                rgb_input, target, ind = batch[0].to(device), batch[1].to(device), batch[2]
                
                with autocast():
                    prediction = net_g(rgb_input)
                
                # 保存測試結果
                if test_iter <= 3:
                    out = torch.cat((rgb_input, prediction, target), 3)
                    save_img(out[0].cpu(), os.path.join('./test/', f'test_epoch_{epoch}_{ind[0]}'))
                
                # 計算指標
                for i in range(rgb_input.size(0)):
                    pred_np = prediction[i].cpu().permute(1, 2, 0).numpy()
                    targ_np = target[i].cpu().permute(1, 2, 0).numpy()
                    
                    pred_np = np.clip(pred_np, 0, 1)
                    targ_np = np.clip(targ_np, 0, 1)
                    
                    img_psnr = psnr(targ_np, pred_np, data_range=1.0)
                    img_ssim = ssim(targ_np, pred_np, 
                                   channel_axis=2,
                                   data_range=1.0, 
                                   win_size=3)
                    
                    total_psnr += img_psnr
                    total_ssim += img_ssim
                    total_images += 1
        
        if total_images > 0:
            avg_psnr = total_psnr / total_images
            avg_ssim = total_ssim / total_images
            
            print(f"=== Epoch {epoch} 測試結果 ===")
            print(f"平均 PSNR: {avg_psnr:.2f} dB")
            print(f"平均 SSIM: {avg_ssim:.4f}")
            print("=" * 40)
            
            # 保存最佳模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_ssim = avg_ssim
                
                best_path = os.path.join(opt.checkpoint_dir, 'best_model.pth')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net_g.module.state_dict() if isinstance(net_g, torch.nn.DataParallel) else net_g.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                    'best_psnr': best_psnr,
                    'best_ssim': best_ssim,
                    'avg_loss': avg_loss
                }
                torch.save(checkpoint, best_path)
                print(f"最佳模型已保存 (PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f})")
    
    # 定期保存檢查點
    if epoch % opt.save_freq == 0:
        checkpoint_path = os.path.join(opt.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net_g.module.state_dict() if isinstance(net_g, torch.nn.DataParallel) else net_g.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"檢查點已保存到 {checkpoint_path}")

print("\n訓練完成！")
print(f"最佳 PSNR: {best_psnr:.2f} dB")
print(f"最佳 SSIM: {best_ssim:.4f}")