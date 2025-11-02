# 引入必要的函式庫
from __future__ import print_function
import argparse
import os
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
# from final_model_AGSSF1 import mymodel 
# from model_without_CA import mymodel
from phaseformer import Restormer
import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 解析命令列參數
parser = argparse.ArgumentParser(description='Spectroformer 訓練腳本')
# --- 資料集與批次大小 ---
parser.add_argument('--dataset_path', default='/danny/Spectroformer_model/LSUI', help='LSUI 資料集的路徑')
parser.add_argument('--batch_size', type=int, default=8, help='訓練批次大小')
parser.add_argument('--test_batch_size', type=int, default=8, help='測試批次大小')
parser.add_argument('--threads', type=int, default=0, help='資料載入器使用的執行緒數量')

# --- 訓練週期與學習率 ---
parser.add_argument('--niter', type=int, default=50, help='固定學習率的訓練週期數')
parser.add_argument('--niter_decay', type=int, default=500, help='學習率衰減的訓練週期數')
parser.add_argument('--lr', type=float, default=0.001, help='初始學習率')
parser.add_argument('--lr_policy', type=str, default='lambda', help='學習率策略')
parser.add_argument('--epoch_count', type=int, default=0, help='起始的 epoch 計數')

# --- 優化器與模型儲存 ---
parser.add_argument('--beta1', type=float, default=0.5, help='Adam 優化器的 beta1 參數')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='儲存模型權重的資料夾')
parser.add_argument('--output_dir', type=str, default='./images', help='儲存訓練/測試影像的資料夾')
parser.add_argument('--resume', type=str, default='', help='恢復訓練的檢查點路徑')

# --- 環境設定 ---
parser.add_argument('--cuda', action='store_true', default=True, help='是否使用 CUDA')
parser.add_argument('--seed', type=int, default=123, help='隨機種子')
opt = parser.parse_args()

print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
if opt.cuda and not torch.cuda.is_available():
    raise Exception("未找到 GPU，請在沒有 --cuda 的情況下執行")

cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

net_g = Restormer()
net_g = net_g.to(device)

# 多卡訓練設置
if torch.cuda.device_count() > 1:
    print(f"檢測到 {torch.cuda.device_count()} 張 GPU")
    net_g = torch.nn.DataParallel(net_g)
    print(f"使用 {torch.cuda.device_count()} 張 GPU 進行訓練")
else:
    print("只檢測到 1 張 GPU，使用單卡訓練")

net_g = net_g.to(device)

# 權重初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

net_g.apply(init_weights)
print('模型參數總量:', sum(p.numel() for p in net_g.parameters() if p.requires_grad))

# 恢復訓練檢查點
start_epoch = opt.epoch_count
if opt.resume:
    if os.path.isfile(opt.resume):
        print(f"載入檢查點 '{opt.resume}'")
        checkpoint = torch.load(opt.resume)
        if isinstance(net_g, torch.nn.DataParallel):
            net_g.module.load_state_dict(checkpoint)
        else:
            net_g.load_state_dict(checkpoint)
        # 從檔名中提取epoch
        epoch_num = int(opt.resume.split('_')[-1].split('.')[0])
        start_epoch = epoch_num
        print(f"從第 {start_epoch} 個epoch恢復訓練")
    else:
        print(f"找不到檢查點文件: {opt.resume}")

class LSUITrainingDataset(Dataset):
    """LSUI 訓練資料集"""
    def __init__(self, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'train')
        self.degraded_path = os.path.join(self.data_path, 'input')
        self.gt_path = os.path.join(self.data_path, 'gt')
        self.image_files = os.listdir(self.degraded_path)
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

class LSUITestDataset(Dataset):
    """LSUI 測試資料集"""
    def __init__(self, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'test')
        self.degraded_path = os.path.join(self.data_path, 'input')
        self.gt_path = os.path.join(self.data_path, 'gt')
        self.image_files = os.listdir(self.degraded_path)
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
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_set = LSUITrainingDataset(opt.dataset_path, transform=train_transform)
test_set = LSUITestDataset(opt.dataset_path, transform=test_transform)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False, pin_memory=True)

print(f"訓練資料數量: {len(train_set)}")
print(f"測試資料數量: {len(test_set)}")


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x, xx):
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g.to(y.device), groups=3)
        gradient_xx = F.conv2d(yy, self.weight_g.to(yy.device), groups=3)
        l = nn.L1Loss()
        return l(gradient_x, gradient_xx)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(weights='DEFAULT').features[:35].eval().to(device)
        self.vgg = nn.Sequential(*vgg)
        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))

# 初始化所有損失函數
Charbonnier_loss = CharbonnierLoss().to(device)
Gradient_Loss = GradientLoss().to(device)
L_per = VGGPerceptualLoss().to(device)
MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented' % opt.lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    scheduler.step()

optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)


def save_img(img_tensor, filename):
    """將張量保存為圖片"""
    if len(img_tensor.shape) == 3:
        img = transforms.ToPILImage()(img_tensor.clamp(0, 1))
        img.save(filename)

# 創建輸出目錄
os.makedirs(opt.output_dir, exist_ok=True)


for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    # 訓練
    epoch_loss = 0.0
    num_batches = 0
    
    # 創建進度條
    progress_bar = tqdm(training_data_loader, desc=f'Epoch {epoch}/{opt.niter + opt.niter_decay}', 
                       unit='batch', leave=True)
    
    for iteration, batch in enumerate(progress_bar, 1):
        # 前向傳播
        rgb, tar, indx = batch[0].to(device), batch[1].to(device), batch[2]
        output = net_g(rgb)
        
        # 處理模型輸出 - Restormer返回兩個輸出
        if isinstance(output, tuple):
            fake_b = output[0]  # 使用第一個輸出作為主要輸出
        else:
            fake_b = output

        ######################
        # 更新生成器
        ######################
        optimizer_g.zero_grad()
      
        # 計算損失（使用Spectroformer原始損失權重）
        loss_g_l1 = (0.03 * Charbonnier_loss(tar, fake_b) +
                      0.025 * L_per(fake_b, tar) +
                      0.02 * Gradient_Loss(fake_b, tar) +
                      0.01 * (1 - MS_SSIM_loss(fake_b, tar)))

        loss_g = loss_g_l1
        
        # 檢查損失是否為nan或inf
        if torch.isnan(loss_g) or torch.isinf(loss_g):
            print(f"警告：在epoch {epoch}, iteration {iteration} 檢測到異常損失: {loss_g.item()}")
            print(f"L1損失: {0.03 * Charbonnier_loss(tar, fake_b):.6f}")
            print(f"感知損失: {0.025 * L_per(fake_b, tar):.6f}")
            print(f"梯度損失: {0.02 * Gradient_Loss(fake_b, tar):.6f}")
            print(f"MS-SSIM損失: {0.01 * (1 - MS_SSIM_loss(fake_b, tar)):.6f}")
            
            # 跳過這個batch的更新
            optimizer_g.zero_grad()
            continue
            
        loss_g.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
        
        optimizer_g.step()
        
        # 累計損失和批次數
        epoch_loss += loss_g.item()
        num_batches += 1
        
        # 更新進度條
        current_loss = epoch_loss / num_batches
        progress_bar.set_postfix({
            'Loss': f'{loss_g.item():.4f}',
            'Avg_Loss': f'{current_loss:.4f}',
            'LR': f'{optimizer_g.param_groups[0]["lr"]:.6f}'
        })

        if iteration % 20 == 0:
            out_image = torch.cat((rgb, fake_b, tar), 3)
            out_image = out_image[0].detach().squeeze(0).cpu()
            save_img(out_image, os.path.join(opt.output_dir, indx[0]))
        
        # # 清理不必要的變數以節省記憶體
        # del fake_b, loss_g
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
    
    # 關閉進度條並顯示epoch統計
    progress_bar.close()
    if num_batches > 0:
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch} 完成 - 平均損失: {avg_epoch_loss:.4f}")
        print(f"學習率: {optimizer_g.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    update_learning_rate(net_g_scheduler, optimizer_g)
  
    # 測試
    test_output_dir = os.path.join('./test/')
    os.makedirs(test_output_dir, exist_ok=True)

    # 初始化總體指標
    total_psnr = 0
    total_ssim = 0
    total_images = 0

    # 創建測試進度條
    test_progress = tqdm(testing_data_loader, desc=f'Testing Epoch {epoch}', 
                        unit='batch', leave=False)
    
    for test_iter, batch in enumerate(test_progress, 1):
        rgb_input, target, ind = batch[0].to(device), batch[1].to(device), batch[2]
        with torch.no_grad():
            output = net_g(rgb_input)
            # 處理模型輸出 - Restormer返回兩個輸出
            if isinstance(output, tuple):
                prediction = output[0]  # 使用第一個輸出作為主要輸出
            else:
                prediction = output
        
        out = torch.cat((prediction, target), 3)
        output_cat = out[0].detach().squeeze(0).cpu()
        save_img(output_cat, os.path.join(test_output_dir, ind[0]))

        # 計算整個 batch 的 PSNR 和 SSIM
        batch_psnr = 0
        batch_ssim = 0
        for i in range(rgb_input.size(0)):
            prediction_np = prediction[i].cpu().permute(1, 2, 0).numpy()
            target_np = target[i].cpu().permute(1, 2, 0).numpy()

            # 累加每張圖片的 PSNR 和 SSIM
            img_psnr = psnr(target_np, prediction_np, data_range=1.0)
            img_ssim = ssim(target_np, prediction_np, multichannel=True, data_range=1.0, win_size=3)
            
            batch_psnr += img_psnr
            batch_ssim += img_ssim
            
            # 累加到總體指標
            total_psnr += img_psnr
            total_ssim += img_ssim
            total_images += 1

        # 計算 batch 平均值
        batch_psnr /= rgb_input.size(0)
        batch_ssim /= rgb_input.size(0)
        
        # 更新測試進度條
        test_progress.set_postfix({
            'PSNR': f'{batch_psnr:.2f}',
            'SSIM': f'{batch_ssim:.4f}'
        })

    # 計算並顯示全部測試影像的平均值
    if total_images > 0:
        avg_psnr = total_psnr / total_images
        avg_ssim = total_ssim / total_images
        # 關閉測試進度條
        test_progress.close()
        
        print(f"\n=== Epoch {epoch} 測試結果 ===")
        print(f"全部測試影像平均 PSNR: {avg_psnr:.2f} dB")
        print(f"全部測試影像平均 SSIM: {avg_ssim:.4f}")
        print(f"總測試影像數量: {total_images}")
        print("=" * 50)

    # 儲存檢查點
    if epoch % 1 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset_path.split('/')[-1])):
            os.mkdir(os.path.join("checkpoint", opt.dataset_path.split('/')[-1]))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(
            opt.dataset_path.split('/')[-1], epoch + 1)
        # 如果使用 DataParallel，保存時需要使用 .module
        if isinstance(net_g, torch.nn.DataParallel):
            torch.save(net_g.module.state_dict(), net_g_model_out_path)
        else:
            torch.save(net_g.state_dict(), net_g_model_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))

print("訓練完成！")