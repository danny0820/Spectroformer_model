# 測試模型腳本
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from final_model_AGSSF1 import mymodel

# 解析命令列參數
parser = argparse.ArgumentParser(description='Spectroformer 測試腳本')
parser.add_argument('--dataset_path', default='/danny/Spectroformer_model/LSUI', help='LSUI 資料集的路徑')
parser.add_argument('--model_path', required=True, help='模型檢查點檔案路徑')
parser.add_argument('--batch_size', type=int, default=1, help='測試批次大小')
parser.add_argument('--output_dir', type=str, default='./test_results', help='儲存測試結果的資料夾')
parser.add_argument('--save_images', action='store_true', default=False, help='是否儲存測試結果圖片')
opt = parser.parse_args()

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

class LSUITestDataset(Dataset):
    """LSUI 測試資料集"""
    def __init__(self, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'test')
        self.degraded_path = os.path.join(self.data_path, 'input')
        self.gt_path = os.path.join(self.data_path, 'gt')
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

def save_img(img_tensor, filename):
    """將張量保存為圖片"""
    if len(img_tensor.shape) == 3:
        img = transforms.ToPILImage()(img_tensor.clamp(0, 1))
        img.save(filename)

def load_model(model_path, device):
    """載入訓練好的模型"""
    print(f"載入模型: {model_path}")
    
    # 初始化模型 - 使用 final_model_AGSSF1.py 中的 mymodel
    net_g = mymodel(
        num_blocks=[2, 3, 3, 4], 
        num_heads=[1, 2, 4, 8], 
        channels=[16, 32, 64, 128], 
        num_refinement=4,
        expansion_factor=2.66, 
        ch=[64, 32, 16, 64]
    )
    
    # 載入檢查點
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # 處理不同格式的檢查點
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # 如果是DataParallel模型，移除module.前綴
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 移除 'module.' 前綴
            else:
                new_state_dict[k] = v
                
        net_g.load_state_dict(new_state_dict)
        print("模型載入成功")
    else:
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")
    
    net_g = net_g.to(device)
    net_g.eval()
    return net_g

def test_model():
    """測試模型並計算指標"""
    # 載入模型
    net_g = load_model(opt.model_path, device)
    
    # 準備資料集
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    test_set = LSUITestDataset(opt.dataset_path, transform=test_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)
    
    print(f"測試資料數量: {len(test_set)}")
    
    # 創建輸出目錄
    if opt.save_images:
        os.makedirs(opt.output_dir, exist_ok=True)
    
    # 初始化指標
    total_psnr = 0
    total_ssim = 0
    total_images = 0
    
    print("開始測試...")
    print("-" * 60)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            rgb_input, target, img_names = batch[0].to(device), batch[1].to(device), batch[2]
            
            # 模型推論 - 使用 RGB_input 作為參數名稱以匹配模型的 forward 方法
            try:
                prediction = net_g(rgb_input)
            except Exception as e:
                print(f"模型推論錯誤: {e}")
                print(f"輸入形狀: {rgb_input.shape}")
                continue
            
            # 確保輸出形狀正確
            if prediction.shape != target.shape:
                print(f"警告: 預測輸出形狀 {prediction.shape} 與目標形狀 {target.shape} 不匹配")
                # 如果需要，可以在這裡調整輸出尺寸
                if prediction.shape[2:] != target.shape[2:]:
                    prediction = torch.nn.functional.interpolate(
                        prediction, size=target.shape[2:], mode='bilinear', align_corners=False
                    )
            
            # 計算每張圖片的PSNR和SSIM
            for j in range(rgb_input.size(0)):
                pred_np = prediction[j].cpu().permute(1, 2, 0).numpy()
                target_np = target[j].cpu().permute(1, 2, 0).numpy()
                
                # 確保數值範圍在[0, 1]
                pred_np = np.clip(pred_np, 0, 1)
                target_np = np.clip(target_np, 0, 1)
                
                # 計算PSNR和SSIM
                try:
                    img_psnr = psnr(target_np, pred_np, data_range=1.0)
                    img_ssim = ssim(target_np, pred_np, multichannel=True, data_range=1.0, win_size=3)
                    
                    total_psnr += img_psnr
                    total_ssim += img_ssim
                    total_images += 1
                    
                    print(f"圖片 {img_names[j]}: PSNR = {img_psnr:.2f} dB, SSIM = {img_ssim:.4f}")
                except Exception as e:
                    print(f"計算指標時發生錯誤 (圖片 {img_names[j]}): {e}")
                    continue
                
                # 儲存結果圖片（如果需要）
                if opt.save_images:
                    # 合併原圖、預測圖和目標圖
                    combined = torch.cat((rgb_input[j], prediction[j], target[j]), 2)  # 在寬度方向拼接
                    save_img(combined.cpu(), os.path.join(opt.output_dir, f"result_{img_names[j]}"))
            
            # 每10個batch顯示一次進度
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                current_avg_psnr = total_psnr / total_images
                current_avg_ssim = total_ssim / total_images
                print(f"進度: {i+1}/{len(test_loader)} batches, "
                      f"目前平均 PSNR: {current_avg_psnr:.2f} dB, "
                      f"目前平均 SSIM: {current_avg_ssim:.4f}")
                print("-" * 60)
    
    # 計算最終平均值
    if total_images > 0:
        avg_psnr = total_psnr / total_images
        avg_ssim = total_ssim / total_images
        
        print("\n" + "=" * 60)
        print("測試完成！最終結果:")
        print("=" * 60)
        print(f"總測試圖片數量: {total_images}")
        print(f"平均 PSNR: {avg_psnr:.2f} dB")
        print(f"平均 SSIM: {avg_ssim:.4f}")
        print("=" * 60)
        
        # 儲存結果到文件 - 使用追加模式以保留歷史記錄
        result_file = os.path.join(opt.output_dir if opt.save_images else '.', 'test_results.txt')
        
        # 添加時間戳記以區分不同次的測試
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(result_file, 'a', encoding='utf-8') as f:  # 使用 'a' 模式追加而不是覆寫
            f.write(f"\n{'='*60}\n")
            f.write(f"測試時間: {timestamp}\n")
            f.write(f"模型檔案: {opt.model_path}\n")
            f.write(f"測試資料集: {opt.dataset_path}\n")
            f.write(f"總測試圖片數量: {total_images}\n")
            f.write(f"平均 PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"平均 SSIM: {avg_ssim:.4f}\n")
            f.write(f"{'='*60}\n")
        
        print(f"詳細結果已儲存至: {result_file}")
        
        return avg_psnr, avg_ssim
    else:
        print("沒有找到測試圖片！")
        return None, None

if __name__ == '__main__':
    print("=" * 70)
    print("Spectroformer (mymodel) 模型測試腳本")
    print("=" * 70)
    print(f"模型檔案: {opt.model_path}")
    print(f"測試資料集: {opt.dataset_path}")
    print(f"批次大小: {opt.batch_size}")
    print(f"是否儲存圖片: {opt.save_images}")
    print(f"輸出目錄: {opt.output_dir}")
    print("=" * 70)
    print()
    
    # 檢查模型檔案是否存在
    if not os.path.exists(opt.model_path):
        print(f"❌ 錯誤: 找不到模型檔案 {opt.model_path}")
        print("請確認模型檔案路徑是否正確")
        exit(1)
    
    # 檢查資料集路徑是否存在
    if not os.path.exists(opt.dataset_path):
        print(f"❌ 錯誤: 找不到資料集路徑 {opt.dataset_path}")
        print("請確認資料集路徑是否正確")
        exit(1)
    
    try:
        avg_psnr, avg_ssim = test_model()
        if avg_psnr is not None:
            print(f"\n✅ 測試成功完成！")
            print(f"📊 最終結果: PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.4f}")
            print("\n使用範例命令:")
            print(f"python test_model.py --model_path {opt.model_path} --dataset_path {opt.dataset_path}")
        else:
            print("❌ 測試失敗: 沒有成功處理任何圖片")
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {str(e)}")
        print("\n請檢查以下項目:")
        print("1. 模型檔案是否正確")
        print("2. 資料集路徑是否正確")
        print("3. GPU 記憶體是否足夠")
        print("4. 相依套件是否已安裝")
        print("\n詳細錯誤信息:")
        import traceback
        traceback.print_exc()
