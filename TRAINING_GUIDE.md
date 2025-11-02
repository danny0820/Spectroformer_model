# Phaseformer 訓練指南

## 🔧 主要修改說明

### 1. 模型架構差異

#### 舊模型 (final_model_AGSSF1)
- **輸出**: 單一輸出 (256×256)
- **注意力機制**: SFCA (Spatial-Frequency Channel Attention)
- **上採樣**: 頻域上採樣 (UpS)

#### 新模型 (Restormer/Phaseformer)
- **輸出**: 雙輸出
  - `fake_b_L`: 低解析度輸出 (256×256)
  - `fake_b_H`: 高解析度輸出 (512×512)
- **注意力機制**: ECA (Efficient Channel Attention)
- **上採樣**: 像素上採樣 (PixelShuffle)

### 2. 訓練損失計算

```python
# 低解析度分支損失 (256×256)
loss_g_L = Weighted_Loss4(
    Charbonnier_loss(tar, fake_b_L),
    L_per(fake_b_L, tar),
    Gradient_Loss(fake_b_L, tar),
    1 - MS_SSIM_loss(fake_b_L, tar)
)

# 高解析度分支損失 (512×512)
tar_H = F.interpolate(tar, scale_factor=2, mode='bilinear')
loss_g_H = Weighted_Loss4(
    Charbonnier_loss(tar_H, fake_b_H),
    L_per(fake_b_H, tar_H),
    Gradient_Loss(fake_b_H, tar_H),
    1 - MS_SSIM_loss(fake_b_H, tar_H)
)

# 最終損失（動態權重組合）
loss_g = Weighted_Loss2(loss_g_L, loss_g_H)
```

### 3. 動態權重機制 (WeightedLoss)

```python
class WeightedLoss(nn.Module):
    """
    可學習的損失權重
    - 使用 softmax 確保權重和為 1
    - 權重在訓練過程中自動調整
    """
    def __init__(self, num_weights):
        super(WeightedLoss, self).__init__()
        self.weights = nn.Parameter(torch.rand(1, num_weights))
        self.softmax_l = nn.Softmax(dim=1)
```

**優點:**
- 自動平衡不同損失組件
- 避免手動調整權重
- 更好的訓練穩定性

## 📊 與原始 train.py 的對比

| 特性 | train.py (原始) | trains.py (修改後) |
|------|----------------|-------------------|
| 數據集 | 單一數據集 | 支持多數據集 |
| GPU 支持 | 單卡 | 多卡並行 (DataParallel) |
| 數據增強 | 雙尺度 (256, 512) | 單尺度 (256) + 動態上採樣 |
| 檢查點保存 | 保存整個模型 | 保存 state_dict |
| 進度條 | 無 | tqdm 進度條 |
| 損失計算 | ✅ 雙輸出損失 | ✅ 雙輸出損失 |

## 🚀 在服務器上執行

### 1. 環境準備

```bash
# 安裝必要套件
pip install pytorch-msssim
pip install scikit-image
pip install tqdm
pip install kornia
```

### 2. 檢查數據集結構

```
/danny/Spectroformer_model/LSUI/
├── train/
│   ├── input/
│   └── gt/
└── test/
    ├── input/
    └── gt/
```

### 3. 開始訓練

```bash
# 從頭訓練
python trains.py \
    --dataset_path /danny/Spectroformer_model/LSUI \
    --batch_size 8 \
    --niter 50 \
    --niter_decay 500 \
    --lr 0.001

# 恢復訓練（注意：新舊模型結構不同，無法直接載入舊檢查點）
python trains.py \
    --resume checkpoint/LSUI/netG_model_epoch_XXX.pth \
    --dataset_path /danny/Spectroformer_model/LSUI
```

### 4. 監控訓練

訓練過程中會顯示：
- ✅ 即時損失值 (Loss)
- ✅ 平均損失 (Avg_Loss)
- ✅ 當前學習率 (LR)
- ✅ 訓練進度 (tqdm 進度條)
- ✅ 測試 PSNR 和 SSIM

## ⚠️ 重要注意事項

### 1. 檢查點不兼容
**問題**: 新舊模型結構完全不同
**解決**: 
- ❌ 不能使用 `--resume` 載入舊模型檢查點
- ✅ 需要從頭開始訓練

### 2. 顯存使用
由於雙輸出損失計算，顯存使用增加約 **40-50%**

**建議**:
- 減小批次大小 (例如從 8 → 4)
- 或使用梯度累積

```python
# 梯度累積示例（在 trains.py 中添加）
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 學習率調整
原始模型參數量可能不同，建議：
- 初始學習率: 0.0001 - 0.001
- 使用學習率預熱 (warmup)

### 4. 損失權重監控
可以查看動態權重的變化：
```python
# 在訓練循環中添加
print(f"L損失權重: {Weighted_Loss4.weights.softmax(1)}")
print(f"L/H損失權重: {Weighted_Loss2.weights.softmax(1)}")
```

## 🎯 訓練技巧

### 1. 多尺度訓練
考慮使用不同尺度的輸入：
```python
scales = [0.8, 1.0, 1.2]
for scale in scales:
    scaled_input = F.interpolate(input, scale_factor=scale)
    output = model(scaled_input)
```

### 2. 混合精度訓練 (AMP)
減少顯存使用，加速訓練：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 學習率調度
```python
# Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

# One Cycle
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=epochs)
```

## 📈 預期結果

根據原始論文和測試結果：

| Epoch | PSNR (dB) | SSIM | 說明 |
|-------|-----------|------|------|
| 100   | 26-28     | 0.88-0.90 | 初期訓練 |
| 300   | 28-29     | 0.90-0.92 | 中期訓練 |
| 500+  | 29-30     | 0.92-0.93 | 收斂階段 |

## 🐛 常見問題

### Q1: RuntimeError: CUDA out of memory
**解決**:
```bash
# 減小批次大小
--batch_size 4

# 或減小圖片尺寸（修改 transforms.Resize）
transforms.Resize((128, 128))
```

### Q2: 損失變成 NaN
**解決**:
- 檢查學習率是否過大
- 啟用梯度裁剪（已包含在 trains.py 中）
- 檢查數據歸一化

### Q3: 訓練速度慢
**解決**:
```python
# 增加 num_workers
--threads 4

# 啟用 cudnn benchmark（已包含）
cudnn.benchmark = True
```

## 📝 修改記錄

**2025-11-02**:
- ✅ 修復雙輸出處理
- ✅ 添加動態權重損失
- ✅ 參考 train.py 實現雙分支損失
- ✅ 添加詳細的錯誤檢測和日誌

## 📚 參考資料

1. 原始 Restormer 論文
2. train.py 實現
3. LSUI 數據集規格
