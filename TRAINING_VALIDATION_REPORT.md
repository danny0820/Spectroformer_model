# Spectroformer 模型訓練驗證報告

## 驗證時間
2025-12-12

## 驗證結果：✅ 通過

## 驗證內容

### 1. 模型結構驗證
- ✅ 模型成功導入並初始化
- ✅ 模型參數量：2.99M
- ✅ 前向傳播正常
- ✅ 反向傳播和梯度計算正常
- ✅ 輸入輸出形狀匹配 (B, 3, H, W)

### 2. 代碼修復
已修復以下問題：
1. ✅ 添加 `mymodel()` 函數供訓練腳本調用
2. ✅ 修復 `DFU` 模組中 `torch.tile` 的維度參數
3. ✅ 重構 `trains.py` 為 `train_spectroformer.py`，解決模組導入時執行訓練的問題

### 3. 環境檢查
- ✅ CUDA 可用：8 張 GPU (TITAN RTX x4 + GTX 1080 Ti x4)
- ✅ 所有依賴套件已安裝
  - torch, torchvision
  - PIL, tqdm
  - pytorch_msssim
  - skimage
- ✅ 數據集完整
  - 訓練集：3879 張圖片
  - 測試集：400 張圖片

### 4. 訓練流程測試
- ✅ 訓練腳本成功啟動
- ✅ 數據載入正常
- ✅ 前向傳播速度：~2.76 batch/s
- ✅ 損失計算正常（初始損失 ~0.15-0.21）
- ✅ 梯度更新正常
- ✅ 學習率調度器正常
- ✅ 模型保存機制正常

## 模型配置

### Spectroformer Small (預設)
```python
dim=16
num_blocks=[2, 3, 3, 4]
num_refinement_blocks=4
num_heads=[1, 2, 4, 8]
ffn_expansion_factor=2.66
參數量: 2.99M
```

### 訓練超參數
```python
batch_size=4
learning_rate=0.001
optimizer=Adam(beta1=0.5, beta2=0.999)
scheduler=lambda (niter=50, niter_decay=500)
```

### 損失函數
使用加權多損失組合：
1. Charbonnier Loss (重建損失)
2. VGG Perceptual Loss (感知損失)
3. Gradient Loss (梯度損失)
4. MS-SSIM Loss (結構相似性損失)

## 文件結構

### 主要文件
- `spectroformer_paper.py` - 模型定義（論文復刻版）
- `train_spectroformer.py` - 訓練腳本（重構版，推薦使用）
- `trains.py` - 原始訓練腳本（已棄用）

### 測試文件
- `test_model_quick.py` - 快速模型測試
- `check_env.py` - 環境檢查
- `test_train.py` - 訓練流程測試

## 如何開始訓練

### 快速開始（使用預設參數）
```bash
python train_spectroformer.py
```

### 自定義參數
```bash
python train_spectroformer.py \
    --dataset_path /path/to/LSUI \
    --batch_size 8 \
    --lr 0.001 \
    --niter 50 \
    --niter_decay 500
```

### 恢復訓練
```bash
python train_spectroformer.py \
    --resume checkpoint/LSUI/run_XXXXXX/best_model.pth
```

### 背景執行
```bash
nohup python train_spectroformer.py > train.log 2>&1 &
```

## 預期訓練時間

基於測試結果：
- 速度：~2.76 batch/s (batch_size=2, 單GPU)
- 每個 epoch：~12 分鐘 (3879 張圖片)
- 550 個 epoch：~110 小時 (~4.5 天)

使用多 GPU 可以顯著加速訓練。

## 輸出文件

### 訓練過程中會生成：
1. `./images/train_YYYYMMDD_HHMMSS/` - 訓練可視化圖片
2. `./test/test_YYYYMMDD_HHMMSS/` - 測試結果圖片
3. `checkpoint/LSUI/run_YYYYMMDD_HHMMSS/best_model.pth` - 最佳模型

### 訓練指標：
- 每個 epoch 記錄平均損失
- 每個 epoch 在測試集上評估 PSNR 和 SSIM
- 自動保存 PSNR 最高的模型

## 監控訓練

### 查看訓練進度
```bash
tail -f train.log  # 如果使用 nohup
```

### 監控 GPU 使用
```bash
watch -n 1 nvidia-smi
```

### TensorBoard（可選）
如需添加 TensorBoard 支持，可以在訓練腳本中添加相應代碼。

## 已知問題與限制

1. ⚠️ 目前硬編碼使用 GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
   - 如需使用其他 GPU，修改 `train_spectroformer.py` 第 235 行
   
2. ⚠️ 圖片尺寸固定為 256x256
   - 如需更改，修改 transforms 中的 Resize 參數

3. ⚠️ 多 GPU 訓練可能出現不平衡警告
   - 混合使用 TITAN RTX 和 GTX 1080 Ti
   - 建議使用 `CUDA_VISIBLE_DEVICES` 選擇同類型 GPU

## 建議

1. ✅ 建議使用 `train_spectroformer.py` 進行訓練
2. ✅ 訓練前執行 `python check_env.py` 確認環境
3. ✅ 使用 `nohup` 或 `tmux` 進行長時間訓練
4. ✅ 定期備份最佳模型
5. ✅ 監控訓練損失，如出現 NaN 或 Inf 會自動跳過該批次

## 總結

Spectroformer 模型已通過完整驗證，可以順利進行訓練。所有核心功能正常運作：
- ✅ 模型結構正確
- ✅ 數據加載正常
- ✅ 訓練流程穩定
- ✅ 損失計算正確
- ✅ 模型保存機制完善

可以開始完整訓練！
