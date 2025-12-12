# Spectroformer 實現對比分析

**文檔版本**: 1.0  
**創建日期**: 2025-12-12  
**對比模型**: 
- `spectroformer_paper.py` - 論文復刻版
- `final_model_AGSSF1.py` - 自定義改進版

---

## 📋 目錄

1. [模型概述](#模型概述)
2. [核心架構對比](#核心架構對比)
3. [模組詳細對比](#模組詳細對比)
4. [技術實現差異](#技術實現差異)
5. [性能與參數對比](#性能與參數對比)
6. [優缺點分析](#優缺點分析)
7. [使用建議](#使用建議)

---

## 1. 模型概述

### spectroformer_paper.py (論文版)
```
目的: 嚴格按照 Spectroformer 論文復刻
特點: 學術標準實現，包含所有論文描述的模組
適用: 論文復現、學術研究、基準測試
```

### final_model_AGSSF1.py (改進版)
```
目的: 在 Restormer 基礎上加入自定義改進
特點: 融合多種注意力機制和頻域處理技術
適用: 實際應用、性能優化、實驗探索
```

---

## 2. 核心架構對比

### 2.1 整體架構

#### spectroformer_paper.py
```
編碼器-解碼器 U-Net 結構
│
├── Encoder (4 levels)
│   ├── Level 1: 16 channels, MQCT blocks [2]
│   ├── Level 2: 32 channels, MQCT blocks [3]
│   ├── Level 3: 64 channels, MQCT blocks [3]
│   └── Level 4: 128 channels, MQCT blocks [4] (Bottleneck)
│
├── Decoder (3 levels)
│   ├── Level 3: HFSUB ↑ + SSFB + MQCT blocks [3]
│   ├── Level 2: HFSUB ↑ + SSFB + MQCT blocks [3]
│   └── Level 1: HFSUB ↑ + SSFB + MQCT blocks [2]
│
└── Refinement: MQCT blocks [4]
```

#### final_model_AGSSF1.py
```
Restormer-based 編碼器-解碼器結構
│
├── Encoder (4 levels)
│   ├── Level 1: 16 channels, Transformer blocks [2]
│   ├── Level 2: 32 channels, Transformer blocks [3]
│   ├── Level 3: 64 channels, Transformer blocks [3]
│   └── Level 4: 128 channels, Transformer blocks [4]
│
├── Decoder (3 levels)
│   ├── Level 3: UpS ↑ + SFCA + Transformer blocks [3]
│   ├── Level 2: UpS ↑ + SFCA + Transformer blocks [3]
│   └── Level 1: UpS ↑ + SFCA + Transformer blocks [2]
│
└── Refinement: Transformer blocks [4]
```

### 2.2 關鍵架構差異

| 特性 | spectroformer_paper.py | final_model_AGSSF1.py |
|------|------------------------|------------------------|
| **基礎架構** | 論文原創設計 | 基於 Restormer |
| **上採樣** | HFSUB (混合頻域-空間) | UpS (頻域+空間拼接) |
| **跳接融合** | SSFB (空間-頻譜融合) | SFCA (空間-頻率通道注意力) |
| **Transformer** | MQCT (多域查詢級聯) | TransformerBlock (標準) |
| **注意力機制** | MQCA (兩階段級聯) | MDTA (多頭雙路徑) |
| **通道注意力** | AKC (自適應核卷積) | AGSSF (自適應門控) |

---

## 3. 模組詳細對比

### 3.1 注意力機制

#### MQCA (spectroformer_paper.py) - 多域查詢級聯注意力
```python
特點:
- 兩階段級聯設計
- Stage 1: 空間域注意力 (Q, K, V from DWConv)
- Stage 2: 頻域-空間混合注意力 (Q from FDFP, K/V from Stage 1 output)
- Q/K 進行 L2 正規化
- 可學習溫度參數

優勢:
✓ 結合空間和頻域信息
✓ 級聯設計增強特徵提取
✓ 論文驗證有效

實現複雜度: ★★★★☆
```

#### MDTA (final_model_AGSSF1.py) - 多頭雙路徑注意力
```python
特點:
- 雙路徑設計
- Path 1: 標準空間域注意力 (Q, K, V from DWConv)
- Path 2: 頻域處理注意力 (Q from FFT+GELU, K/V from Path 1)
- 單一溫度參數

優勢:
✓ 雙路徑並行處理
✓ 實現相對簡單
✓ 計算效率高

實現複雜度: ★★★☆☆
```

**主要差異**:
- MQCA 是**級聯** (sequential)，MDTA 是**並行** (parallel)
- MQCA 使用 FDFP 頻域處理器，MDTA 直接使用 FFT
- MQCA 輸出來自 Stage 2，MDTA 輸出來自 Path 2

### 3.2 頻域特徵處理

#### FDFP (spectroformer_paper.py) - 頻域特徵處理器
```python
class FDFP(nn.Module):
    """專用頻域處理模組"""
    流程:
    1. FFT 轉頻域
    2. 分離實部和虛部 → 通道拼接 (C → 2C)
    3. Conv1x1 → GELU → Conv1x1 處理
    4. 重組複數 → IFFT 回空間域
    
特點:
✓ 專門設計的頻域處理模組
✓ 保留相位信息
✓ 可學習的頻域特徵變換
```

#### FFT 處理 (final_model_AGSSF1.py)
```python
# 簡單的 FFT 處理
x_fft = fft.fftn(x, dim=(-2, -1)).real  # 只取實部
x_fft = F.gelu(x_fft)
x_fft = self.q1X1_1(x_fft)
qf = fft.ifftn(x_fft, dim=(-2, -1)).real
    
特點:
✓ 實現簡單直接
✓ 僅使用實部，丟棄虛部
✗ 相位信息丟失
```

**主要差異**:
- FDFP **保留完整複數信息**（實部+虛部）
- final_model 只使用 **實部**，可能損失相位信息

### 3.3 上採樣模組

#### HFSUB (spectroformer_paper.py) - 混合傅立葉-空間上採樣
```python
class HFSUB:
    組成:
    1. DFU (Deep Fourier Upsampling)
       - 分離振幅和相位
       - 分別處理振幅和相位
       - 頻域 tile 2x2
       
    2. Spatial Upsampling (Pixel Shuffle)
       - Conv → PixelShuffle
       
    3. Fusion
       - 拼接兩個分支
       - 1x1 Conv 融合
    
特點:
✓ 頻域和空間域雙路徑
✓ DFU 保留相位信息
✓ 理論上更豐富的特徵
```

#### UpS (final_model_AGSSF1.py) - 頻域+空間上採樣
```python
class UpS:
    組成:
    1. UpSample (Frequency-based)
       - 分離振幅和相位
       - 1x1 Conv 處理
       - torch.tile(mag/pha, (2, 2))  # 錯誤：應該是 (1,1,2,2)
       - 重建複數 → IFFT
       
    2. UpSample1 (Spatial)
       - Conv → PixelShuffle
       
    3. Reduce
       - 拼接 → 1x1 Conv
    
特點:
✓ 類似的雙路徑設計
⚠ tile 實現有 bug
✓ 結構相對簡單
```

**主要差異**:
- HFSUB 的 DFU 使用 **Conv 處理振幅和相位**，UpSample 使用 **tile 直接擴展**
- HFSUB 更複雜但理論上更強
- UpS 有實現 bug (`torch.tile` 維度參數錯誤)

### 3.4 跳接融合

#### SSFB (spectroformer_paper.py) - 空間-頻譜融合注意力塊
```python
class SSFB:
    路徑:
    1. 空間路徑
       - Conv1x1 → PReLU → DWConv3x3
       - 殘差連接
       
    2. 頻譜路徑
       - FFT → Conv1x1 → GELU → Conv1x1 → IFFT
       - 殘差連接
       
    3. 融合
       - 空間 + 頻譜
       - AKC 通道注意力
       - 與解碼器特徵拼接
    
特點:
✓ 增強編碼器特徵
✓ 空間+頻域雙路徑
✓ AKC 自適應通道注意力
```

#### SFCA (final_model_AGSSF1.py) - 空間-頻率通道注意力
```python
class SFCA:
    路徑:
    1. 空間路徑
       - Conv1x1 → split → concat → LeakyReLU
       - DWConv3x3 → LeakyReLU
       - 殘差連接
       
    2. 頻率路徑
       - FFT.real → GELU → Conv1x1 → Conv1x1 → IFFT.real
       - 殘差連接
       
    3. 融合
       - 拼接 → Conv1x1
       - AGSSF 門控注意力
    
特點:
✓ 類似的雙路徑設計
✓ AGSSF 使用逆振幅特徵
✗ FFT 只用實部
```

**主要差異**:
- SSFB 使用 **AKC** (自適應核卷積)，SFCA 使用 **AGSSF** (自適應門控)
- SSFB 的頻譜路徑**保留複數**，SFCA **僅用實部**

### 3.5 前饋網路

#### GDFN (兩個模型相同)
```python
class GDFN:
    """門控深度卷積前饋網路"""
    結構:
    1. Conv1x1: C → 2*hidden_C
    2. DWConv3x3: groups=2*hidden_C
    3. Split: hidden_C, hidden_C
    4. Gate: GELU(x1) * x2
    5. Conv1x1: hidden_C → C
```

**無差異**: 兩個模型使用相同的 GDFN 實現

### 3.6 通道注意力機制

#### AKC (spectroformer_paper.py) - 自適應核卷積
```python
class AKC:
    核心:
    - 自適應核大小計算
      k = |log2(C)/γ + b/γ|_odd
    - Global Average Pooling
    - 1D Conv (kernel_size=k)
    - Sigmoid 激活
    
特點:
✓ 核大小根據通道數自適應
✓ 理論基礎：ECA-Net
✓ 輕量級設計
```

#### AGSSF (final_model_AGSSF1.py) - 自適應門控空間-頻譜特徵
```python
class AGSSF:
    核心:
    - inv_mag(x): 反振幅特徵
      fft → 保留相位 → ifft.real
    - Global Average Pooling
    - 1D Conv (自適應核大小)
    - Sigmoid 激活
    
特點:
✓ 使用頻域信息 (相位)
✓ 更複雜的特徵表示
✓ 理論創新
```

**主要差異**:
- AKC 直接在**空間域**操作
- AGSSF 先提取**頻域相位特徵** (inv_mag)，然後操作

---

## 4. 技術實現差異

### 4.1 LayerNorm 位置

| 模型 | LayerNorm 實現 |
|------|----------------|
| spectroformer_paper.py | 自定義 `LayerNorm` 類，支持 (B, C, H, W) |
| final_model_AGSSF1.py | 使用標準 `nn.LayerNorm`，需要手動轉換維度 |

### 4.2 下採樣

**相同**: 兩者都使用 **PixelUnshuffle**
```python
# spectroformer_paper.py
Conv(C, C//2) → PixelUnshuffle(2) → Conv(C*2, out_C)

# final_model_AGSSF1.py
Conv(C, C//2) → PixelUnshuffle(2)
```

### 4.3 初始化

| 特性 | spectroformer_paper.py | final_model_AGSSF1.py |
|------|------------------------|------------------------|
| 權重初始化 | Xavier Uniform | 無顯式初始化 |
| 偏置初始化 | 設為 0 | 預設 |
| LayerNorm | 權重=1, 偏置=0 | 預設 |

### 4.4 頻域操作對比

#### spectroformer_paper.py - 標準 FFT/IFFT
```python
# FDFP
x_fft = torch.fft.rfft2(x, norm='ortho')  # Real FFT
x_real = x_fft.real
x_imag = x_fft.imag
# ... 處理 ...
x = torch.fft.irfft2(x_fft_processed, s=(h, w), norm='ortho')

# DFU
x_fft = torch.fft.fft2(x)  # Complex FFT
mag = torch.abs(x_fft)
pha = torch.angle(x_fft)
```

#### final_model_AGSSF1.py - 簡化 FFT
```python
# SFCA / MDTA
x_fft = fft.fftn(x, dim=(-2, -1)).real  # 只取實部
x_fft = F.gelu(self.conv_f1(x_fft))
x_reconstructed = fft.ifftn(x_fft, dim=(-2, -1)).real

# inv_mag
fft_ = torch.fft.fft2(x)
fft_ = torch.fft.ifft2(1*torch.exp(1j*(fft_.angle())))  # 保留相位
return fft_.real
```

**差異總結**:
- spectroformer 使用**完整複數** (實部+虛部)
- final_model 多數情況**只用實部**，僅 inv_mag 提取相位

---

## 5. 性能與參數對比

### 5.1 模型參數量

#### spectroformer_small (spectroformer_paper.py)
```
配置:
  dim=16
  num_blocks=[2, 3, 3, 4]
  num_refinement_blocks=4
  num_heads=[1, 2, 4, 8]
  
參數量: 2.99M
```

#### mymodel (final_model_AGSSF1.py)
```
配置:
  channels=[16, 32, 64, 128]
  num_blocks=[2, 3, 3, 4]
  num_refinement=4
  num_heads=[1, 2, 4, 8]
  
參數量: ~3.5M (估計)
```

### 5.2 計算複雜度估計

| 操作 | spectroformer_paper.py | final_model_AGSSF1.py |
|------|------------------------|------------------------|
| **注意力** | MQCA (兩階段) | MDTA (雙路徑) |
| **頻域處理** | 多次完整 FFT/IFFT | 簡化 FFT |
| **跳接融合** | SSFB (複雜) | SFCA (較簡單) |
| **上採樣** | HFSUB (DFU+Spatial) | UpS (Freq+Spatial) |
| **總體複雜度** | ★★★★☆ | ★★★☆☆ |

### 5.3 記憶體使用

| 階段 | spectroformer_paper.py | final_model_AGSSF1.py |
|------|------------------------|------------------------|
| **前向傳播** | 較高 (保留完整複數) | 中等 (主要用實部) |
| **跳接** | 較高 (SSFB 雙路徑) | 中等 (SFCA) |
| **上採樣** | 中等 (HFSUB) | 中等 (UpS) |

---

## 6. 優缺點分析

### 6.1 spectroformer_paper.py (論文版)

#### 優點 ✅
1. **學術標準**
   - 嚴格遵循論文描述
   - 適合論文復現和對比實驗
   
2. **理論完整性**
   - MQCA 兩階段級聯設計有理論支撐
   - FDFP 完整處理複數頻域信息
   - SSFB 全面融合空間-頻譜特徵
   
3. **頻域處理**
   - 保留完整相位信息
   - 使用 `norm='ortho'` 的標準化 FFT
   
4. **代碼質量**
   - 詳細的文檔註釋
   - 清晰的模組劃分
   - 包含論文公式說明

#### 缺點 ❌
1. **計算複雜度**
   - MQCA 兩階段級聯增加計算量
   - HFSUB 的 DFU 處理複雜
   
2. **實現複雜**
   - 多個專用模組
   - 代碼量大 (~800 行)
   
3. **訓練難度**
   - 參數多，可能需要更長的訓練時間
   - 可能對超參數更敏感

### 6.2 final_model_AGSSF1.py (改進版)

#### 優點 ✅
1. **基於成熟架構**
   - Restormer 已被驗證有效
   - 訓練穩定性好
   
2. **創新改進**
   - AGSSF 使用頻域相位信息
   - inv_mag 函數的創新設計
   
3. **實現簡潔**
   - 代碼量少 (~280 行)
   - 模組化設計清晰
   
4. **計算效率**
   - MDTA 雙路徑並行
   - 簡化的頻域處理

#### 缺點 ❌
1. **理論依據不足**
   - 改進缺乏論文支撐
   - 某些設計選擇不明確
   
2. **頻域信息損失**
   - 多數操作僅用 FFT 實部
   - 可能損失重要相位信息
   
3. **實現錯誤**
   - `torch.tile` 維度參數錯誤
   - 需要修復才能正常訓練
   
4. **缺乏文檔**
   - 註釋較少
   - 模組設計意圖不清楚

---

## 7. 實現細節對比表

### 7.1 核心模組對比

| 模組 | spectroformer_paper.py | final_model_AGSSF1.py | 相似度 |
|------|------------------------|------------------------|--------|
| **Transformer Block** | MQCT | TransformerBlock | 50% |
| **注意力** | MQCA (兩階段級聯) | MDTA (雙路徑並行) | 40% |
| **FFN** | GDFN | GDFN | 100% |
| **跳接融合** | SSFB | SFCA | 60% |
| **上採樣** | HFSUB (DFU+Spatial) | UpS (Freq+Spatial) | 65% |
| **下採樣** | DownSample | DownSample | 95% |
| **通道注意力** | AKC | AGSSF | 50% |
| **頻域處理** | FDFP | 無專用模組 | 30% |

### 7.2 代碼對比統計

| 指標 | spectroformer_paper.py | final_model_AGSSF1.py |
|------|------------------------|------------------------|
| **總行數** | ~800 | ~280 |
| **類數量** | 11 | 10 |
| **參數量** | 2.99M | ~3.5M |
| **文檔註釋** | ★★★★★ | ★★☆☆☆ |
| **代碼複雜度** | 高 | 中 |

---

## 8. 使用建議

### 8.1 選擇 spectroformer_paper.py 的場景

✅ **推薦情況**:
1. **學術研究**
   - 論文復現
   - 基準測試
   - 與論文結果對比

2. **追求最佳性能**
   - 完整的頻域信息處理
   - 理論支撐的設計

3. **長期項目**
   - 代碼文檔完整
   - 易於理解和維護

### 8.2 選擇 final_model_AGSSF1.py 的場景

✅ **推薦情況**:
1. **快速實驗**
   - 代碼簡潔
   - 修改方便

2. **資源受限**
   - 計算效率較高
   - 記憶體佔用較少

3. **基於 Restormer 改進**
   - 已有 Restormer 經驗
   - 想要嘗試創新設計

⚠️ **注意**: 使用前需要**修復 `torch.tile` bug**!

### 8.3 修復建議

如果選擇使用 `final_model_AGSSF1.py`，建議進行以下修復：

#### 1. 修復 `torch.tile` bug
```python
# UpSample 類中，修改：
# 錯誤:
amp_fuse = torch.tile(Mag, (2, 2))
pha_fuse = torch.tile(Pha, (2, 2))

# 正確:
amp_fuse = torch.tile(Mag, (1, 1, 2, 2))
pha_fuse = torch.tile(Pha, (1, 1, 2, 2))
```

#### 2. 考慮改進頻域處理
```python
# 建議保留完整複數信息
# 當前只用實部：
x_fft = fft.fftn(x, dim=(-2, -1)).real  # 損失虛部

# 改進方案：
x_fft = fft.fftn(x, dim=(-2, -1))
x_real = x_fft.real
x_imag = x_fft.imag
x_freq = torch.cat([x_real, x_imag], dim=1)  # 通道拼接
```

#### 3. 添加權重初始化
```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

model = mymodel()
model.apply(init_weights)
```

---

## 9. 實驗對比建議

### 9.1 公平對比實驗設置

為了公平對比兩個模型，建議：

1. **相同的訓練配置**
   ```python
   batch_size=6
   learning_rate=0.001
   optimizer=Adam(beta1=0.5, beta2=0.999)
   epochs=550
   ```

2. **相同的數據增強**
   ```python
   transforms.Compose([
       transforms.Resize((256, 256)),
       transforms.ToTensor(),
   ])
   ```

3. **相同的損失函數**
   - Charbonnier Loss
   - VGG Perceptual Loss
   - Gradient Loss
   - MS-SSIM Loss

4. **相同的評估指標**
   - PSNR
   - SSIM
   - 訓練時間
   - GPU 記憶體使用

### 9.2 評估維度

| 維度 | 指標 |
|------|------|
| **準確性** | PSNR, SSIM |
| **效率** | 訓練時間/epoch, 推理速度 |
| **資源** | GPU 記憶體, 參數量 |
| **穩定性** | 損失曲線, 收斂速度 |
| **可維護性** | 代碼複雜度, 文檔完整性 |

---

## 10. 總結

### 核心差異
1. **設計理念**
   - spectroformer_paper: 論文驗證的完整系統
   - final_model_AGSSF1: 實驗性改進設計

2. **頻域處理**
   - spectroformer_paper: **完整複數** (實部+虛部)
   - final_model_AGSSF1: **簡化處理** (主要用實部)

3. **注意力機制**
   - spectroformer_paper: **級聯** (sequential)
   - final_model_AGSSF1: **並行** (parallel)

4. **實現質量**
   - spectroformer_paper: **高質量**，文檔完整
   - final_model_AGSSF1: **有 bug**，需要修復

### 推薦
- **學術研究**: spectroformer_paper.py ⭐⭐⭐⭐⭐
- **快速實驗**: final_model_AGSSF1.py (修復後) ⭐⭐⭐⭐
- **生產部署**: spectroformer_paper.py ⭐⭐⭐⭐⭐

---

**最後更新**: 2025-12-12
