# Spectroformer (mymodel) 技術報告

## 概述

**Spectroformer** 是一個基於 Transformer 架構的影像復原網路，專門設計用於水下影像增強（Underwater Image Enhancement）任務。該模型結合了空間域和頻率域的特徵處理，透過創新的頻譜注意力機制來提升影像品質。

- **檔案位置**: `final_model_AGSSF1.py`
- **類別名稱**: `mymodel`
- **參數量**: ~2.53M (2,530,193 參數)

---

## 模型架構

### 整體架構圖

```
輸入 RGB 影像 (3, 256, 256)
           │
           ▼
    ┌──────────────────┐
    │  Embed Conv RGB  │  (3 → 16 channels)
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │   Encoder 1      │  TransformerBlock × 2 (16 ch, 1 head)
    └──────────────────┘
           │
     DownSample (PixelUnshuffle)
           │
           ▼
    ┌──────────────────┐
    │   Encoder 2      │  TransformerBlock × 3 (32 ch, 2 heads)
    └──────────────────┘
           │
     DownSample
           │
           ▼
    ┌──────────────────┐
    │   Encoder 3      │  TransformerBlock × 3 (64 ch, 4 heads)
    └──────────────────┘
           │
     DownSample
           │
           ▼
    ┌──────────────────┐
    │   Encoder 4      │  TransformerBlock × 4 (128 ch, 8 heads)
    └──────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────┐
    │              DECODER PATH                     │
    │  (with Skip Connections + SFCA Attention)     │
    └──────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────┐
    │   Refinement     │  TransformerBlock × 4
    └──────────────────┘
           │
           ▼
    ┌──────────────────┐
    │   Output Conv    │  (32 → 8 → 3 channels)
    └──────────────────┘
           │
           ▼
    輸出影像 (3, 256, 256)
```

---

## 核心模組詳解

### 1. 反向幅度函數 (inv_mag)

```python
def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1*torch.exp(1j*(fft_.angle())))
    return fft_.real
```

**功能**: 提取訊號的相位資訊，將幅度設為 1，僅保留相位成分。

**數學原理**:
- 對輸入進行 2D FFT：$F(u,v) = |F(u,v)| \cdot e^{j\phi(u,v)}$
- 僅保留相位：$F'(u,v) = 1 \cdot e^{j\phi(u,v)}$
- 進行逆 FFT 得到相位特徵

**應用**: 用於 AGSSF 模組中，提取影像的結構資訊。

---

### 2. AGSSF (Adaptive Global Spectral-Spatial Fusion)

**自適應全局頻譜空間融合模組**

```python
class AGSSF(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        # 自適應核大小計算
        # 1D 卷積用於通道注意力
        # Sigmoid 激活
```

**處理流程**:
1. 使用 `inv_mag()` 提取相位特徵
2. 全局平均池化 (GAP) 壓縮空間維度
3. 1D 卷積進行通道間交互（自適應核大小）
4. Sigmoid 生成注意力權重
5. 與原始特徵相乘

**自適應核大小公式**:
$$k = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|$$

其中 $C$ 為通道數，$\gamma=2$, $b=1$

---

### 3. SFCA (Spectral-Frequency Channel Attention)

**頻譜頻率通道注意力模組**

```python
class SFCA(nn.Module):
    # 雙分支架構：空間域 + 頻率域
```

**架構圖**:
```
輸入 x
    │
    ├─────────────────┐
    │                 │
    ▼                 ▼
┌─────────┐     ┌─────────┐
│ 空間分支 │     │ 頻率分支 │
│  Conv   │     │  FFT    │
│ LeakyReLU│    │  GELU   │
│  Conv   │     │  IFFT   │
└────┬────┘     └────┬────┘
     │               │
     │ + identity1   │ + identity2
     │               │
     └───────┬───────┘
             │
        Concatenate
             │
         Conv 2→1
             │
           AGSSF
             │
           輸出
```

**處理步驟**:
1. **空間分支**: 1×1 擴展 → LeakyReLU → 3×3 深度卷積 → 殘差連接
2. **頻率分支**: FFT → GELU → 1×1 卷積 → IFFT → 殘差連接
3. **融合**: 拼接兩分支 → 1×1 降維 → AGSSF 注意力

---

### 4. MDTA (Multi-Dconv Transposed Attention)

**多深度卷積轉置注意力**

這是一個結合空間域和頻率域的混合注意力機制。

**空間域分支**:
```python
q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
# Multi-head attention
attn = softmax(q @ k.T * temperature)
out = project_out(attn @ v)
```

**頻率域分支**:
```python
x_fft = fft.fftn(x).real
x_fft = GELU(x_fft)
qf = fft.ifftn(conv(x_fft)).real
# 使用空間域的 k 與頻率域的 q 進行交叉注意力
attnf = softmax(qf @ k.T * temperature)
outf = project_outf(attnf @ vf)
```

**特點**:
- 使用可學習的 temperature 參數
- 頻率域 query 與空間域 key 進行交叉注意力
- 輸出為頻率增強的特徵

---

### 5. GDFN (Gated-Dconv Feed-Forward Network)

**門控深度卷積前饋網路**

```python
class GDFN(nn.Module):
    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x
```

**架構**:
- 1×1 卷積擴展到 `2 × expansion_factor × channels`
- 3×3 深度可分離卷積
- GELU 門控機制：$\text{GELU}(x_1) \odot x_2$
- 1×1 卷積投影回原通道數

---

### 6. TransformerBlock

**完整的 Transformer 區塊**

```
輸入 x
    │
LayerNorm1 → MDTA → + (殘差)
    │
LayerNorm2 → GDFN → + (殘差)
    │
輸出
```

---

### 7. UpSample (頻域上採樣)

**基於頻率域的上採樣模組**

```python
class UpSample(nn.Module):
    def forward(self, x):
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)      # 幅度
        pha_x = torch.angle(fft_x)    # 相位
        
        Mag = self.amp_fuse(mag_x)    # 處理幅度
        Pha = self.pha_fuse(pha_x)    # 處理相位
        
        # 頻譜平鋪實現上採樣
        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))
        
        # 重建複數頻譜
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        
        return torch.abs(torch.fft.ifft2(out))
```

**原理**: 在頻率域中進行幅度和相位的平鋪（tiling），然後通過逆 FFT 實現 2× 上採樣。

---

### 8. UpS (混合上採樣)

```python
class UpS(nn.Module):
    def forward(self, x):
        # 頻域上採樣
        freq_up = self.Fups(x)
        # 空間域上採樣 (PixelShuffle)
        spatial_up = self.Sups(x)
        # 融合
        return self.reduce(torch.cat([freq_up, spatial_up], dim=1))
```

**特點**: 結合頻率域和空間域的上採樣方法。

---

### 9. DownSample

```python
class DownSample(nn.Module):
    def forward(self, x):
        # Conv 3×3 → PixelUnshuffle(2)
        return self.body(x)
```

**效果**: 空間尺寸減半，通道數加倍。

---

## 訓練配置

### 損失函數

訓練使用 **4 種損失函數的動態加權組合**：

#### 1. Charbonnier Loss
```python
loss = torch.mean(torch.sqrt((diff * diff) + (eps * eps)))
```
- 平滑版的 L1 損失
- $\epsilon = 10^{-3}$
- 對異常值更魯棒

#### 2. VGG Perceptual Loss
```python
loss = F.l1_loss(VGG(pred), VGG(target))
```
- 使用 VGG19 前 35 層
- 提取高層語義特徵
- 增強感知品質

#### 3. Gradient Loss (Laplacian)
```python
kernel = [[0,1,0], [1,-4,1], [0,1,0]]  # Laplacian kernel
loss = L1(conv(pred, kernel), conv(target, kernel))
```
- 使用拉普拉斯算子提取邊緣
- 保持銳利的邊緣和細節

#### 4. MS-SSIM Loss
```python
loss = 1 - MS_SSIM(pred, target)
```
- 多尺度結構相似性
- 視窗大小: 11, sigma: 1.5
- 保持結構一致性

### 動態權重學習

```python
class WeightedLoss(nn.Module):
    def __init__(self, num_weights):
        self.weights = nn.Parameter(torch.rand(1, num_weights))
        self.softmax_l = nn.Softmax(dim=1)
    
    def forward(self, *losses):
        weights = self.softmax_l(self.weights)
        return sum(loss * weight for loss, weight in zip(losses, weights))
```

**特點**: 損失權重是可學習的參數，訓練過程中自動調整各損失的重要性。

### 最終損失

$$\mathcal{L}_{total} = w_1 \cdot \mathcal{L}_{char} + w_2 \cdot \mathcal{L}_{per} + w_3 \cdot \mathcal{L}_{grad} + w_4 \cdot \mathcal{L}_{ssim}$$

其中 $w_i$ 通過 Softmax 正規化，且會在訓練過程中學習。

---

## 資料流程

### 編碼器階段

| 階段 | 輸入尺寸 | 輸出尺寸 | 通道數 | Transformer Blocks |
|------|----------|----------|--------|---------------------|
| Embed | 256×256 | 256×256 | 3 → 16 | - |
| Encoder 1 | 256×256 | 256×256 | 16 | 2 |
| Down 1 | 256×256 | 128×128 | 16 → 32 | - |
| Encoder 2 | 128×128 | 128×128 | 32 | 3 |
| Down 2 | 128×128 | 64×64 | 32 → 64 | - |
| Encoder 3 | 64×64 | 64×64 | 64 | 3 |
| Down 3 | 64×64 | 32×32 | 64 → 128 | - |
| Encoder 4 | 32×32 | 32×32 | 128 | 4 |

### 解碼器階段

| 階段 | 操作 | 通道變化 |
|------|------|----------|
| Up 1 | UpS + SFCA + Concat | 128 → 64 |
| Decoder 1 | TransformerBlock × 3 | 64 |
| Up 2 | UpS + SFCA + Concat | 64 → 32 |
| Decoder 2 | TransformerBlock × 3 | 32 |
| Up 3 | UpS + SFCA + Concat | 32 → 32 |
| Decoder 3 | TransformerBlock × 2 | 32 |
| Refinement | TransformerBlock × 4 | 32 |
| Output | Conv layers | 32 → 8 → 3 |

---

## 關鍵創新點

### 1. 頻譜-空間雙域處理
- SFCA 模組同時處理空間和頻率特徵
- MDTA 使用交叉注意力融合兩域資訊
- 頻域上採樣保留更多頻譜資訊

### 2. 相位感知注意力
- `inv_mag()` 函數提取純相位資訊
- AGSSF 使用相位特徵計算通道注意力
- 相位對結構資訊更敏感

### 3. 自適應損失權重
- 4 種損失函數的權重自動學習
- Softmax 確保權重和為 1
- 適應不同訓練階段的需求

### 4. 混合上採樣
- 結合 PixelShuffle (空間域) 和頻譜平鋪 (頻率域)
- 避免棋盤效應
- 保留更多高頻細節

---

## 訓練參數建議

| 參數 | 建議值 | 說明 |
|------|--------|------|
| Batch Size | 4-8 | 受 VGG 感知損失影響較大 |
| Learning Rate | 1e-4 | Adam 優化器 |
| Epochs | 200-500 | 根據資料集大小調整 |
| Image Size | 256×256 | 訓練時的輸入尺寸 |
| Beta1 | 0.5 | Adam 的動量參數 |

---

## 評估指標

- **PSNR** (Peak Signal-to-Noise Ratio): 衡量像素級精確度
- **SSIM** (Structural Similarity Index): 衡量結構相似性

---

## 檔案結構

```
Spectroformer_model/
├── final_model_AGSSF1.py    # Spectroformer 模型定義
├── trains.py                 # 訓練腳本
├── phaseformer.py           # Phaseformer 模型（另一版本）
├── LSUI/                    # 資料集
│   ├── train/
│   │   ├── input/           # 水下退化影像
│   │   └── gt/              # Ground Truth
│   └── test/
│       ├── input/
│       └── gt/
└── checkpoint/              # 模型檢查點
```

---

## 結論

Spectroformer 是一個創新的水下影像增強模型，其核心貢獻包括：

1. **頻譜-空間雙域 Transformer 架構**
2. **相位感知的通道注意力機制 (AGSSF)**
3. **頻率域上採樣技術**
4. **自適應多損失學習策略**

這些設計使模型能夠有效處理水下影像的色偏、霧化和對比度低等問題。
