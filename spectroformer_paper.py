"""
Spectroformer - 論文復刻版本
===========================

此版本嚴格按照 Spectroformer 論文描述實作，包含以下核心模組：
1. MQCA (Multi-Domain Query Cascaded Attention) - 多域查詢級聯注意力
2. FDFP (Frequency Domain Feature Processor) - 頻域特徵處理器
3. SSFB (Spatio-Spectro Fusion-Based Attention Block) - 空間-頻譜融合注意力區塊
4. HFSUB (Hybrid Fourier-Spatial Upsampling Block) - 混合傅立葉-空間上採樣區塊
5. GDFN (Gated-Dconv Feed-Forward Network) - 門控深度卷積前饋網路
6. MQCT (Multi-Domain Query Cascaded Transformer Block) - Transformer 區塊

作者：根據 Spectroformer 論文復刻
日期：2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 基礎模組
# ==============================================================================

class LayerNorm(nn.Module):
    """
    適用於 (B, C, H, W) 格式的 Layer Normalization
    """
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        return self.body(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ==============================================================================
# GDFN - Gated-Dconv Feed-Forward Network (論文 Section 3.2)
# ==============================================================================

class GDFN(nn.Module):
    """
    門控深度卷積前饋網路 (Gated-Dconv Feed-Forward Network)
    
    論文公式：
    Y = Conv1×1(GELU(Conv1×1(X)W1) ⊙ DWConv3×3(Conv1×1(X)W2))
    
    其中 ⊙ 表示逐元素乘法（門控機制）
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(GDFN, self).__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        # 輸入投影：將通道數擴展到 hidden_features * 2（用於分割成兩個分支）
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        # 深度可分離卷積：應用於整個擴展特徵
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 
            kernel_size=3, stride=1, padding=1, 
            groups=hidden_features * 2, bias=bias
        )
        
        # 輸出投影：將通道數還原
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 1. 輸入投影 + 深度卷積
        x = self.dwconv(self.project_in(x))
        
        # 2. 分割成兩個分支
        x1, x2 = x.chunk(2, dim=1)
        
        # 3. 門控機制：GELU(x1) * x2
        x = F.gelu(x1) * x2
        
        # 4. 輸出投影
        x = self.project_out(x)
        
        return x


# ==============================================================================
# FDFP - Frequency Domain Feature Processor (論文 Section 3.3)
# ==============================================================================

class FDFP(nn.Module):
    """
    頻域特徵處理器 (Frequency Domain Feature Processor)
    
    論文描述：用於在頻域中處理 Query，捕捉全局頻率資訊
    
    流程：
    1. 對輸入進行 FFT 轉換到頻域
    2. 在頻域進行卷積處理（實部和虛部分開處理）
    3. 通過 IFFT 轉換回空間域
    """
    def __init__(self, dim, bias=False):
        super(FDFP, self).__init__()
        
        # 頻域處理：由於複數分解為實部和虛部，通道數翻倍
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. 快速傅立葉變換 (使用 rfft2 處理實數輸入)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 將複數分解為實部和虛部，堆疊為新的通道維度
        # x_fft: (B, C, H, W//2+1) complex
        x_real = x_fft.real  # (B, C, H, W//2+1)
        x_imag = x_fft.imag  # (B, C, H, W//2+1)
        x_freq = torch.cat([x_real, x_imag], dim=1)  # (B, 2C, H, W//2+1)
        
        # 3. 在頻域進行卷積處理
        x_freq = self.conv1(x_freq)
        x_freq = self.act(x_freq)
        x_freq = self.conv2(x_freq)
        
        # 4. 重新組合為複數形式
        x_real, x_imag = x_freq.chunk(2, dim=1)
        x_fft_processed = torch.complex(x_real, x_imag)
        
        # 5. 逆傅立葉變換回空間域
        x = torch.fft.irfft2(x_fft_processed, s=(h, w), norm='ortho')
        
        return x


# ==============================================================================
# MQCA - Multi-Domain Query Cascaded Attention (論文 Section 3.3)
# ==============================================================================

class MQCA(nn.Module):
    """
    多域查詢級聯注意力 (Multi-Domain Query Cascaded Attention)
    
    論文描述的兩階段級聯注意力機制：
    
    Stage 1 (空間域注意力):
        Q1, K1, V1 = Linear(DWConv(Linear(X)))
        Attn1 = Softmax(Q1 @ K1^T / √d) @ V1
        Out1 = Linear(Attn1)
    
    Stage 2 (頻域-空間混合注意力):
        Q2 = FDFP(X)  # 頻域處理的 Query
        K2, V2 = Linear(DWConv(Linear(Out1)))  # 從第一階段輸出生成
        Attn2 = Softmax(Q2 @ K2^T / √d) @ V2
        Out2 = Linear(Attn2)
    
    關鍵設計：
    - Q/K 進行 L2 正規化以穩定訓練
    - 可學習的溫度參數 (temperature)
    """
    def __init__(self, dim, num_heads=8, bias=False):
        super(MQCA, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 可學習的溫度參數
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Stage 1: Q, K, V 生成網路
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv1_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, 
            groups=dim * 3, bias=bias
        )
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Stage 2: 頻域特徵處理器 (用於生成 Q2)
        self.fdfp = FDFP(dim, bias)
        
        # Stage 2: K, V 生成網路 (從 Stage 1 輸出生成)
        self.kv2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv2_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, 
            groups=dim * 2, bias=bias
        )
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # ============= Stage 1: 空間域注意力 =============
        # 生成 Q1, K1, V1
        qkv1 = self.qkv1_dwconv(self.qkv1(x))
        q1, k1, v1 = qkv1.chunk(3, dim=1)
        
        # 重塑為多頭注意力格式
        q1 = q1.reshape(b, self.num_heads, self.head_dim, h * w)
        k1 = k1.reshape(b, self.num_heads, self.head_dim, h * w)
        v1 = v1.reshape(b, self.num_heads, self.head_dim, h * w)
        
        # L2 正規化 Q 和 K
        q1 = F.normalize(q1, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        
        # 計算注意力分數
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        
        # 應用注意力權重
        out1 = (attn1 @ v1).reshape(b, c, h, w)
        out1 = self.project_out1(out1)
        
        # ============= Stage 2: 頻域-空間混合注意力 =============
        # Q2 從原始輸入經過 FDFP 生成
        q2 = self.fdfp(x)
        
        # K2, V2 從 Stage 1 輸出生成
        kv2 = self.kv2_dwconv(self.kv2(out1))
        k2, v2 = kv2.chunk(2, dim=1)
        
        # 重塑為多頭注意力格式
        q2 = q2.reshape(b, self.num_heads, self.head_dim, h * w)
        k2 = k2.reshape(b, self.num_heads, self.head_dim, h * w)
        v2 = v2.reshape(b, self.num_heads, self.head_dim, h * w)
        
        # L2 正規化 Q 和 K
        q2 = F.normalize(q2, dim=-1)
        k2 = F.normalize(k2, dim=-1)
        
        # 計算注意力分數
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        
        # 應用注意力權重
        out2 = (attn2 @ v2).reshape(b, c, h, w)
        out2 = self.project_out2(out2)
        
        return out2


# ==============================================================================
# MQCT - Multi-Domain Query Cascaded Transformer Block (論文 Section 3.2)
# ==============================================================================

class MQCT(nn.Module):
    """
    多域查詢級聯 Transformer 區塊
    
    結構：
    X' = X + MQCA(LayerNorm(X))
    Y = X' + GDFN(LayerNorm(X'))
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super(MQCT, self).__init__()
        
        self.norm1 = LayerNorm(dim)
        self.attn = MQCA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 殘差連接 + MQCA
        x = x + self.attn(self.norm1(x))
        # 殘差連接 + GDFN
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# AKC - Adaptive Kernel Convolution (論文 Section 3.4)
# ==============================================================================

class AKC(nn.Module):
    """
    自適應核心卷積 (Adaptive Kernel Convolution)
    
    論文公式：
    k = |log2(C)/γ + b/γ|_odd
    其中 γ=2, b=1（預設值）
    
    用於 SSFB 中的通道注意力
    """
    def __init__(self, channels, gamma=2, b=1):
        super(AKC, self).__init__()
        
        self.channels = channels
        
        # 計算自適應核心大小
        t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
        k = t if t % 2 else t + 1  # 確保為奇數
        k = max(3, k)  # 最小核心大小為 3
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 1D 卷積用於通道間交互
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        
        # Sigmoid 激活
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        y = self.gap(x)
        
        # 重塑為 1D 卷積輸入: (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 1D 卷積
        y = self.conv(y)
        
        # 還原形狀: (B, 1, C) -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid 激活並應用通道注意力
        return x * self.sigmoid(y)


# ==============================================================================
# SSFB - Spatio-Spectro Fusion-Based Attention Block (論文 Section 3.4)
# ==============================================================================

class SSFB(nn.Module):
    """
    空間-頻譜融合注意力區塊 (Spatio-Spectro Fusion-Based Attention Block)
    
    論文描述：用於取代傳統的跳接連接，在編碼器特徵傳遞給解碼器前進行增強
    
    結構：
    1. 空間路徑：Conv1x1 -> PReLU -> DWConv3x3
    2. 頻譜路徑：FFT -> Conv1x1 -> GELU -> Conv1x1 -> IFFT
    3. 融合：空間特徵 + 頻譜特徵
    4. 通道注意力：AKC (Adaptive Kernel Convolution)
    5. 與解碼器特徵拼接並壓縮通道
    """
    def __init__(self, dim, bias=False):
        super(SSFB, self).__init__()
        
        # ============= 空間路徑 =============
        self.spatial_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.spatial_act = nn.PReLU()
        self.spatial_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, 
            groups=dim, bias=bias
        )
        
        # ============= 頻譜路徑 =============
        # 處理複數的實部和虛部（通道數翻倍）
        self.spectral_conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.spectral_act = nn.GELU()
        self.spectral_conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        
        # ============= 融合與注意力 =============
        self.fusion_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.akc = AKC(dim)
        
        # ============= 殘差連接 =============
        self.identity_spatial = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.identity_spectral = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # ============= 與解碼器融合 =============
        self.compress = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x_encoder, x_decoder):
        """
        Args:
            x_encoder: 來自編碼器的特徵 (B, C, H, W)
            x_decoder: 來自解碼器上一層的特徵 (B, C, H, W)
        
        Returns:
            融合後的特徵 (B, C, H, W)
        """
        b, c, h, w = x_encoder.shape
        
        # ============= 空間路徑 =============
        x_spatial = self.spatial_conv1(x_encoder)
        x_spatial = self.spatial_act(x_spatial)
        x_spatial = self.spatial_dwconv(x_spatial)
        x_spatial = x_spatial + self.identity_spatial(x_encoder)  # 殘差連接
        
        # ============= 頻譜路徑 =============
        # FFT
        x_fft = torch.fft.rfft2(x_encoder, norm='ortho')
        x_real = x_fft.real
        x_imag = x_fft.imag
        x_freq = torch.cat([x_real, x_imag], dim=1)
        
        # 頻域卷積
        x_freq = self.spectral_conv1(x_freq)
        x_freq = self.spectral_act(x_freq)
        x_freq = self.spectral_conv2(x_freq)
        
        # IFFT
        x_real, x_imag = x_freq.chunk(2, dim=1)
        x_fft_processed = torch.complex(x_real, x_imag)
        x_spectral = torch.fft.irfft2(x_fft_processed, s=(h, w), norm='ortho')
        x_spectral = x_spectral + self.identity_spectral(x_encoder)  # 殘差連接
        
        # ============= 融合 =============
        x_fused = x_spatial + x_spectral
        x_fused = self.fusion_conv(x_fused)
        
        # ============= 通道注意力 (AKC) =============
        x_fused = self.akc(x_fused)
        
        # ============= 與解碼器特徵拼接並壓縮 =============
        output = torch.cat([x_fused, x_decoder], dim=1)
        output = self.compress(output)
        
        return output


# ==============================================================================
# DFU - Deep Fourier Upsampling (論文 Section 3.5)
# ==============================================================================

class DFU(nn.Module):
    """
    深度傅立葉上採樣 (Deep Fourier Upsampling)
    
    論文描述：在頻域進行上採樣
    
    流程：
    1. FFT 轉換到頻域
    2. 分離振幅和相位
    3. 分別處理振幅和相位
    4. 頻域上採樣 (tile 2x2)
    5. IFFT 轉換回空間域
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super(DFU, self).__init__()
        
        # 振幅處理分支
        self.amp_conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=bias)
        self.amp_act = nn.LeakyReLU(0.1, inplace=False)
        self.amp_conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=bias)
        
        # 相位處理分支
        self.pha_conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=bias)
        self.pha_act = nn.LeakyReLU(0.1, inplace=False)
        self.pha_conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=bias)
        
        # 輸出通道調整
        self.post = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. FFT
        x_fft = torch.fft.fft2(x)
        
        # 2. 分離振幅和相位
        mag = torch.abs(x_fft)
        pha = torch.angle(x_fft)
        
        # 3. 處理振幅
        mag = self.amp_conv1(mag)
        mag = self.amp_act(mag)
        mag = self.amp_conv2(mag)
        
        # 4. 處理相位
        pha = self.pha_conv1(pha)
        pha = self.pha_act(pha)
        pha = self.pha_conv2(pha)
        
        # 5. 頻域上採樣 (tile 2x2)
        mag_up = torch.tile(mag, (2, 2))
        pha_up = torch.tile(pha, (2, 2))
        
        # 6. 重建複數頻譜
        real = mag_up * torch.cos(pha_up)
        imag = mag_up * torch.sin(pha_up)
        x_fft_up = torch.complex(real, imag)
        
        # 7. IFFT
        x_up = torch.fft.ifft2(x_fft_up)
        x_up = torch.abs(x_up)  # 取絕對值確保實數輸出
        
        # 8. 調整輸出通道
        output = self.post(x_up)
        
        return output


# ==============================================================================
# HFSUB - Hybrid Fourier-Spatial Upsampling Block (論文 Section 3.5)
# ==============================================================================

class HFSUB(nn.Module):
    """
    混合傅立葉-空間上採樣區塊 (Hybrid Fourier-Spatial Upsampling Block)
    
    論文描述：結合頻域上採樣 (DFU) 和空間域上採樣 (Pixel Shuffle)
    
    結構：
    1. 頻域上採樣分支 (DFU)
    2. 空間上採樣分支 (Pixel Shuffle)
    3. 兩分支拼接並壓縮通道
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super(HFSUB, self).__init__()
        
        # 頻域上採樣分支 (DFU)
        self.dfu = DFU(in_dim, out_dim, bias)
        
        # 空間上採樣分支 (Pixel Shuffle)
        self.spatial_up = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2)
        )
        
        # 融合壓縮
        self.fusion = nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 頻域上採樣
        x_freq = self.dfu(x)
        
        # 空間上採樣
        x_spatial = self.spatial_up(x)
        
        # 拼接並融合
        output = torch.cat([x_freq, x_spatial], dim=1)
        output = self.fusion(output)
        
        return output


# ==============================================================================
# DownSample - 下採樣模組
# ==============================================================================

class DownSample(nn.Module):
    """
    下採樣模組
    
    使用 PixelUnshuffle 進行下採樣，比 stride-2 卷積更穩定
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super(DownSample, self).__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.PixelUnshuffle(2)  # 空間維度 /2，通道數 *4
        )
        
        # PixelUnshuffle 後通道數 = in_dim // 2 * 4 = in_dim * 2
        # 需要調整到 out_dim
        self.adjust = nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.body(x)
        x = self.adjust(x)
        return x


# ==============================================================================
# Spectroformer - 完整模型 (論文 Figure 2)
# ==============================================================================

class Spectroformer(nn.Module):
    """
    Spectroformer 完整模型
    
    論文描述的 U-Net 結構 Transformer，用於水下影像增強
    
    架構：
    - 編碼器：4 個階段，每個階段包含多個 MQCT 區塊
    - 解碼器：3 個階段，使用 HFSUB 上採樣和 SSFB 跳接融合
    - 精煉層：額外的 MQCT 區塊用於精煉輸出
    
    參數：
        dim: 基礎通道數
        num_blocks: 每個階段的 Transformer 區塊數量
        num_refinement_blocks: 精煉層的區塊數量
        num_heads: 每個階段的注意力頭數
        ffn_expansion_factor: GDFN 的通道擴展因子
        bias: 是否使用偏置
    """
    def __init__(
        self,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        num_heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    ):
        super(Spectroformer, self).__init__()
        
        # ============= 淺層特徵提取 =============
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        
        # ============= 編碼器 (Encoder) =============
        # Level 1
        self.encoder_level1 = nn.Sequential(
            *[MQCT(dim=dim, num_heads=num_heads[0], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[0])]
        )
        self.down1_2 = DownSample(dim, dim * 2, bias=bias)
        
        # Level 2
        self.encoder_level2 = nn.Sequential(
            *[MQCT(dim=dim * 2, num_heads=num_heads[1], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[1])]
        )
        self.down2_3 = DownSample(dim * 2, dim * 4, bias=bias)
        
        # Level 3
        self.encoder_level3 = nn.Sequential(
            *[MQCT(dim=dim * 4, num_heads=num_heads[2], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[2])]
        )
        self.down3_4 = DownSample(dim * 4, dim * 8, bias=bias)
        
        # ============= 瓶頸層 (Bottleneck) =============
        self.bottleneck = nn.Sequential(
            *[MQCT(dim=dim * 8, num_heads=num_heads[3], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[3])]
        )
        
        # ============= 解碼器 (Decoder) =============
        # Level 3 Decoder
        self.up4_3 = HFSUB(dim * 8, dim * 4, bias=bias)
        self.ssfb3 = SSFB(dim * 4, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[MQCT(dim=dim * 4, num_heads=num_heads[2], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[2])]
        )
        
        # Level 2 Decoder
        self.up3_2 = HFSUB(dim * 4, dim * 2, bias=bias)
        self.ssfb2 = SSFB(dim * 2, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[MQCT(dim=dim * 2, num_heads=num_heads[1], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[1])]
        )
        
        # Level 1 Decoder
        self.up2_1 = HFSUB(dim * 2, dim, bias=bias)
        self.ssfb1 = SSFB(dim, bias=bias)
        self.decoder_level1 = nn.Sequential(
            *[MQCT(dim=dim, num_heads=num_heads[0], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_blocks[0])]
        )
        
        # ============= 精煉層 (Refinement) =============
        self.refinement = nn.Sequential(
            *[MQCT(dim=dim, num_heads=num_heads[0], 
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
              for _ in range(num_refinement_blocks)]
        )
        
        # ============= 輸出層 =============
        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        """
        前向傳播
        
        Args:
            inp: 輸入圖像 (B, 3, H, W)
        
        Returns:
            增強後的圖像 (B, 3, H, W)
        """
        # 保存輸入用於殘差學習
        inp_img = inp
        
        # ============= 淺層特徵提取 =============
        x = self.patch_embed(inp)
        
        # ============= 編碼器 =============
        enc1 = self.encoder_level1(x)
        x = self.down1_2(enc1)
        
        enc2 = self.encoder_level2(x)
        x = self.down2_3(enc2)
        
        enc3 = self.encoder_level3(x)
        x = self.down3_4(enc3)
        
        # ============= 瓶頸層 =============
        bottleneck = self.bottleneck(x)
        
        # ============= 解碼器 =============
        # Level 3: 上採樣 -> SSFB 融合 -> Transformer
        dec3 = self.up4_3(bottleneck)
        dec3 = self.ssfb3(enc3, dec3)
        dec3 = self.decoder_level3(dec3)
        
        # Level 2: 上採樣 -> SSFB 融合 -> Transformer
        dec2 = self.up3_2(dec3)
        dec2 = self.ssfb2(enc2, dec2)
        dec2 = self.decoder_level2(dec2)
        
        # Level 1: 上採樣 -> SSFB 融合 -> Transformer
        dec1 = self.up2_1(dec2)
        dec1 = self.ssfb1(enc1, dec1)
        dec1 = self.decoder_level1(dec1)
        
        # ============= 精煉與輸出 =============
        refined = self.refinement(dec1)
        out = self.output(refined)
        
        # 殘差學習：輸出 = 輸入 + 學習到的殘差
        return inp_img + out


# ==============================================================================
# 模型變體
# ==============================================================================

def spectroformer_small():
    """小型 Spectroformer (參考官方實作的配置)"""
    return Spectroformer(
        dim=16,
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=4,
        num_heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    )


def spectroformer_base():
    """基礎 Spectroformer (論文描述的配置)"""
    return Spectroformer(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        num_heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    )


def spectroformer_large():
    """大型 Spectroformer"""
    return Spectroformer(
        dim=64,
        num_blocks=[6, 8, 8, 10],
        num_refinement_blocks=6,
        num_heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False
    )


def mymodel():
    """供訓練腳本使用的模型函數（默認使用 small 版本）"""
    return spectroformer_small()


# ==============================================================================
# 測試程式碼
# ==============================================================================

if __name__ == '__main__':
    # 測試模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建小型模型進行測試
    model = spectroformer_small().to(device)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params / 1e6:.2f}M")
    print(f"可訓練參數量: {trainable_params / 1e6:.2f}M")
    
    # 測試前向傳播
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"輸入形狀: {x.shape}")
    print(f"輸出形狀: {y.shape}")
    
    # 測試基礎模型
    print("\n--- 基礎模型 ---")
    model_base = spectroformer_base().to(device)
    total_params_base = sum(p.numel() for p in model_base.parameters())
    print(f"基礎模型參數量: {total_params_base / 1e6:.2f}M")
