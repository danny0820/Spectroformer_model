import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """
    論文中使用的 Layer Normalization。
    將輸入張量的最後一個維度 (通道維度) 進行標準化。
    """
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        # 輸入 x 的維度應為 (B, C, H, W)
        # LayerNorm 需要 (B, H, W, C)
        return self.body(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN)。
    此模組源於 Restormer 論文，是 Spectroformer FFN 的具體實現。
    它包含一個門控機制，能更有效地處理特徵。
    """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(GDFN, self).__init__()

        # 計算隱藏層的特徵維度
        hidden_features = int(dim * ffn_expansion_factor)

        # 輸入投射層：使用 1x1 卷積將通道數從 dim 擴展到 hidden_features * 2
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 深度可分離卷積層 (Depth-wise Convolution)
        # 只作用於其中一個分支，用於捕捉局部空間資訊
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        # 輸出投射層：使用 1x1 卷積將通道數從 hidden_features 還原到 dim
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 1. 輸入投射，並將特徵圖在通道維度上分割成兩部分：x1 和 x2
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        
        # 2. 門控機制：
        #    - 第一個分支 x1 通過 GELU 激活函數
        #    - 第二個分支 x2 通過 3x3 深度卷積
        #    - 兩者逐元素相乘 (element-wise multiplication)
        x = F.gelu(x1) * self.dwconv(x2)
        
        # 3. 輸出投射，還原通道數
        x = self.project_out(x)
        
        return x

class FDFP(nn.Module):
    """
    頻域特徵處理器 (Frequency Domain Feature Processor)。
    此模組用於在頻域中處理查詢 (Query)，以捕捉全局資訊。
    """
    def __init__(self, dim, bias=False):
        super(FDFP, self).__init__()
        # 由於複數會分解為實部和虛部，輸入通道數會翻倍
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)

    def forward(self, x):
        # 取得輸入的空間維度
        _, _, H, W = x.shape
        
        # 快速傅立葉變換 (FFT)
        # torch.fft.rfft2 處理實數輸入，輸出複數頻譜
        x_fft = torch.fft.rfft2(x, norm='ortho')

        # 在頻域中進行卷積操作
        # 將複數分解為實部和虛部，堆疊成新的通道維度
        x_fft_real = torch.stack([x_fft.real, x_fft.imag], dim=1)
        B, _, C, H_freq, W_freq = x_fft_real.shape
        x_fft_real = x_fft_real.reshape(B, C*2, H_freq, W_freq)
        
        # 對實數張量進行卷積
        x_fft_real = self.conv1(x_fft_real)
        x_fft_real = self.act(x_fft_real)
        x_fft_real = self.conv2(x_fft_real)
        
        # 重新組合為複數形式
        x_fft_real = x_fft_real.reshape(B, 2, C, H_freq, W_freq)
        x_fft = torch.complex(x_fft_real[:, 0], x_fft_real[:, 1])
        
        # 逆快速傅立葉變換 (IFFT)
        # torch.fft.irfft2 將頻譜轉換回空間域
        x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        
        return x

class MQCA(nn.Module):
    """
    多域查詢級聯注意力 (Multi-Domain Query Cascaded Attention)。
    這是 Spectroformer 的核心注意力機制，分為兩個階段。
    第一階段在空間域處理，第二階段結合了空間域的 Key/Value 和頻域的 Query。
    """
    def __init__(self, dim, num_heads=8, bias=False):
        super(MQCA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # 可學習的溫度參數 a

        # 頻域特徵處理器 (FDFP)
        self.q_fdfp = FDFP(dim, bias)

        # Q, K, V 生成網路
        # 使用 1x1 卷積後接 3x3 深度可分離卷積
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # --- 第一階段注意力 ---
        qkv1 = self.qkv_dwconv(self.qkv(x))
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        # 重塑 Q, K, V 以進行矩陣乘法
        q1 = q1.reshape(b, self.num_heads, -1, h * w)
        k1 = k1.reshape(b, self.num_heads, -1, h * w)
        v1 = v1.reshape(b, self.num_heads, -1, h * w)

        # 計算注意力分數
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        
        # 應用注意力權重到 V
        out1 = (attn1 @ v1).reshape(b, c, h, w)

        # --- 第二階段注意力 ---
        # 從第一階段的輸出生成 K2 和 V2
        kv2 = self.qkv_dwconv(self.qkv(out1))
        # 論文中指出 Q2 是由原始輸入 x 經過 FDFP 生成的
        q2 = self.q_fdfp(x) 
        _, k2, v2 = kv2.chunk(3, dim=1)

        # 重塑
        q2 = q2.reshape(b, self.num_heads, -1, h * w)
        k2 = k2.reshape(b, self.num_heads, -1, h * w)
        v2 = v2.reshape(b, self.num_heads, -1, h * w)

        # 計算注意力分數
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        
        # 應用注意力權重
        out2 = (attn2 @ v2).reshape(b, c, h, w)

        # 最後的線性投射
        output = self.project_out(out2)
        return output

class MQCT(nn.Module):
    """
    多域查詢級聯 Transformer 區塊 (Multi-Domain Query Cascaded Transformer Block)。
    標準的 Transformer 區塊結構，包含 LayerNorm, MQCA 注意力, 和 FFN。
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super(MQCT, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MQCA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 殘差連接
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SSFB(nn.Module):
    """
    空間-頻譜融合注意力區塊 (Spatio-Spectro Fusion-Based Attention Block)。
    用於取代傳統的跳接 (skip-connection)，對編碼器的特徵進行增強後再傳遞給解碼器。
    """
    def __init__(self, dim, bias=False):
        super(SSFB, self).__init__()
        
        # 空間處理路徑
        self.spatial_conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.spatial_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.spatial_act = nn.PReLU()

        # 頻譜處理路徑 (處理複數需要翻倍通道數)
        self.spectral_conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.spectral_act = nn.GELU()
        self.spectral_conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        
        # 融合後的處理
        self.conv_fuse = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 通道注意力 (Adaptive Kernel Convolution)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # 最後壓縮通道的卷積層
        self.conv_compress = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        
        # 預先定義可能的通道調整卷積層
        self.channel_adjust_2x = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.channel_adjust_4x = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False)
        self.channel_adjust_8x = nn.Conv2d(dim * 8, dim, kernel_size=1, bias=False)
        
        self.dim = dim

    def forward(self, x_encoder, x_decoder):
        # x_encoder: 來自編碼器的特徵
        # x_decoder: 來自解碼器上一層的特徵
        
        # 確保兩個輸入的空間維度一致
        if x_encoder.shape[2:] != x_decoder.shape[2:]:
            # 如果尺寸不匹配，將x_decoder上採樣到x_encoder的尺寸
            x_decoder = F.interpolate(x_decoder, size=x_encoder.shape[2:], mode='bilinear', align_corners=False)
        
        # 確保兩個輸入的通道維度一致
        if x_encoder.shape[1] != x_decoder.shape[1]:
            # 根據通道倍數選擇適當的調整層
            decoder_channels = x_decoder.shape[1]
            encoder_channels = x_encoder.shape[1]
            
            if decoder_channels == encoder_channels * 2:
                x_decoder = self.channel_adjust_2x(x_decoder)
            elif decoder_channels == encoder_channels * 4:
                x_decoder = self.channel_adjust_4x(x_decoder)
            elif decoder_channels == encoder_channels * 8:
                x_decoder = self.channel_adjust_8x(x_decoder)
            else:
                # 如果不是標準倍數，使用插值調整
                x_decoder = F.adaptive_avg_pool2d(x_decoder, x_encoder.shape[2:])
                # 使用1x1卷積調整通道數
                if not hasattr(self, f'channel_adjust_{decoder_channels}_to_{encoder_channels}'):
                    setattr(self, f'channel_adjust_{decoder_channels}_to_{encoder_channels}',
                           nn.Conv2d(decoder_channels, encoder_channels, kernel_size=1, bias=False).to(x_encoder.device))
                adjust_layer = getattr(self, f'channel_adjust_{decoder_channels}_to_{encoder_channels}')
                x_decoder = adjust_layer(x_decoder)
        
        # --- 空間路徑 ---
        x_s = self.spatial_dwconv(self.spatial_act(self.spatial_conv1(x_encoder)))
        
        # --- 頻譜路徑 ---
        _, _, H, W = x_encoder.shape
        x_f = torch.fft.rfft2(x_encoder, norm='ortho')
        
        # 將複數分解為實部和虛部進行卷積處理
        x_f_real = torch.stack([x_f.real, x_f.imag], dim=1)
        B, _, C, H_freq, W_freq = x_f_real.shape
        x_f_real = x_f_real.reshape(B, C*2, H_freq, W_freq)
        
        # 對實數張量進行卷積
        x_f_real = self.spectral_conv2(self.spectral_act(self.spectral_conv1(x_f_real)))
        
        # 重新組合為複數形式
        x_f_real = x_f_real.reshape(B, 2, C, H_freq, W_freq)
        x_f = torch.complex(x_f_real[:, 0], x_f_real[:, 1])
        
        x_f = torch.fft.irfft2(x_f, s=(H, W), norm='ortho')

        # 融合空間和頻譜特徵
        x_fused = self.conv_fuse(x_s + x_f)
        
        # --- 自適應核心通道注意力 ---
        # 根據論文公式計算自適應核心大小 [cite: 236, 237, 238]
        # k = |log2(C) + b / a|_odd, a=1, b=2
        C = self.dim
        t = int(abs((math.log(C, 2) + 2) / 1))
        k = t if t % 2 else t + 1
        
        attn_weights = self.gap(x_fused)
        # 這裡的 AKC (Adaptive kernel Convolution) 是一個 1D 卷積
        akc = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False).to(x_encoder.device)
        attn_weights = akc(attn_weights.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attn_weights = self.sigmoid(attn_weights)
        
        x_attentive = x_fused * attn_weights
        
        # 將注意力增強後的特徵與解碼器特徵拼接
        # 論文圖示是拼接後通過 1x1 卷積將通道數減半 [cite: 213]
        output = torch.cat([x_attentive, x_decoder], dim=1)
        output = self.conv_compress(output)
        
        return output

class HFSUB(nn.Module):
    """
    混合傅立葉-空間上採樣區塊 (Hybrid Fourier-Spatial Upsampling Block)。
    結合了像素重組 (Pixel Shuffle) 和深度傅立葉上採樣 (DFU) 來提升特徵解析度。
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super(HFSUB, self).__init__()
        
        # 空間上採樣路徑 (Pixel Shuffle)
        self.spatial_upsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, bias=bias),
            nn.PixelShuffle(2)
        )
        self.conv_block = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=bias)
        
        # 添加通道調整層，用於處理skip_feature的通道數不匹配問題
        self.channel_adjust = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias)

    def forward(self, x, skip_feature):
        # x: 來自解碼器上一層的特徵
        # skip_feature: 來自 SSFB 的跳接特徵
        
        # 空間上採樣 (Pixel Shuffle)
        x_up = self.spatial_upsample(x)

        # 調整skip_feature的通道數和空間尺寸以匹配x_up
        if skip_feature.shape[1] != x_up.shape[1]:
            # 使用通道調整層調整通道數
            skip_feature_adjusted = self.channel_adjust(skip_feature)
        else:
            skip_feature_adjusted = skip_feature
            
        # 確保空間維度匹配
        if x_up.shape[2:] != skip_feature_adjusted.shape[2:]:
            # 將skip_feature調整到x_up的尺寸
            skip_feature_adjusted = F.interpolate(skip_feature_adjusted, size=x_up.shape[2:], mode='bilinear', align_corners=False)
        
        # 論文圖示將上採樣結果與跳接特徵拼接，然後通過 DFU
        # 這裡我們簡化實現為加法融合然後進行卷積融合
        output = self.conv_block(x_up + skip_feature_adjusted)
        
        return output

class Spectroformer(nn.Module):
    """
    Spectroformer 完整模型。
    這是一個 U-Net 結構的 Transformer，用於水下影像增強。
    """
    def __init__(self, 
                 dim=48, 
                 num_blocks=[4, 6, 6, 8], 
                 num_refinement_blocks=4,
                 num_heads=[1, 2, 4, 8], 
                 ffn_expansion_factor=2.66, 
                 bias=False):
        super(Spectroformer, self).__init__()

        # --- 淺層特徵提取 ---
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # --- 編碼器 (Encoder) ---
        self.encoder_level1 = nn.Sequential(*[MQCT(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])
        self.down1_2 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1, bias=bias) # Pixel Unshuffle
        
        self.encoder_level2 = nn.Sequential(*[MQCT(dim=dim * 2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[1])])
        self.down2_3 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1, bias=bias) # Pixel Unshuffle

        self.encoder_level3 = nn.Sequential(*[MQCT(dim=dim * 4, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])
        self.down3_4 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1, bias=bias) # Pixel Unshuffle

        # --- 瓶頸層 (Bottleneck) ---
        self.latent = nn.Sequential(*[MQCT(dim=dim * 8, num_heads=num_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[3])])

        # --- 解碼器 (Decoder) ---
        self.up4_3 = HFSUB(dim * 8, dim * 4, bias=bias)
        self.ssfb3 = SSFB(dim * 4, bias=bias)
        self.decoder_level3 = nn.Sequential(*[MQCT(dim=dim * 4, num_heads=num_heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = HFSUB(dim * 4, dim * 2, bias=bias)
        self.ssfb2 = SSFB(dim * 2, bias=bias)
        self.decoder_level2 = nn.Sequential(*[MQCT(dim=dim * 2, num_heads=num_heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[1])])
        
        self.up2_1 = HFSUB(dim * 2, dim, bias=bias)
        self.ssfb1 = SSFB(dim, bias=bias)
        self.decoder_level1 = nn.Sequential(*[MQCT(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks[0])])

        # --- 影像重建 ---
        self.refinement = nn.Sequential(*[MQCT(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        # 儲存原始輸入，用於最後的殘差學習
        inp_img = inp

        # 淺層特徵提取
        x1 = self.patch_embed(inp)
        
        # --- 編碼器 ---
        enc1 = self.encoder_level1(x1)
        x2 = self.down1_2(enc1)
        
        enc2 = self.encoder_level2(x2)
        x3 = self.down2_3(enc2)
        
        enc3 = self.encoder_level3(x3)
        x4 = self.down3_4(enc3)

        # --- 瓶頸層 ---
        latent_out = self.latent(x4)

        # --- 解碼器 ---
        # 第一層解碼：從瓶頸層開始上採樣
        dec3 = self.up4_3(latent_out, latent_out)
        skip3_fused = self.ssfb3(enc3, dec3)
        dec3 = self.decoder_level3(skip3_fused)
        
        dec2 = self.up3_2(dec3, dec3)
        skip2_fused = self.ssfb2(enc2, dec2)
        dec2 = self.decoder_level2(skip2_fused)

        dec1 = self.up2_1(dec2, dec2)
        skip1_fused = self.ssfb1(enc1, dec1)
        dec1 = self.decoder_level1(skip1_fused)

        # --- 影像重建 ---
        refined = self.refinement(dec1)
        out = self.output(refined)
        
        # 最終輸出是原始輸入加上網路學習到的殘差
        return inp_img + out