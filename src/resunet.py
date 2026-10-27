"""
ResU-Net — Residual U-Net for MRI Denoising
============================================
在标准 U-Net 基础上，将每个 ConvBlock 替换为带 shortcut 的残差块。
其余结构与 UNet baseline 完全相同，保证对比实验公平性。

唯一的改动：ConvBlock → ResConvBlock
    主路径：Conv → BN → ReLU → Conv → BN
    shortcut：当 in_ch ≠ out_ch 时，1×1 Conv + BN 对齐通道
    输出：ReLU(主路径 + shortcut)

架构：
    Encoder:    1 → 64 → 128 → 256 → 512
    Bottleneck: 512 → 1024
    Decoder:    1024 → 512 → 256 → 128 → 64
    Output:     残差输出 (input - predicted_noise)

参数量：约 31.5 M（比 UNet 多约 1.5%，shortcut 的 1×1 conv）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConvBlock(nn.Module):
    """
    残差卷积块（He et al. 2016）

    与普通 ConvBlock 的区别：
        1. 最后一个 ReLU 移到残差加法之后
        2. 增加 shortcut，让网络更容易学"恒等映射"
        3. 去噪任务中，低噪声区域网络倾向于输出接近 0 的修正量
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        # 主路径：最后一层 BN 之后不加 ReLU，先做残差加法再激活
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            # 注意：这里不加 ReLU
        )

        # Shortcut：通道不同时用 1×1 卷积对齐，相同时直接 identity
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))


class ResUNet(nn.Module):
    """
    Residual U-Net

    架构与 UNet baseline 完全相同，仅 ConvBlock → ResConvBlock。

    使用说明：
        model = ResUNet()
        noisy  = torch.randn(B, 1, 256, 256)
        clean_pred = model(noisy)       # 直接输出去噪结果
        loss = criterion(clean_pred, clean_gt)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)

        # ── Encoder ──────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc_blocks.append(ResConvBlock(ch, f))   # ← 唯一改动
            ch = f

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = ResConvBlock(features[-1], features[-1] * 2)  # ← 唯一改动

        # ── Decoder ───────────────────────────────────────────
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ResConvBlock(f * 2, f))  # ← 唯一改动
            ch = f

        # ── Output ────────────────────────────────────────────
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x  # 保留原始输入，用于残差输出

        # Encoder
        skips = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up(x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # 残差输出：与 UNet baseline 完全一致
        return input_x - self.out_conv(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResUNet()
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"ResUNet")
    print(f"  Input  : {x.shape}")
    print(f"  Output : {y.shape}")
    print(f"  Params : {model.count_params():,}")
    assert y.shape == x.shape
    print("  shape check passed")
