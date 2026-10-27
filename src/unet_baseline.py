"""
UNet Baseline — Standard U-Net for MRI Denoising
=================================================
干净的基础 U-Net，用于对比实验。
结构与 Ronneberger et al. 2015 保持一致。

架构：
    Encoder:    1 → 64 → 128 → 256 → 512
    Bottleneck: 512 → 1024
    Decoder:    1024 → 512 → 256 → 128 → 64
    Output:     残差输出 (input - predicted_noise)

参数量：约 31 M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """两层 3×3 Conv + BN + ReLU。U-Net 的基本单元。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    标准 U-Net（基础对比版本）

    使用说明：
        model = UNet()
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
            self.enc_blocks.append(ConvBlock(ch, f))
            ch = f

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # ── Decoder ───────────────────────────────────────────
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(f * 2, f))
            ch = f

        # ── Output ────────────────────────────────────────────
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x  # 保留原始输入，用于残差输出

        # Encoder：保存每层输出用于 skip connection
        skips = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder：上采样 + skip connection + conv
        skips = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up(x)
            skip = skips[i]
            # 尺寸对齐（应对非 2 的幂次输入）
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # 残差输出：网络预测噪声，输入减去噪声得到 clean image
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
    model = UNet()
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"UNet")
    print(f"  Input  : {x.shape}")
    print(f"  Output : {y.shape}")
    print(f"  Params : {model.count_params():,}")
    assert y.shape == x.shape
    print("  shape check passed")
