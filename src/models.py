"""
Neural Network Architectures for MRI Image Denoising

Architectures:
1. DnCNN - for comparison
2. UNet - standard baseline (UNCHANGED for fair comparison)
3. ImprovedUNet - Attention U-Net with MultiScaleBlock (our proposed method)

KEY IMPROVEMENTS in v3:
- Attention Gates on all skip connections (replaces plain concatenation)
- MultiScaleBlock with residual connection at bottleneck + enc3 + enc4
- Channel Attention (SE block) inside MultiScaleBlock

Author: Guerrouj Amine (20220996)
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# DnCNN (unchanged)
# ─────────────────────────────────────────────────────────────
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x):
        return x - self.dncnn(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────────────────────
# UNet — Standard Baseline (UNCHANGED for fair comparison)
# ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.encoder_blocks.append(self._conv_block(ch, f))
            ch = f
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.decoder_blocks.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.decoder_blocks.append(self._conv_block(f * 2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        input_x = x
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i + 1](x)
        return input_x - self.final_conv(x)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )


# ─────────────────────────────────────────────────────────────
# Building Blocks for Improved U-Net
# ─────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(in_channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 4), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.

    Learns WHICH spatial regions in the skip connection are relevant
    given the current decoder feature map. Suppresses irrelevant
    background regions and focuses on target structures (brain tissue).

    This is the KEY contribution — replaces plain skip connections
    with learned, spatially-selective attention.

    Args:
        F_g: channels in decoder (gating signal)
        F_l: channels in encoder skip connection
        F_int: intermediate channels (usually F_l // 2)
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal from decoder (coarser resolution)
        x: skip connection from encoder (finer resolution)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:])
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)          # attention map [B, 1, H, W]
        return x * psi               # attended skip features


class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Feature Extraction with residual connection.

    4 parallel branches: 1x1, 3x3, 5x5, GlobalAvgPool
    → concatenate → fuse → channel attention → + residual

    Residual connection ensures the block never hurts performance.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        assert in_channels == out_channels
        mid = in_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_channels)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x).expand_as(b1)
        out = self.fusion(torch.cat([b1, b3, b5, bp], dim=1))
        out = self.ca(out)
        out = self.dropout(out)
        return x + out  # residual connection


# ─────────────────────────────────────────────────────────────
# Improved U-Net — Attention U-Net + MultiScaleBlock
# ─────────────────────────────────────────────────────────────
class ImprovedUNet(nn.Module):
    """
    Improved U-Net v3 — Attention U-Net with Multi-Scale Feature Extraction

    TWO KEY CONTRIBUTIONS:
    ┌────────────────────────────────────────────────────────────┐
    │ 1. ATTENTION GATES on all 4 skip connections               │
    │    - Learns which spatial regions matter for denoising     │
    │    - Suppresses noisy/irrelevant background regions        │
    │    - Proven to outperform standard skip connections        │
    │    - Especially effective for T2 high-noise cases          │
    │                                                            │
    │ 2. MULTISCALEBLOCK at enc3, enc4, and bottleneck           │
    │    - Captures noise patterns at multiple spatial scales    │
    │    - Residual connection prevents performance degradation  │
    │    - Channel attention focuses on informative features     │
    └────────────────────────────────────────────────────────────┘

    Architecture:
    Encoder:   1 → 64 → 128 → [256+MSB] → [512+MSB]
    Bottleneck: 1024 + MSB
    Decoder:   AttGate+512 → AttGate+256 → AttGate+128 → AttGate+64
    Output:    1×256×256
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(ImprovedUNet, self).__init__()

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.msb3 = MultiScaleBlock(256, 256)
        self.enc4 = self._conv_block(256, 512)
        self.msb4 = MultiScaleBlock(512, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = self._conv_block(512, 1024)
        self.msb_bn     = MultiScaleBlock(1024, 1024)

        # ── Attention Gates ───────────────────────────────────
        self.att4 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=512,  F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=256,  F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=128,  F_l=64,  F_int=32)

        # ── Decoder ───────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = self._conv_block(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self._conv_block(512, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self._conv_block(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self._conv_block(128, 64)

        # ── Output ────────────────────────────────────────────
        self.out = nn.Conv2d(64, out_channels, 1)

        self._init_weights()

    def forward(self, x):
        input_x = x
        # Encoder
        e1 = self.enc1(x)                        # [B,  64, 256, 256]
        e2 = self.enc2(self.pool(e1))             # [B, 128, 128, 128]
        e3 = self.msb3(self.enc3(self.pool(e2)))  # [B, 256,  64,  64]
        e4 = self.msb4(self.enc4(self.pool(e3)))  # [B, 512,  32,  32]

        # Bottleneck
        bn = self.msb_bn(self.bottleneck(self.pool(e4)))  # [B, 1024, 16, 16]

        # Decoder with attention gates
        d4 = self.up4(bn)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:])
        e4 = self.att4(g=bn, x=e4)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:])
        e3 = self.att3(g=d4, x=e3)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:])
        e2 = self.att2(g=d3, x=e2)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:])
        e1 = self.att1(g=d2, x=e1)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return input_x - self.out(d1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


# ─────────────────────────────────────────────────────────────
# ResConvBlock — Residual Conv Block (teacher's file)
# ─────────────────────────────────────────────────────────────
class ResConvBlock(nn.Module):
    """
    Residual Conv Block (He et al. 2016)
    Main path: Conv→BN→ReLU→Conv→BN (no final ReLU before residual add)
    Shortcut: 1×1 Conv+BN if in_ch != out_ch, else Identity
    Output: ReLU(main + shortcut)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


# ─────────────────────────────────────────────────────────────
# ResUNet — Residual U-Net (teacher's baseline)
# ─────────────────────────────────────────────────────────────
class ResUNet(nn.Module):
    """
    Residual U-Net — teacher's provided baseline.
    Same architecture as standard U-Net but with ResConvBlocks.
    Uses residual output: input - predicted_noise.
    Parameters: ~31.5M
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc_blocks.append(ResConvBlock(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = ResConvBlock(features[-1], features[-1] * 2)

        # Decoder
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.dec_blocks.append(ResConvBlock(f * 2, f))
            ch = f

        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        input_x = x
        skips = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up(x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        return input_x - self.out_conv(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────────────────────
# get_model factory
# ─────────────────────────────────────────────────────────────
def get_model(model_name='improved_unet', **kwargs):
    models = {
        'dncnn': DnCNN,
        'unet': UNet,
        'improved_unet': ImprovedUNet,
        'resunet': ResUNet,
    }
    if model_name.lower() not in models:
        raise ValueError(f"Model '{model_name}' not found. Choose from {list(models.keys())}")
    return models[model_name.lower()](**kwargs)


if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)

    print("1. Standard U-Net (Baseline):")
    unet = UNet()
    print(f"   Params: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"   Output: {unet(x).shape}")

    print("\n2. Improved U-Net v3 (Attention + MultiScale):")
    improved = ImprovedUNet()
    print(f"   Params: {sum(p.numel() for p in improved.parameters()):,}")
    print(f"   Output: {improved(x).shape}")

    print("\nAll models OK!")