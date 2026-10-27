"""
Training Script for MRI Image Denoising

Supports DnCNN, U-Net (baseline), and Improved U-Net (Attention U-Net)
with Rician or Gaussian noise on IXI dataset.

KEY CHANGE vs previous version:
- Added CombinedLoss: MSE + SSIM loss
  → Directly optimizes for both pixel accuracy AND structural similarity
  → Much better for medical image denoising than MSE alone
  → Use --loss combined for Improved U-Net (recommended)
  → Standard U-Net keeps --loss mse for fair baseline comparison
- Added AMP (Automatic Mixed Precision) support via --amp flag
  → Uses float16 for forward pass, float32 for optimizer updates
  → ~2x speedup on RTX 4090 with no accuracy loss

Author: Guerrouj Amine (20220996)
Date: 2025
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from models import get_model
from dataset import create_dataloaders
from metrics import evaluate_batch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# ─────────────────────────────────────────────────────────────
# SSIM Loss
# ─────────────────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """
    SSIM-based loss function.
    Minimizing (1 - SSIM) pushes the network to preserve
    structural similarity, edges, and contrast — critical for MRI.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size)

    def _gaussian(self, window_size, sigma):
        import math
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / (2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size):
        _1D = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D

    def _ssim(self, img1, img2, window, window_size, channel):
        window = window.to(img1.device)
        mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12   = nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return ssim_map.mean() if self.size_average else ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        return 1 - self._ssim(img1, img2, self.window, self.window_size, self.channel)


class CombinedLoss(nn.Module):
    """
    Combined MSE + SSIM Loss.

    total_loss = alpha * MSE + (1 - alpha) * SSIM_loss

    alpha=0.5 gives equal weight to pixel accuracy and structural similarity.
    This directly optimizes for what we measure (SSIM) while keeping
    pixel-level accuracy via MSE.
    """
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse  = nn.MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        # Model outputs can temporarily go outside [0,1] due to residual learning.
        # Clamping stabilizes SSIM computation and keeps the combined objective aligned
        # with evaluation, without changing the model architecture.
        pred_c = torch.clamp(pred, 0.0, 1.0)
        target_c = torch.clamp(target, 0.0, 1.0)
        return self.alpha * self.mse(pred_c, target_c) + (1 - self.alpha) * self.ssim(pred_c, target_c)


# ─────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Train MRI Denoising Model')

    parser.add_argument('--model', type=str, default='improved_unet',
                        choices=['dncnn', 'unet', 'improved_unet', 'resunet'])
    parser.add_argument('--num_layers', type=int, default=17)
    parser.add_argument('--features', type=int, default=64)

    parser.add_argument('--train_dir', type=str, default='../data/IXI_T1/train')
    parser.add_argument('--val_dir',   type=str, default=None)
    parser.add_argument('--noise_level', type=float, default=0.09)
    parser.add_argument('--noise_type',  type=str, default='rician',
                        choices=['rician', 'gaussian'])
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--num_slices',  type=int, default=30)

    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scheduler',    type=str,   default='cosine',
                        choices=['cosine', 'plateau', 'step', 'none'])
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['mse', 'l1', 'smooth_l1', 'combined'],
                        help='Loss function. Use combined for improved_unet, mse for unet baseline.')
    parser.add_argument('--ssim_alpha', type=float, default=0.5,
                        help='Weight for MSE in combined loss (default 0.5)')

    parser.add_argument('--amp', action='store_true',
                        help='Enable Automatic Mixed Precision (float16) — ~2x faster on RTX 4090')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='../models')
    parser.add_argument('--resume',   type=str, default=None)

    return parser.parse_args()


def get_loss_function(loss_name, ssim_alpha=0.5):
    if loss_name == 'mse':        return nn.MSELoss()
    if loss_name == 'l1':         return nn.L1Loss()
    if loss_name == 'smooth_l1':  return nn.SmoothL1Loss()
    if loss_name == 'combined':   return CombinedLoss(alpha=ssim_alpha)
    raise ValueError(f"Unknown loss: {loss_name}")


def get_scheduler(optimizer, name, epochs):
    if name == 'cosine':  return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    if name == 'plateau': return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    if name == 'step':    return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    return None


# ─────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    use_amp = scaler is not None
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for i, (noisy, clean) in enumerate(pbar):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                loss = criterion(model(noisy), clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{total_loss/(i+1):.6f}'})
    return total_loss / len(loader)


def validate(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss, metrics_list = 0, []
    use_amp = scaler is not None
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc='Validating'):
            noisy, clean = noisy.to(device), clean.to(device)
            if use_amp:
                with autocast():
                    out = model(noisy)
                    loss = criterion(out, clean)
            else:
                out = model(noisy)
                loss = criterion(out, clean)
            total_loss += loss.item()
            # Clamp for metrics to keep PSNR/SSIM meaningful.
            out_c = torch.clamp(out, 0.0, 1.0)
            clean_c = torch.clamp(clean, 0.0, 1.0)
            metrics_list.append(evaluate_batch(clean_c, out_c))
    return (
        total_loss / len(loader),
        np.mean([m['psnr'] for m in metrics_list]),
        np.mean([m['ssim'] for m in metrics_list])
    )


def save_checkpoint(model, optimizer, epoch, loss, path, is_best=False):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
    torch.save(ckpt, path)
    if is_best:
        torch.save(ckpt, os.path.join(os.path.dirname(path), 'best_model.pth'))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == 'cuda':
        # Speed optimizations for NVIDIA GPUs (safe for training)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    amp_available = device.type == 'cuda' and args.amp
    print(f"AMP (mixed precision): {'ON ✓' if amp_available else 'OFF'}")
    print(f"num_workers: {args.num_workers} | batch_size: {args.batch_size}")

    # AMP scaler — only created when amp is enabled on CUDA
    scaler = GradScaler() if amp_available else None

    # Model
    print(f"\nCreating {args.model.upper()}...")
    if args.model == 'dncnn':
        model = get_model('dncnn', num_layers=args.num_layers, features=args.features)
    else:
        model = get_model(args.model)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    print(f"\nLoading data — noise: {args.noise_type} {args.noise_level*100:.0f}%")
    train_loader, val_loader = create_dataloaders(
        args.train_dir, args.val_dir,
        batch_size=args.batch_size,
        noise_level=args.noise_level,
        noise_type=args.noise_type,
        num_workers=args.num_workers,
        num_slices_per_volume=args.num_slices,
        target_size=args.target_size
    )
    print(f"Train: {len(train_loader.dataset)} slices")
    if val_loader:
        print(f"Val:   {len(val_loader.dataset)} slices")

    # Training setup
    criterion = get_loss_function(args.loss, args.ssim_alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)

    start_epoch, best_psnr = 0, 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(os.path.join(args.save_dir, 'runs')) if HAS_TENSORBOARD else None
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    print(f"\n{'='*80}")
    print(f"Training: {args.model.upper()} | Loss: {args.loss} | "
          f"Noise: {args.noise_type} {args.noise_level*100:.0f}%")
    print(f"{'='*80}\n")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, scaler)
        history['train_loss'].append(train_loss)

        if val_loader:
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, scaler)
            history['val_loss'].append(val_loss)
            history['val_psnr'].append(val_psnr)
            history['val_ssim'].append(val_ssim)

            if args.scheduler == 'plateau': scheduler.step(val_loss)
            elif scheduler:                 scheduler.step()

            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val',   val_loss,   epoch)
                writer.add_scalar('PSNR',       val_psnr,   epoch)
                writer.add_scalar('SSIM',       val_ssim,   epoch)

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr

            print(f"Epoch {epoch+1}/{args.epochs} ({time.time()-t0:.1f}s) | "
                  f"Loss: {train_loss:.6f} | PSNR: {val_psnr:.2f} dB | "
                  f"SSIM: {val_ssim:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                  + (" *** BEST ***" if is_best else ""))
        else:
            if scheduler and args.scheduler != 'plateau': scheduler.step()
            is_best = False
            print(f"Epoch {epoch+1}/{args.epochs} ({time.time()-t0:.1f}s) | Loss: {train_loss:.6f}")

        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            save_checkpoint(model, optimizer, epoch+1, train_loss,
                            os.path.join(args.save_dir, f'{args.model}_epoch_{epoch+1}.pth'), is_best)
        elif is_best:
            save_checkpoint(model, optimizer, epoch+1, train_loss,
                            os.path.join(args.save_dir, f'{args.model}_best.pth'), is_best=True)

    total_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model}_final.pth'))
    if writer: writer.close()

    print(f"\n{'='*80}")
    print(f"Done in {total_time/3600:.1f}h | Best PSNR: {best_psnr:.2f} dB")
    print(f"Saved to: {args.save_dir}")
    print(f"{'='*80}")

    np.save(os.path.join(args.save_dir, f'{args.model}_history.npy'), history)


if __name__ == "__main__":
    main()