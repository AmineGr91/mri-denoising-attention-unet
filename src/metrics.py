"""
Evaluation Metrics for Image Denoising

This module implements standard metrics for evaluating denoising performance:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)

Author: Thesis Project
Date: 2025
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    PSNR measures the ratio between maximum signal power and noise power.
    Higher PSNR indicates better image quality.
    
    Args:
        img1 (np.ndarray or torch.Tensor): First image
        img2 (np.ndarray or torch.Tensor): Second image
        data_range (float): Dynamic range of the images (default: 1.0 for [0,1])
    
    Returns:
        float: PSNR value in dB
    
    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)
        where MAX is the maximum possible pixel value
    """
    # Convert torch tensors to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)
    
    return psnr


def calculate_ssim(img1, img2, data_range=1.0, multichannel=False):
    """
    Calculate Structural Similarity Index (SSIM)
    
    SSIM measures the structural similarity between two images.
    It considers luminance, contrast, and structure.
    SSIM ranges from -1 to 1, where 1 indicates perfect similarity.
    
    Args:
        img1 (np.ndarray or torch.Tensor): First image
        img2 (np.ndarray or torch.Tensor): Second image
        data_range (float): Dynamic range of the images (default: 1.0)
        multichannel (bool): Whether images have multiple channels
    
    Returns:
        float: SSIM value [0, 1]
    
    Formula:
        SSIM(x,y) = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
        where μ is mean, σ is variance, σxy is covariance
    """
    # Convert torch tensors to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
    
    # Handle batch dimension
    if len(img1.shape) == 4:  # [B, C, H, W]
        img1 = img1.squeeze()
        img2 = img2.squeeze()
    
    # Calculate SSIM
    if len(img1.shape) == 3 and img1.shape[0] in [1, 3]:  # [C, H, W]
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        multichannel = True
    
    ssim = structural_similarity(
        img1, img2, 
        data_range=data_range,
        channel_axis=2 if multichannel else None
    )
    
    return ssim


def calculate_mae(img1, img2):
    """
    Calculate Mean Absolute Error (MAE)
    
    MAE measures the average absolute difference between pixels.
    Lower MAE indicates better similarity.
    
    Args:
        img1 (np.ndarray or torch.Tensor): First image
        img2 (np.ndarray or torch.Tensor): Second image
    
    Returns:
        float: MAE value
    
    Formula:
        MAE = (1/N) * Σ|img1 - img2|
    """
    # Convert torch tensors to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
    
    # Calculate MAE
    mae = np.mean(np.abs(img1 - img2))
    
    return mae


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE)
    
    MSE measures the average squared difference between pixels.
    Lower MSE indicates better similarity.
    
    Args:
        img1 (np.ndarray or torch.Tensor): First image
        img2 (np.ndarray or torch.Tensor): Second image
    
    Returns:
        float: MSE value
    
    Formula:
        MSE = (1/N) * Σ(img1 - img2)²
    """
    # Convert torch tensors to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape: {img1.shape} vs {img2.shape}")
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    return mse


def evaluate_batch(clean_images, denoised_images, data_range=1.0):
    """
    Evaluate a batch of images using multiple metrics
    
    Args:
        clean_images (torch.Tensor): Clean reference images [B, C, H, W]
        denoised_images (torch.Tensor): Denoised images [B, C, H, W]
        data_range (float): Dynamic range of images
    
    Returns:
        dict: Dictionary containing average PSNR, SSIM, MAE, MSE
    
    Example:
        >>> metrics = evaluate_batch(clean_batch, denoised_batch)
        >>> print(f"PSNR: {metrics['psnr']:.2f} dB")
        >>> print(f"SSIM: {metrics['ssim']:.4f}")
    """
    batch_size = clean_images.shape[0]
    
    psnr_values = []
    ssim_values = []
    mae_values = []
    mse_values = []
    
    for i in range(batch_size):
        clean = clean_images[i]
        denoised = denoised_images[i]
        
        # Calculate metrics
        psnr = calculate_psnr(clean, denoised, data_range)
        ssim = calculate_ssim(clean, denoised, data_range)
        mae = calculate_mae(clean, denoised)
        mse = calculate_mse(clean, denoised)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mae_values.append(mae)
        mse_values.append(mse)
    
    # Return average metrics
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'mae': np.mean(mae_values),
        'mse': np.mean(mse_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create test images
    img1 = np.random.rand(256, 256).astype(np.float32)
    img2 = img1 + np.random.normal(0, 0.1, img1.shape).astype(np.float32)
    img2 = np.clip(img2, 0, 1)
    
    # Calculate metrics
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    mae = calculate_mae(img1, img2)
    mse = calculate_mse(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    
    # Test with torch tensors
    print("\nTesting with torch tensors...")
    img1_torch = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    img2_torch = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0)
    
    metrics = evaluate_batch(img1_torch, img2_torch)
    print(f"Batch PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"Batch SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")