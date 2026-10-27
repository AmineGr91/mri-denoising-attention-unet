"""
Diagnostic script to identify the training/testing issue
"""
import torch
import numpy as np
from models import get_model
from dataset import MRIDataset
import os

# Load a sample from the dataset
dataset = MRIDataset('../data/raw', noise_level=25, patch_size=128, is_train=False, augment=False)
noisy, clean = dataset[0]

print("Dataset Check:")
print(f"Noisy shape: {noisy.shape}, range: [{noisy.min():.3f}, {noisy.max():.3f}]")
print(f"Clean shape: {clean.shape}, range: [{clean.min():.3f}, {clean.max():.3f}]")
print(f"Difference (noise): range [{(noisy-clean).min():.3f}, {(noisy-clean).max():.3f}]")

# Load the trained model
model = get_model('dncnn', num_layers=17, features=64)
checkpoint_path = '../models/dncnn_epoch_5_best.pth'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("\nModel loaded successfully")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        noisy_batch = noisy.unsqueeze(0)  # Add batch dimension
        output = model(noisy_batch)
        output = output.squeeze(0)  # Remove batch dimension
    
    print("\nModel Output Check:")
    print(f"Input (noisy) range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Output (denoised) range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Target (clean) range: [{clean.min():.3f}, {clean.max():.3f}]")
    
    # Calculate what the model actually predicted
    # Since model does: out = x - noise_prediction
    # We can recover: noise_prediction = x - out
    noise_pred = noisy - output
    actual_noise = noisy - clean
    
    print("\nNoise Analysis:")
    print(f"Actual noise range: [{actual_noise.min():.3f}, {actual_noise.max():.3f}]")
    print(f"Predicted noise range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
    print(f"Noise prediction mean: {noise_pred.mean():.6f} (should be ~0)")
    print(f"Actual noise mean: {actual_noise.mean():.6f} (should be ~0)")
    
    # Check if output is reasonable
    from metrics import calculate_psnr, calculate_ssim
    
    psnr_noisy = calculate_psnr(clean.numpy()[0], noisy.numpy()[0])
    psnr_denoised = calculate_psnr(clean.numpy()[0], output.numpy()[0])
    ssim_noisy = calculate_ssim(clean.numpy()[0], noisy.numpy()[0])
    ssim_denoised = calculate_ssim(clean.numpy()[0], output.numpy()[0])
    
    print("\nMetrics Comparison:")
    print(f"Noisy image    - PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")
    print(f"Denoised image - PSNR: {psnr_denoised:.2f} dB, SSIM: {ssim_denoised:.4f}")
    
    if psnr_denoised < psnr_noisy:
        print("\n⚠️ WARNING: Model is making images WORSE, not better!")
        print("This indicates a training or model architecture issue.")
else:
    print(f"Model checkpoint not found at {checkpoint_path}")