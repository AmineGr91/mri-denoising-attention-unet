"""
Testing and Inference Script for MRI Image Denoising

Supports testing with DnCNN, U-Net, and Improved U-Net models.
Can process NIfTI volumes or standard image files.

Usage:
    # Test Improved U-Net on IXI test data
    python test.py --model improved_unet --weights ../models/best_model.pth \
        --input ../data/IXI_T1/test --output ../data/results \
        --noise_type rician --noise_level 0.09 --evaluate --visualize

    # Test on standard image files
    python test.py --model improved_unet --weights ../models/best_model.pth \
        --input ../data/raw --output ../data/results --evaluate --visualize

Author: Guerrouj Amine (20220996)
Date: 2025
"""

import os
import argparse
import glob
import time
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from models import get_model
from metrics import calculate_psnr, calculate_ssim, calculate_mae
from dataset import add_rician_noise, add_gaussian_noise, normalize_image, load_pickle_volume, extract_slices_from_volume

# Try nibabel
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def parse_args():
    parser = argparse.ArgumentParser(description='Test MRI Denoising Model')
    
    parser.add_argument('--model', type=str, default='improved_unet',
                        choices=['dncnn', 'unet', 'resunet', 'improved_unet'],
                        help='Model architecture')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--num_layers', type=int, default=17,
                        help='Number of layers for DnCNN')
    parser.add_argument('--features', type=int, default=64,
                        help='Number of feature maps for DnCNN')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (NIfTI or standard images)')
    parser.add_argument('--output', type=str, default='../data/results',
                        help='Output directory for results')
    parser.add_argument('--noise_level', type=float, default=0.09,
                        help='Noise level (0.03 for 3%%, 0.09 for 9%%)')
    parser.add_argument('--noise_type', type=str, default='rician',
                        choices=['rician', 'gaussian'],
                        help='Noise type (default: rician)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Save visual comparisons')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate metrics (requires clean images)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_images', type=int, default=50,
                        help='Maximum number of images to process')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible noise')
    
    return parser.parse_args()


def load_standard_image(image_path):
    """Load a standard image file (PNG, JPG, etc.)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = image.astype(np.float32)
    return normalize_image(image)


def add_noise(image, noise_level, noise_type='rician'):
    """Add noise to image."""
    if noise_type == 'rician':
        return add_rician_noise(image, noise_level)
    else:
        return add_gaussian_noise(image, noise_level)


def denoise_image(model, image_tensor, device):
    """Denoise a single image tensor."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        denoised = model(image_tensor)
        denoised = torch.clamp(denoised, 0, 1)
    return denoised


def save_comparison(clean, noisy, denoised, save_path, metrics=None):
    """Save side-by-side comparison image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Clean (Ground Truth)')
        axes[0].axis('off')
        
        axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Noisy Input')
        axes[1].axis('off')
        
        title = 'Denoised (Our Method)'
        if metrics:
            title += f'\nPSNR: {metrics["psnr"]:.2f} dB, SSIM: {metrics["ssim"]:.4f}'
        axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save comparison: {e}")


def process_nifti_directory(args, model, device):
    """Process NIfTI volumes from a directory."""
    nifti_files = sorted(
        glob.glob(os.path.join(args.input, '*.nii.gz')) +
        glob.glob(os.path.join(args.input, '*.nii'))
    )
    
    if len(nifti_files) == 0:
        print(f"No NIfTI files found in {args.input}")
        return
    
    print(f"Found {len(nifti_files)} NIfTI volumes")
    os.makedirs(args.output, exist_ok=True)
    
    all_metrics = []
    image_count = 0
    
    for filepath in nifti_files:
        if image_count >= args.max_images:
            break
        
        try:
            volume = nib.load(filepath).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
        
        # Extract central slices
        total_slices = volume.shape[2]
        mid = total_slices // 2
        slice_indices = range(max(0, mid - 5), min(total_slices, mid + 5))
        
        for si in slice_indices:
            if image_count >= args.max_images:
                break
            
            slice_2d = volume[:, :, si]
            if slice_2d.max() < 1e-8:
                continue
            
            # Resize and normalize
            slice_2d = cv2.resize(slice_2d, (256, 256), interpolation=cv2.INTER_LINEAR)
            clean = normalize_image(slice_2d)
            
            # Add noise
            noisy = add_noise(clean, args.noise_level, args.noise_type)
            
            # Denoise
            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
            denoised_tensor = denoise_image(model, noisy_tensor, device)
            denoised = denoised_tensor.cpu().numpy()[0, 0]
            
            # Metrics
            if args.evaluate:
                psnr = calculate_psnr(clean, denoised)
                ssim = calculate_ssim(clean, denoised)
                mae = calculate_mae(clean, denoised)
                
                metrics = {'psnr': psnr, 'ssim': ssim, 'mae': mae}
                all_metrics.append(metrics)
            
            # Save visualization
            if args.visualize:
                vol_name = os.path.basename(filepath).replace('.nii.gz', '').replace('.nii', '')
                vis_dir = os.path.join(args.output, f'{vol_name}_slice{si:03d}')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, 'Comparison.jpg')
                save_comparison(clean, noisy, denoised, vis_path,
                                metrics if args.evaluate else None)
            
            image_count += 1
    
    # Print summary
    if args.evaluate and len(all_metrics) > 0:
        print_metrics_summary(all_metrics, args)


def process_pkl_directory(args, model, device):
    """Process IXI pickle volumes from a directory."""
    pkl_files = sorted(glob.glob(os.path.join(args.input, "*.pkl")))

    if len(pkl_files) == 0:
        print(f"No .pkl files found in {args.input}")
        return

    print(f"Found {len(pkl_files)} .pkl volumes")
    os.makedirs(args.output, exist_ok=True)

    all_metrics = []
    image_count = 0

    for filepath in tqdm(pkl_files, desc="Processing volumes"):
        if image_count >= args.max_images:
            break

        try:
            volume = load_pickle_volume(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        slices = extract_slices_from_volume(volume, num_slices=30, target_size=256)

        for i, clean in enumerate(slices):
            if image_count >= args.max_images:
                break

            noisy = add_noise(clean, args.noise_level, args.noise_type)

            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
            denoised_tensor = denoise_image(model, noisy_tensor, device)
            denoised = denoised_tensor.cpu().numpy()[0, 0]

            if args.evaluate:
                psnr = calculate_psnr(clean, denoised)
                ssim = calculate_ssim(clean, denoised)
                mae = calculate_mae(clean, denoised)
                metrics = {"psnr": psnr, "ssim": ssim, "mae": mae}
                all_metrics.append(metrics)

            if args.visualize:
                vol_name = os.path.splitext(os.path.basename(filepath))[0]
                vis_dir = os.path.join(args.output, vol_name)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"slice_{i:03d}_comparison.jpg")
                save_comparison(clean, noisy, denoised, vis_path,
                                metrics if args.evaluate else None)

            image_count += 1

    if args.evaluate and len(all_metrics) > 0:
        print_metrics_summary(all_metrics, args)


def process_standard_directory(args, model, device):
    """Process standard image files from a directory."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input, ext.upper())))
    
    if len(image_paths) == 0:
        print(f"No images found in {args.input}")
        return
    
    image_paths = sorted(image_paths)[:args.max_images]
    print(f"Processing {len(image_paths)} images")
    os.makedirs(args.output, exist_ok=True)
    
    all_metrics = []
    
    for idx, image_path in enumerate(tqdm(image_paths, desc='Processing')):
        try:
            clean = load_standard_image(image_path)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
        
        # Add noise
        np.random.seed(args.seed + idx)
        noisy = add_noise(clean, args.noise_level, args.noise_type)
        
        # Denoise
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)
        denoised_tensor = denoise_image(model, noisy_tensor, device)
        denoised = denoised_tensor.cpu().numpy()[0, 0]
        
        # Metrics
        if args.evaluate:
            psnr = calculate_psnr(clean, denoised)
            ssim = calculate_ssim(clean, denoised)
            mae = calculate_mae(clean, denoised)
            
            metrics = {'image': os.path.basename(image_path),
                       'psnr': psnr, 'ssim': ssim, 'mae': mae}
            all_metrics.append(metrics)
        
        # Save denoised image
        name = os.path.splitext(os.path.basename(image_path))[0]
        denoised_uint8 = (denoised * 255).astype(np.uint8)
        out_dir = os.path.join(args.output, name, 'images')
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, 'Denoising.jpg'), denoised_uint8)
        
        # Save comparison
        if args.visualize:
            vis_path = os.path.join(out_dir, 'Comparison.jpg')
            save_comparison(clean, noisy, denoised, vis_path,
                            metrics if args.evaluate else None)
    
    if args.evaluate and len(all_metrics) > 0:
        print_metrics_summary(all_metrics, args)


def print_metrics_summary(all_metrics, args):
    """Print and save metrics summary."""
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    std_psnr = np.std([m['psnr'] for m in all_metrics])
    std_ssim = np.std([m['ssim'] for m in all_metrics])
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS ({args.model.upper()})")
    print(f"Noise: {args.noise_type} at {args.noise_level*100:.0f}%")
    print(f"{'='*80}")
    print(f"  PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  MAE:  {avg_mae:.6f}")
    print(f"  Images evaluated: {len(all_metrics)}")
    print(f"{'='*80}")
    
    # Check targets
    print(f"\nTarget Check:")
    print(f"  PSNR > 30 dB: {'✅ PASS' if avg_psnr > 30 else '❌ FAIL'} ({avg_psnr:.2f})")
    print(f"  SSIM > 0.90:  {'✅ PASS' if avg_ssim > 0.90 else '❌ FAIL'} ({avg_ssim:.4f})")
    
    # Save to file
    metrics_file = os.path.join(args.output, 'metrics_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Noise: {args.noise_type} at {args.noise_level*100:.0f}%\n")
        f.write(f"PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"MAE: {avg_mae:.6f}\n")
        f.write(f"Images: {len(all_metrics)}\n")
    print(f"\nMetrics saved to: {metrics_file}")


def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading {args.model.upper()} model...")
    if args.model == 'dncnn':
        model = get_model('dncnn', num_layers=args.num_layers, features=args.features)
    elif args.model == 'unet':
        model = get_model('unet')
    elif args.model == 'resunet':
        model = get_model('resunet')
    else:
        model = get_model('improved_unet')
    
    # Load weights
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    # Detect input type and process
    pkl_files = glob.glob(os.path.join(args.input, '*.pkl'))
    nifti_files = glob.glob(os.path.join(args.input, '*.nii.gz')) + \
                  glob.glob(os.path.join(args.input, '*.nii'))
    
    start_time = time.time()
    
    if len(pkl_files) > 0:
        print(f"\nProcessing IXI pickle volumes from {args.input}")
        process_pkl_directory(args, model, device)
    elif len(nifti_files) > 0 and HAS_NIBABEL:
        print(f"\nProcessing NIfTI volumes from {args.input}")
        process_nifti_directory(args, model, device)
    else:
        print(f"\nProcessing standard images from {args.input}")
        process_standard_directory(args, model, device)
    
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()