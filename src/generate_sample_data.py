"""
Generate Sample MRI Images for Testing

This script generates synthetic MRI-like images for testing the denoising system
when real MRI data is not available.

Usage:
    python generate_sample_data.py --output ../data/raw --num_images 50

Author: Thesis Project
Date: 2025
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2


def generate_brain_like_image(size=512):
    """
    Generate a synthetic brain-like MRI image
    
    Args:
        size (int): Image size
    
    Returns:
        np.ndarray: Synthetic MRI image
    """
    # Create base image
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add elliptical brain structure
    center_y, center_x = size // 2, size // 2
    
    # Outer skull
    cv2.ellipse(image, (center_x, center_y), (size // 3, size // 2 - 20), 
                0, 0, 360, 0.3, -1)
    
    # Brain tissue
    cv2.ellipse(image, (center_x, center_y), (size // 3 - 20, size // 2 - 40), 
                0, 0, 360, 0.6, -1)
    
    # Add some internal structures
    # Ventricles
    cv2.ellipse(image, (center_x - 30, center_y), (15, 25), 0, 0, 360, 0.2, -1)
    cv2.ellipse(image, (center_x + 30, center_y), (15, 25), 0, 0, 360, 0.2, -1)
    
    # Add texture
    texture = np.random.rand(size, size) * 0.1
    image = image + texture
    
    # Smooth the image
    image = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Add some random bright spots (lesions/features)
    num_spots = np.random.randint(3, 8)
    for _ in range(num_spots):
        spot_x = np.random.randint(center_x - 80, center_x + 80)
        spot_y = np.random.randint(center_y - 100, center_y + 100)
        spot_size = np.random.randint(5, 15)
        cv2.circle(image, (spot_x, spot_y), spot_size, 0.8, -1)
    
    # Normalize
    image = np.clip(image, 0, 1)
    
    return image


def generate_tissue_image(size=512):
    """
    Generate a synthetic tissue MRI image
    
    Args:
        size (int): Image size
    
    Returns:
        np.ndarray: Synthetic MRI image
    """
    # Create base with gradient
    y, x = np.ogrid[:size, :size]
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add circular structures (organs/tissues)
    num_circles = np.random.randint(5, 10)
    for _ in range(num_circles):
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 4, 3 * size // 4)
        radius = np.random.randint(30, 80)
        intensity = np.random.rand() * 0.5 + 0.3
        
        mask = (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2
        image[mask] = intensity
    
    # Add texture
    texture = np.random.rand(size, size) * 0.15
    image = image + texture
    
    # Smooth
    image = cv2.GaussianBlur(image, (21, 21), 0)
    
    # Normalize
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    return image


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Sample MRI Images')
    
    parser.add_argument('--output', type=str, default='../data/raw',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images to generate')
    parser.add_argument('--size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--type', type=str, default='both', choices=['brain', 'tissue', 'both'],
                        help='Type of images to generate')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Generating {args.num_images} synthetic MRI images...")
    print(f"Output directory: {args.output}")
    print(f"Image size: {args.size}x{args.size}")
    
    for i in range(args.num_images):
        # Choose image type
        if args.type == 'both':
            img_type = 'brain' if i % 2 == 0 else 'tissue'
        else:
            img_type = args.type
        
        # Generate image
        if img_type == 'brain':
            image = generate_brain_like_image(args.size)
        else:
            image = generate_tissue_image(args.size)
        
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Save image
        filename = f'mri_{img_type}_{i + 1:03d}.png'
        filepath = os.path.join(args.output, filename)
        Image.fromarray(image_uint8).save(filepath)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_images} images")
    
    print(f"\nSuccessfully generated {args.num_images} images!")
    print(f"Images saved to: {args.output}")


if __name__ == "__main__":
    main()