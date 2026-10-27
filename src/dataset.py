"""
Dataset and Data Preprocessing Module for MRI Image Denoising

This module handles:
- Loading MRI images from pickle format (.pkl) - preprocessed IXI dataset
- Loading MRI images from NIfTI format (.nii, .nii.gz) - raw IXI dataset
- Loading MRI images from standard formats (PNG, JPG)
- Adding Rician noise (realistic MRI noise model)
- Adding Gaussian noise (for comparison)
- Data augmentation
- Creating PyTorch datasets and dataloaders

Author: Guerrouj Amine (20220996)
Date: 2025
"""

import os
import sys
import glob
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage import io

# Try to import nibabel for NIfTI support
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. NIfTI file support disabled.")
    print("Install with: pip install nibabel")

# Try to import pydicom for DICOM support
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def add_rician_noise(image, noise_level):
    """
    Add Rician noise to a clean MRI image.
    
    Rician noise arises from taking the magnitude of a complex signal
    where both real and imaginary channels have additive Gaussian noise.
    This is the correct noise model for MRI magnitude images.
    
    Args:
        image (np.ndarray): Clean image with values in [0, 1]
        noise_level (float): Noise level as fraction (e.g., 0.03 for 3%, 0.09 for 9%)
    
    Returns:
        np.ndarray: Noisy image with Rician noise, clipped to [0, 1]
    """
    max_val = image.max()
    if max_val < 1e-8:
        return image.copy()
    
    sigma = noise_level * max_val
    
    # Add Gaussian noise to real and imaginary channels independently
    noise_real = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noise_imag = np.random.normal(0, sigma, image.shape).astype(np.float32)
    
    # Rician magnitude: |S| = sqrt((signal + n_real)^2 + n_imag^2)
    noisy = np.sqrt((image + noise_real) ** 2 + noise_imag ** 2)
    
    noisy = np.clip(noisy, 0, 1)
    return noisy.astype(np.float32)


def add_gaussian_noise(image, noise_level):
    """
    Add Gaussian noise to image (for comparison experiments).
    
    Args:
        image (np.ndarray): Clean image with values in [0, 1]
        noise_level (float): Noise standard deviation (0-1 scale)
    
    Returns:
        np.ndarray: Noisy image
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 1)
    return noisy.astype(np.float32)


def normalize_image(img):
    """
    Normalize image to [0, 1] range.
    
    Args:
        img (np.ndarray): Input image
    
    Returns:
        np.ndarray: Normalized image in [0, 1]
    """
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    
    return ((img - img_min) / (img_max - img_min)).astype(np.float32)


def load_pickle_volume(filepath):
    """
    Load a preprocessed MRI volume from pickle format (.pkl).
    
    The preprocessed IXI dataset stores volumes as pickle files.
    Common format: shape (2, D, H, W) where channel 0 = MRI image, channel 1 = label/mask.
    We extract channel 0 (the MRI image data) as a 3D volume.
    
    Args:
        filepath (str): Path to .pkl file
    
    Returns:
        np.ndarray: 3D volume as float32 array, normalized to [0, 1]
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different pickle data formats
    if isinstance(data, np.ndarray):
        volume = data.astype(np.float32)
    elif isinstance(data, dict):
        for key in ['data', 'volume', 'image', 'img', 'T1']:
            if key in data:
                volume = np.array(data[key], dtype=np.float32)
                break
        else:
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    volume = val.astype(np.float32)
                    break
            else:
                raise ValueError(f"Cannot find volume data in pickle file: {filepath}")
    elif isinstance(data, (list, tuple)):
        volume = np.array(data, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported pickle data type: {type(data)} in {filepath}")
    
    # Handle 4D data: shape (C, D, H, W) — take channel 0 (MRI image)
    if volume.ndim == 4:
        volume = volume[0]
    
    # Normalize to [0, 1]
    volume = normalize_image(volume)
    
    return volume


def load_nifti_volume(filepath):
    """
    Load a 3D MRI volume from NIfTI format (.nii or .nii.gz).
    
    Args:
        filepath (str): Path to NIfTI file
    
    Returns:
        np.ndarray: 3D volume as float32 array
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required to load NIfTI files. Install with: pip install nibabel")
    
    nii = nib.load(filepath)
    volume = nii.get_fdata().astype(np.float32)
    return volume


def extract_slices_from_volume(volume, num_slices=100, target_size=256):
    """
    Extract slices from a 3D MRI volume and resize to target size.
    Extracts central slices along the axial axis.
    
    Args:
        volume (np.ndarray): 2D or 3D volume
        num_slices (int): Number of central slices to extract
        target_size (int): Target image size (square)
    
    Returns:
        list: List of 2D numpy arrays, each normalized to [0, 1]
    """
    def _best_slice_axis(vol: np.ndarray) -> int:
        """
        Choose the axis that yields the most 'square' slice plane.

        NIfTI volumes are commonly shaped (H, W, D) while some preprocessed
        arrays are (D, H, W). We pick the axis whose slice has the aspect ratio
        closest to 1 to make slicing consistent across formats.
        """
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got ndim={vol.ndim}")
        best_axis = 0
        best_score = float("inf")
        for axis in (0, 1, 2):
            a, b = [vol.shape[i] for i in range(3) if i != axis]
            if min(a, b) < 1:
                continue
            score = abs(np.log(a / b))  # 0 when perfectly square
            if score < best_score:
                best_score = score
                best_axis = axis
        return best_axis

    def _take_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
        if axis == 0:
            return vol[idx, :, :]
        if axis == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]

    if volume.ndim == 2:
        slice_2d = normalize_image(volume)
        if slice_2d.shape[0] != target_size or slice_2d.shape[1] != target_size:
            slice_2d = cv2.resize(slice_2d, (target_size, target_size),
                                  interpolation=cv2.INTER_LINEAR)
        return [slice_2d]
    
    if volume.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume, got shape {volume.shape}")
    
    slice_axis = _best_slice_axis(volume)
    total_slices = volume.shape[slice_axis]
    num_slices = min(num_slices, total_slices)
    
    start = max(0, (total_slices - num_slices) // 2)
    end = start + num_slices
    
    slices = []
    for i in range(start, end):
        slice_2d = _take_slice(volume, slice_axis, i)
        
        if slice_2d.max() < 1e-8:
            continue
        
        non_zero_ratio = np.count_nonzero(slice_2d > 0.01) / slice_2d.size
        if non_zero_ratio < 0.05:
            continue
        
        if slice_2d.shape[0] != target_size or slice_2d.shape[1] != target_size:
            slice_2d = cv2.resize(slice_2d, (target_size, target_size),
                                  interpolation=cv2.INTER_LINEAR)
        
        slice_2d = normalize_image(slice_2d)
        slices.append(slice_2d)
    
    return slices


# Keep backward compatibility
def extract_central_slices(volume, num_slices=100, target_size=256):
    """Backward compatible wrapper for extract_slices_from_volume."""
    return extract_slices_from_volume(volume, num_slices, target_size)


class IXIDataset(Dataset):
    """
    PyTorch Dataset for IXI MRI Data with Rician Noise.
    
    Loads ALL slices into memory for maximum training speed.
    Safe to use with num_workers=0 (no multiprocessing pickle overhead).
    
    Memory usage: ~40,000 slices x 256x256 x 4 bytes = ~2.5 GB RAM
    This is fine for most machines (the previous MemoryError was caused
    by num_workers>0 trying to pickle this data, not by holding it in RAM).
    
    Args:
        data_dir (str): Directory containing .pkl or .nii.gz files
        noise_level (float): Rician noise level (0.03 for 3%, 0.09 for 9%)
        noise_type (str): 'rician' or 'gaussian'
        num_slices_per_volume (int): Number of central slices per volume
        target_size (int): Target image size (square)
        augment (bool): Whether to apply data augmentation
    """
    
    def __init__(self, data_dir, noise_level=0.09, noise_type='rician',
                 num_slices_per_volume=30, target_size=256, augment=True):
        self.data_dir = data_dir
        self.noise_level = noise_level
        self.noise_type = noise_type.lower()
        self.target_size = target_size
        self.augment = augment
        
        # Load ALL slices into memory for fast access
        self.slices = []
        self._load_all_volumes(num_slices_per_volume)
        
        if len(self.slices) == 0:
            raise ValueError(f"No valid slices found in {data_dir}")
        
        # Stack into a single numpy array for even faster access
        self.slices = np.stack(self.slices, axis=0)  # shape: (N, H, W)
        
        print(f"IXIDataset: Loaded {len(self.slices)} slices from {data_dir}")
        print(f"  Data shape: {self.slices.shape}, Memory: {self.slices.nbytes / 1024**3:.2f} GB")
        print(f"  Noise type: {self.noise_type}, level: {self.noise_level}")
    
    def _load_all_volumes(self, num_slices_per_volume):
        """Load all volumes and extract slices into memory."""
        pkl_files = sorted(glob.glob(os.path.join(self.data_dir, '*.pkl')))
        nifti_files = sorted(
            glob.glob(os.path.join(self.data_dir, '*.nii.gz')) +
            glob.glob(os.path.join(self.data_dir, '*.nii'))
        )
        
        volume_files = []
        file_type = None
        
        if len(pkl_files) > 0:
            volume_files = pkl_files
            file_type = 'pkl'
            print(f"Found {len(pkl_files)} pickle (.pkl) files in {self.data_dir}")
        elif len(nifti_files) > 0:
            volume_files = nifti_files
            file_type = 'nifti'
            print(f"Found {len(nifti_files)} NIfTI files in {self.data_dir}")
        else:
            print(f"Warning: No .pkl or NIfTI files found in {self.data_dir}")
            return
        
        loaded_count = 0
        failed_count = 0
        
        for filepath in volume_files:
            try:
                if file_type == 'pkl':
                    volume = load_pickle_volume(filepath)
                else:
                    volume = load_nifti_volume(filepath)
                
                slices = extract_slices_from_volume(
                    volume, num_slices_per_volume, self.target_size
                )
                self.slices.extend(slices)
                loaded_count += 1
                
                if loaded_count % 50 == 0:
                    print(f"  Loaded {loaded_count}/{len(volume_files)} files, "
                          f"{len(self.slices)} slices so far...")
                    
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"Warning: Failed to load {os.path.basename(filepath)}: {e}")
                elif failed_count == 6:
                    print("  (Suppressing further warnings...)")
        
        print(f"  Successfully loaded {loaded_count}/{len(volume_files)} files "
              f"({failed_count} failed)")
    
    def _add_noise(self, image):
        """Add noise to clean image based on configured noise type."""
        if self.noise_type == 'rician':
            return add_rician_noise(image, self.noise_level)
        else:
            return add_gaussian_noise(image, self.noise_level)
    
    def _augment_image(self, image):
        """Apply random augmentation: flip and rotation."""
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        if np.random.rand() > 0.5:
            image = np.flipud(image)
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        return image.copy()
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            tuple: (noisy_image, clean_image) as torch tensors [1, H, W]
        """
        clean_image = self.slices[idx].copy()
        
        if self.augment:
            clean_image = self._augment_image(clean_image)
        
        noisy_image = self._add_noise(clean_image)
        
        # Convert to torch tensors [C, H, W]
        noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_image).unsqueeze(0)
        
        return noisy_tensor, clean_tensor


class MRIDataset(Dataset):
    """
    PyTorch Dataset for MRI images from standard image formats (PNG, JPG, DICOM).
    
    This is the legacy dataset class for loading individual image files.
    For IXI dataset, use IXIDataset instead.
    """
    
    def __init__(self, image_dir, noise_level=25, noise_type='rician',
                 patch_size=256, augment=True, is_train=True):
        self.image_dir = image_dir
        self.noise_type = noise_type.lower()
        self.patch_size = patch_size
        self.augment = augment and is_train
        self.is_train = is_train
        
        if self.noise_type == 'gaussian' and noise_level > 1:
            self.noise_level = noise_level / 255.0
        else:
            self.noise_level = noise_level
        
        self.image_paths = self._load_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"MRIDataset: Loaded {len(self.image_paths)} images from {image_dir}")
    
    def _load_image_paths(self):
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.dcm']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))
        return sorted(image_paths)
    
    def _load_image(self, path):
        if path.lower().endswith('.dcm') and HAS_PYDICOM:
            dicom = pydicom.dcmread(path)
            image = dicom.pixel_array.astype(np.float32)
        else:
            image = io.imread(path)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.astype(np.float32)
        
        return normalize_image(image)
    
    def _add_noise(self, image):
        if self.noise_type == 'rician':
            return add_rician_noise(image, self.noise_level)
        else:
            return add_gaussian_noise(image, self.noise_level)
    
    def _random_crop(self, image):
        h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size))
            return image
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        return image[top:top + self.patch_size, left:left + self.patch_size]
    
    def _augment_image(self, image):
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        if np.random.rand() > 0.5:
            image = np.flipud(image)
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        return image.copy()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        clean_image = self._load_image(self.image_paths[idx])
        
        if self.is_train:
            clean_image = self._random_crop(clean_image)
        
        if self.augment:
            clean_image = self._augment_image(clean_image)
        
        noisy_image = self._add_noise(clean_image)
        
        noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_image).unsqueeze(0)
        
        return noisy_tensor, clean_tensor


def create_dataloaders(data_dir, val_dir=None, batch_size=8, noise_level=0.09,
                       noise_type='rician', num_workers=0, dataset_type='auto',
                       num_slices_per_volume=30, target_size=256):
    """
    Create training and validation dataloaders.
    
    IMPORTANT: Uses num_workers=0 by default to avoid Windows MemoryError.
    Data is loaded into RAM once at startup for maximum training speed.
    
    Args:
        data_dir (str): Training data directory
        val_dir (str): Validation data directory (optional)
        batch_size (int): Batch size
        noise_level (float): Noise level
        noise_type (str): 'rician' or 'gaussian'
        num_workers (int): Data loading workers (MUST be 0 on Windows)
        dataset_type (str): 'auto', 'ixi', or 'standard'
        num_slices_per_volume (int): Slices per volume
        target_size (int): Target image size
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Force num_workers=0 on Windows to avoid multiprocessing pickle/memory issues
    if sys.platform == 'win32' and num_workers > 0:
        print("Note: Setting num_workers=0 on Windows to avoid multiprocessing memory issues.")
        num_workers = 0
    
    # Auto-detect dataset type
    if dataset_type == 'auto':
        pkl_files = glob.glob(os.path.join(data_dir, '*.pkl'))
        nifti_files = glob.glob(os.path.join(data_dir, '*.nii.gz')) + \
                      glob.glob(os.path.join(data_dir, '*.nii'))
        if len(pkl_files) > 0 or len(nifti_files) > 0:
            dataset_type = 'ixi'
        else:
            dataset_type = 'standard'
    
    print(f"Using dataset type: {dataset_type}")
    
    if dataset_type == 'ixi':
        train_dataset = IXIDataset(
            data_dir, noise_level=noise_level, noise_type=noise_type,
            num_slices_per_volume=num_slices_per_volume,
            target_size=target_size, augment=True
        )
    else:
        train_dataset = MRIDataset(
            data_dir, noise_level=noise_level, noise_type=noise_type,
            patch_size=target_size, augment=True, is_train=True
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = None
    if val_dir is not None:
        if dataset_type == 'ixi':
            val_dataset = IXIDataset(
                val_dir, noise_level=noise_level, noise_type=noise_type,
                num_slices_per_volume=num_slices_per_volume,
                target_size=target_size, augment=False
            )
        else:
            val_dataset = MRIDataset(
                val_dir, noise_level=noise_level, noise_type=noise_type,
                patch_size=target_size, augment=False, is_train=False
            )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Dataset Module...\n")
    
    # Test Rician noise
    print("1. Testing Rician noise generation:")
    clean = np.random.rand(256, 256).astype(np.float32)
    noisy_3 = add_rician_noise(clean, 0.03)
    noisy_9 = add_rician_noise(clean, 0.09)
    print(f"   Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"   Noisy (3%) range: [{noisy_3.min():.3f}, {noisy_3.max():.3f}]")
    print(f"   Noisy (9%) range: [{noisy_9.min():.3f}, {noisy_9.max():.3f}]")
    
    # Test with pickle files (IXI dataset)
    ixi_train_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'IXI_T1', 'Train')
    if os.path.exists(ixi_train_dir):
        pkl_files = glob.glob(os.path.join(ixi_train_dir, '*.pkl'))
        if len(pkl_files) > 0:
            print(f"\n2. Testing IXIDataset with pickle files from {ixi_train_dir}:")
            print(f"   Found {len(pkl_files)} .pkl files")
            try:
                vol = load_pickle_volume(pkl_files[0])
                print(f"   Volume shape after loading: {vol.shape}, dtype: {vol.dtype}")
                print(f"   Value range: [{vol.min():.3f}, {vol.max():.3f}]")
                
                slices = extract_slices_from_volume(vol, num_slices=5, target_size=256)
                print(f"   Extracted {len(slices)} slices (requested 5)")
                if len(slices) > 0:
                    print(f"   Slice shape: {slices[0].shape}")
                    noisy = add_rician_noise(slices[0], 0.09)
                    print(f"   Noisy slice range: [{noisy.min():.3f}, {noisy.max():.3f}]")
                    print("   ✓ Pickle loading + slice extraction + noise OK!")
                    
            except Exception as e:
                print(f"   Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\nDataset module test complete!")