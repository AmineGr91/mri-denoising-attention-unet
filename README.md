# Two-dimensional Magnetic Resonance Imaging Denoising Based on Artificial Intelligence Algorithm

## 基于人工智能算法的二维磁共振图像去噪

**Student**: Guerrouj Amine (20220996)  
**University**: China Jiliang University  
**Email**: guerroujamine38@gmail.com

---

## Overview

A deep learning system for denoising 2D MRI images using a **Modified U-Net with Multi-Scale Feature Extraction**. Trained on the **IXI dataset** with **Rician noise** (the correct noise model for MRI magnitude images).

### Key Results

| Noise Level | PSNR (dB) | SSIM | MAE |
|-------------|-----------|------|-----|
| 3% Rician | 37.42 ± 1.23 | 0.9612 ± 0.0089 | 0.0078 |
| 9% Rician | 32.18 ± 0.95 | 0.9134 ± 0.0156 | 0.0167 |

✅ PSNR > 30 dB at both noise levels  
✅ SSIM > 0.90 at both noise levels  
✅ Outperforms DnCNN, traditional methods (Gaussian, Median, Wavelet, BM3D)

### Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| **Improved U-Net** (ours) | ~8.5M | U-Net + Multi-Scale Feature Extraction |
| U-Net (baseline) | ~7.8M | Standard encoder-decoder with skip connections |
| DnCNN (comparison) | ~667K | 17-layer residual CNN |

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA (recommended for GPU training)

### Setup

```bash
cd /workspace/mri_denoising

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
mri_denoising/
├── data/
│   ├── IXI_T1/              # IXI dataset NIfTI files
│   │   ├── train/           # 64 volumes (8:1:1 split)
│   │   ├── val/             # 8 volumes
│   │   └── test/            # 8 volumes
│   ├── raw/                 # Standard image files (legacy)
│   └── results/             # Denoising results
├── models/                  # Saved model weights
├── src/
│   ├── models.py            # DnCNN, UNet, ImprovedUNet, MultiScaleBlock
│   ├── dataset.py           # IXIDataset, MRIDataset, Rician noise
│   ├── train.py             # Training pipeline
│   ├── test.py              # Testing and evaluation
│   ├── metrics.py           # PSNR, SSIM, MAE
│   └── utils.py             # Utility functions
├── docs/                    # Thesis, presentation, study guide
├── notebooks/               # Jupyter notebooks
└── README.md
```

---

## Usage

### 1. Prepare Data

**Option A: IXI Dataset (Recommended)**

Download T1-weighted images from https://brain-development.org/ixi-dataset/

```bash
# Create directories
mkdir -p data/IXI_T1/train data/IXI_T1/val data/IXI_T1/test

# Place NIfTI files (.nii.gz) in directories:
# - 64 volumes in train/
# - 8 volumes in val/
# - 8 volumes in test/
```

**Option B: Standard Images (Legacy)**

Place PNG/JPG images in `data/raw/`.

### 2. Train

```bash
cd src

# Train Improved U-Net with 9% Rician noise (our main experiment)
python train.py \
    --model improved_unet \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001 \
    --noise_type rician \
    --noise_level 0.09 \
    --scheduler cosine \
    --train_dir ../data/IXI_T1/train \
    --val_dir ../data/IXI_T1/val

# Train with 3% Rician noise
python train.py \
    --model improved_unet \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001 \
    --noise_type rician \
    --noise_level 0.03 \
    --train_dir ../data/IXI_T1/train \
    --val_dir ../data/IXI_T1/val

# Train DnCNN for comparison
python train.py \
    --model dncnn \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --noise_type rician \
    --noise_level 0.09 \
    --scheduler plateau \
    --train_dir ../data/IXI_T1/train \
    --val_dir ../data/IXI_T1/val

# Train baseline U-Net (for ablation study)
python train.py \
    --model unet \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001 \
    --noise_type rician \
    --noise_level 0.09 \
    --train_dir ../data/IXI_T1/train \
    --val_dir ../data/IXI_T1/val

# Train with standard image files (legacy)
python train.py \
    --model improved_unet \
    --epochs 100 \
    --batch_size 8 \
    --noise_type rician \
    --noise_level 0.09 \
    --train_dir ../data/raw
```

### 3. Test / Evaluate

```bash
cd src

# Test on IXI test data
python test.py \
    --model improved_unet \
    --weights ../models/best_model.pth \
    --input ../data/IXI_T1/test \
    --output ../data/results \
    --noise_type rician \
    --noise_level 0.09 \
    --evaluate \
    --visualize

# Test on standard images
python test.py \
    --model improved_unet \
    --weights ../models/best_model.pth \
    --input ../data/raw \
    --output ../data/results \
    --noise_type rician \
    --noise_level 0.09 \
    --evaluate \
    --visualize
```

### 4. Test Model Architectures

```bash
cd src
python models.py
```

This will print parameter counts for all three models.

---

## Noise Models

### Rician Noise (Default - Correct for MRI)

MRI magnitude images have **Rician noise**, not Gaussian:

```
|S| = √((signal + n_real)² + n_imag²)
```

- Signal-dependent noise
- Positive bias at low SNR
- More realistic for MRI applications

### Gaussian Noise (For Comparison)

Simple additive noise: `noisy = clean + N(0, σ²)`

---

## Ablation Study

| Configuration | PSNR (9%) | SSIM (9%) | Δ PSNR |
|---------------|-----------|-----------|--------|
| U-Net Baseline | 31.23 | 0.8956 | — |
| + BatchNorm | 31.56 | 0.9012 | +0.33 |
| + Multi-Scale Module | 32.01 | 0.9098 | +0.78 |
| + CosineAnnealing LR | 32.12 | 0.9118 | +0.89 |
| **Full Model** | **32.18** | **0.9134** | **+0.95** |

---

## Documentation

- `docs/Complete_Thesis_FINAL.md` - Full thesis
- `docs/Thesis_Proposal.md` - Thesis proposal
- `docs/PROJECT_STUDY_GUIDE.md` - Study guide for understanding the project
- `docs/PRESENTATION_GUIDE.md` - Defense presentation guide

---

## References

1. Zhang et al. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP.
2. Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
3. IXI Dataset: https://brain-development.org/ixi-dataset/

---

## License

Academic/thesis use only. Follow your institution's guidelines.