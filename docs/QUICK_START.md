# Quick Start Guide

This guide will help you get started with the MRI Image Denoising system quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation

### Step 1: Install Dependencies

```bash
cd /workspace/mri_denoising
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Demo

### Option 1: Using Sample Data

Generate synthetic MRI images for testing:

```bash
cd src
python generate_sample_data.py --output ../data/raw --num_images 50 --size 256
```

### Option 2: Using Your Own Data

Place your MRI images in the `data/raw/` directory. Supported formats:
- PNG, JPG, JPEG
- DICOM (.dcm)
- TIFF, BMP

## Training

### Basic Training

Train DnCNN model with default parameters:

```bash
cd src
python train.py --model dncnn --epochs 50 --batch_size 16 --train_dir ../data/raw
```

### Advanced Training

Train with custom parameters:

```bash
python train.py \
    --model dncnn \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --noise_level 25 \
    --patch_size 128 \
    --save_dir ../models
```

### Training U-Net

```bash
python train.py --model unet --epochs 50 --batch_size 8 --train_dir ../data/raw
```

## Testing

### Denoise Images

Denoise all images in a directory:

```bash
cd src
python test.py \
    --model dncnn \
    --weights ../models/dncnn_epoch_50_best.pth \
    --input ../data/raw \
    --output ../data/results \
    --visualize
```

### Denoise Single Image

```bash
python test.py \
    --model dncnn \
    --weights ../models/dncnn_epoch_50_best.pth \
    --input ../data/raw/image001.png \
    --output ../data/results \
    --visualize
```

### Evaluate Performance

Test with synthetic noise and calculate metrics:

```bash
python test.py \
    --model dncnn \
    --weights ../models/dncnn_epoch_50_best.pth \
    --input ../data/raw \
    --output ../data/results \
    --evaluate \
    --noise_level 25 \
    --visualize
```

## Using Jupyter Notebook

Launch the demo notebook:

```bash
cd notebooks
jupyter notebook demo.ipynb
```

The notebook provides an interactive demonstration of:
- Data loading and visualization
- Model training
- Testing and evaluation
- Performance analysis

## Expected Results

After training for 50 epochs on synthetic data, you should expect:

- **PSNR**: 28-35 dB
- **SSIM**: 0.85-0.95
- **Processing Time**: <0.1s per image (GPU), ~1s per image (CPU)

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image size:

```bash
python train.py --batch_size 8 --patch_size 64
```

### Slow Training

Enable GPU acceleration:

```bash
python train.py --device cuda
```

Or reduce number of workers:

```bash
python train.py --num_workers 2
```

### No Images Found

Make sure your images are in the correct directory and format:

```bash
ls -la data/raw/
```

## Next Steps

1. **Experiment with Parameters**: Try different noise levels, architectures, and hyperparameters
2. **Use Real Data**: Replace synthetic data with real MRI images
3. **Compare Models**: Train both DnCNN and U-Net and compare their performance
4. **Analyze Results**: Use the visualization tools to understand model behavior
5. **Document Findings**: Record your experiments for thesis writing

## Getting Help

- Check the main README.md for detailed documentation
- Review the code comments in `src/` directory
- Refer to the thesis proposal for theoretical background
- Contact your supervisor for guidance

## For Thesis Defense

Before your defense, ensure you have:

1. ✅ Trained models with good performance metrics
2. ✅ Comprehensive test results with visualizations
3. ✅ Comparison with traditional methods
4. ✅ Demo application ready to present
5. ✅ Documentation complete and well-organized
6. ✅ Thesis document following university format

Good luck with your thesis! 🎓