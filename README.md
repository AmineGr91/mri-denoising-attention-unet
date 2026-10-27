# MRI Denoising with an Attention U-Net

Denoising of 2D brain MRI corrupted with Rician noise, using a U-Net extended with multi-scale feature blocks, attention gates, and squeeze-and-excitation channel attention. Trained and evaluated on the IXI dataset (T1 and T2) and compared against a standard U-Net baseline.

This repository accompanies my B.Eng. graduation thesis, *Two-dimensional Magnetic Resonance Imaging Denoising Based on Artificial Intelligence Algorithm*, China Jiliang University, 2026 (supervisor: Prof. Li Xia).

## Why Rician noise

MRI magnitude images are computed from a complex signal whose real and imaginary parts each carry Gaussian noise. The magnitude of that signal follows a Rician distribution, which depends on the signal and biases low-intensity regions upward. Filters that assume Gaussian noise, such as BM3D, handle it badly. Noise here is simulated as:

```
I_noisy = sqrt((I + n_real)^2 + n_imag^2),   n ~ N(0, sigma^2),   sigma = noise_level * max(I)
```

with noise levels of 5%, 9%, and 13%, and the output clipped to [0, 1].

## Models

| | Standard U-Net (baseline) | Improved U-Net |
|---|---|---|
| Encoder channels | 64, 128, 256, 512 | 64, 128, 256, 512 |
| Bottleneck | 1024 | 1024 + MultiScaleBlock |
| MultiScaleBlock | none | after enc3, enc4, and the bottleneck |
| Skip connections | plain concatenation | attention gate on all 4 |
| Channel attention | none | squeeze-and-excitation inside each block |
| Dropout | none | Dropout2d (p = 0.1) in each block |
| Training loss | MSE | 0.5 * MSE + 0.5 * (1 - SSIM) |
| Output | residual (input minus prediction) | residual (input minus prediction) |

The standard U-Net has about 31.0 million trainable parameters. Both models use batch normalization and Kaiming normal initialization.

**MultiScaleBlock:** four parallel branches (1x1 convolution, 3x3 convolution, 5x5 convolution, and global average pooling with a 1x1 convolution), each producing a quarter of the channels. The outputs are concatenated, fused with a 1x1 convolution, recalibrated by squeeze-and-excitation (reduction ratio 16, minimum 4 channels), and added back to the block input through a residual connection.

**Attention gate:** the gating signal from the decoder and the encoder skip features are projected to an intermediate depth, summed, passed through ReLU, a 1x1 convolution, and a sigmoid, then multiplied with the skip features to suppress background and noisy regions.

## Results

200 test images per setting, fixed seed 42. Mean values; standard deviations are in the thesis.

**T1-weighted**

| Noise | Model | PSNR (dB) | SSIM | MAE |
|---|---|---|---|---|
| 5% | Standard U-Net | 33.62 | 0.9442 | 0.013878 |
| 5% | Improved U-Net | 33.70 | 0.9506 | 0.013526 |
| 9% | Standard U-Net | 30.47 | 0.8990 | 0.019432 |
| 9% | Improved U-Net | 30.61 | 0.9130 | 0.018738 |
| 13% | Standard U-Net | 28.60 | 0.8581 | 0.023677 |
| 13% | Improved U-Net | 28.69 | 0.8773 | 0.023022 |

**T2-weighted**

| Noise | Model | PSNR (dB) | SSIM | MAE |
|---|---|---|---|---|
| 5% | Standard U-Net | 36.69 | 0.9495 | 0.008524 |
| 5% | Improved U-Net | 36.31 | 0.9517 | 0.008686 |
| 9% | Standard U-Net | 33.56 | 0.9215 | 0.011569 |
| 9% | Improved U-Net | 33.50 | 0.9350 | 0.011116 |
| 13% | Standard U-Net | 31.59 | 0.9024 | 0.013846 |
| 13% | Improved U-Net | 31.63 | 0.9178 | 0.013214 |

What the numbers show:

- The Improved U-Net has the higher SSIM in all six conditions, and the gap grows with noise (on T1, from +0.0064 at 5% to +0.0192 at 13%).
- On T2 at 5% and 9% noise, the baseline has slightly better PSNR and MAE. It is trained with MSE alone, which directly optimizes pixel error.
- The two models also differ in training loss, so the gains cannot be credited to the architecture alone. Training the baseline on the combined loss would separate the two effects, and I have not run it yet.

Per-run metric summaries are in `results/*/metrics_summary.txt`. The saved comparison images are not included in this repository because of their size.

## Setup

- Data: IXI T1 and T2 volumes, split at the volume level (80:10:10) so that no subject appears in more than one partition. Central axial slices (30 per volume by default), resized to 256x256 and min-max normalized to [0, 1].
- Augmentation (training only): random horizontal and vertical flips and rotations by multiples of 90 degrees.
- Optimization: Adam, learning rate 1e-4, weight decay 1e-5, cosine annealing to 1e-7 over 50 epochs, batch size 16. Optional mixed precision.
- Metrics: PSNR, SSIM (11x11 Gaussian window, sigma 1.5), and MAE.
- Hardware: one NVIDIA RTX 4090.

## Repository layout

```
src/
  models.py            standard U-Net and Improved U-Net
  unet_baseline.py     baseline U-Net
  resunet.py           residual U-Net variant
  dataset.py           IXI loading, slicing, Rician noise
  train.py             training
  test.py              evaluation
  metrics.py           PSNR, SSIM, MAE
  diagnose.py          diagnostics
  generate_sample_data.py
app/demo_app.py        small interactive demo
notebooks/             demo notebook
results/               per-run metrics_summary.txt
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The IXI dataset is not included. Download the T1 and T2 volumes from <https://brain-development.org/ixi-dataset/> and point the scripts at them.

```bash
python src/train.py --help
python src/test.py --help
```

## Limitations

- Noise is synthetic Rician noise added to already-acquired images. How well it matches real scanner noise has not been checked on raw k-space data.
- The models are 2D and treat each slice independently, so inter-slice information is not used.
- Only quantitative metrics were used; there was no radiologist assessment.
- IXI contains healthy adults from a few London sites, so behavior on pathological anatomy or other scanners is unknown.

## Future work

A combined-loss baseline to isolate the effect of the architecture, 3D or inter-slice attention, self-supervised training, and evaluation on real clinical data.

## Citation and references

If you use this code, please cite the thesis above. Main references: Ronneberger et al. (2015), U-Net; Schlemper et al. (2019), Attention Gated Networks; Hu et al. (2018), Squeeze-and-Excitation Networks; Wang et al. (2004), SSIM; IXI dataset, Brain Development Organization.
