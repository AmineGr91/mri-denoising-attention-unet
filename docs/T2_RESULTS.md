# T2 MRI Denoising Results

## Experimental Setup

- Dataset: IXI T2-weighted MRI
- Noise type: Rician noise
- Noise levels: 5%, 9%, 13%
- Models compared:
  - Standard U-Net
  - Improved U-Net
- Evaluation metrics:
  - PSNR (dB)
  - SSIM
  - MAE
- Test protocol:
  - Fixed random seed: 42
  - Images evaluated: 200 per setting

## Quantitative Results

| Noise Level | Model | PSNR (dB) | SSIM | MAE | Comment |
|---|---|---:|---:|---:|---|
| 5% | Standard U-Net | **36.69 ± 0.41** | 0.9495 ± 0.0064 | **0.008524** | Slightly higher PSNR and lower MAE |
| 5% | Improved U-Net | 36.31 ± 0.40 | **0.9517 ± 0.0076** | 0.008686 | Slightly higher SSIM |
| 9% | Standard U-Net | **33.56 ± 0.38** | 0.9215 ± 0.0089 | 0.011569 | Slightly higher PSNR |
| 9% | Improved U-Net | 33.50 ± 0.40 | **0.9350 ± 0.0100** | **0.011116** | Higher SSIM and lower MAE |
| 13% | Standard U-Net | 31.59 ± 0.38 | 0.9024 ± 0.0109 | 0.013846 | Meets both targets |
| 13% | Improved U-Net | **31.63 ± 0.41** | **0.9178 ± 0.0119** | **0.013214** | Best performance at highest noise level |

## Key Observations

1. Both models performed strongly on T2-weighted MRI at all tested noise levels.
2. At 5% noise, the Standard U-Net showed a slight advantage in PSNR and MAE, while the Improved U-Net achieved slightly better SSIM.
3. At 9% noise, the Standard U-Net again showed a marginally higher PSNR, but the Improved U-Net achieved clearly better SSIM and lower MAE.
4. At 13% noise, the Improved U-Net outperformed the Standard U-Net across PSNR, SSIM, and MAE.
5. All T2 experiments satisfied both target thresholds: PSNR > 30 dB and SSIM > 0.90.

## Thesis-Style Interpretation

The T2-weighted MRI denoising results demonstrate strong performance for both models across all evaluated Rician noise levels. At 5% noise, both the Standard U-Net and the Improved U-Net achieved excellent quantitative performance, with the baseline showing a slight advantage in PSNR and MAE while the proposed model achieved slightly higher SSIM. At 9% noise, the Standard U-Net again showed a marginally higher PSNR, but the Improved U-Net achieved clearly better SSIM and lower MAE, indicating improved structural preservation. At 13% noise, the proposed model outperformed the baseline across PSNR, SSIM, and MAE, while both models still satisfied the target thresholds. Overall, these results indicate that the Improved U-Net provides more robust structural denoising performance, particularly as the noise level increases.

## Table Caption

**Table X.** Comparison of denoising performance between the Standard U-Net and the proposed Improved U-Net on T2-weighted MRI images corrupted with 5%, 9%, and 13% Rician noise. Results are reported as mean ± standard deviation over the test images. The proposed model demonstrated stronger structural preservation and the best overall performance at the highest noise level.
