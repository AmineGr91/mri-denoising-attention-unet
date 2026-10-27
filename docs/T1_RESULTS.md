# T1 MRI Denoising Results

## Experimental Setup

- Dataset: IXI T1-weighted MRI
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
| 5% | Standard U-Net | 33.62 ± 0.33 | 0.9442 ± 0.0052 | 0.013878 | Meets both targets |
| 5% | Improved U-Net | **33.70 ± 0.35** | **0.9506 ± 0.0049** | **0.013526** | Best performance at this noise level |
| 9% | Standard U-Net | 30.47 ± 0.35 | 0.8990 ± 0.0103 | 0.019432 | Marginally below SSIM threshold |
| 9% | Improved U-Net | **30.61 ± 0.39** | **0.9130 ± 0.0092** | **0.018738** | Meets both targets |
| 13% | Standard U-Net | 28.60 ± 0.36 | 0.8581 ± 0.0150 | 0.023677 | Below both targets at highest noise level |
| 13% | Improved U-Net | **28.69 ± 0.40** | **0.8773 ± 0.0131** | **0.023022** | Best performance at 13%, but below targets |

## Key Observations

1. The Improved U-Net outperformed the Standard U-Net at all tested noise levels.
2. At 5% noise, both models achieved strong denoising performance, with the Improved U-Net showing slightly better PSNR, SSIM, and MAE.
3. At 9% noise, the advantage of the Improved U-Net became more noticeable, and it was the only model that satisfied both quality targets.
4. At 13% noise, both models showed expected performance degradation due to the increased difficulty of the denoising task; however, the Improved U-Net still achieved the best overall results.

## Thesis-Style Interpretation

The experimental results show that the proposed Improved U-Net consistently provides better denoising performance than the standard U-Net across all tested Rician noise levels. At 5% noise, both models achieved strong performance, with the proposed model providing slightly higher PSNR and SSIM and lower MAE. At 9% noise, the performance gap became more noticeable, and the proposed model was the only one to satisfy both the PSNR and SSIM quality targets. At 13% noise, both models experienced the expected reduction in performance due to the increased difficulty of the denoising task; however, the proposed model still achieved the best overall results. These findings indicate that the Improved U-Net offers greater robustness as the noise level increases.

## Table Caption

**Table X.** Comparison of denoising performance between the Standard U-Net and the proposed Improved U-Net on T1-weighted MRI images corrupted with 5%, 9%, and 13% Rician noise. Results are reported as mean ± standard deviation over the test images. The proposed model consistently outperformed the baseline at all noise levels.
