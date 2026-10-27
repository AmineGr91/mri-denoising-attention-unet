# Combined T1 and T2 MRI Denoising Results

## Overview

This document summarizes the final denoising results for T1-weighted and T2-weighted MRI under three Rician noise levels: 5%, 9%, and 13%. Two models were compared:

- Standard U-Net
- Improved U-Net

Evaluation metrics:
- PSNR (dB)
- SSIM
- MAE

Test protocol:
- Fixed random seed: 42
- Images evaluated: 200 per setting

---

## Combined Results Table

| Modality | Noise | Standard U-Net PSNR | Standard U-Net SSIM | Standard U-Net MAE | Improved U-Net PSNR | Improved U-Net SSIM | Improved U-Net MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 5%  | 33.62 ± 0.33 | 0.9442 ± 0.0052 | 0.013878 | **33.70 ± 0.35** | **0.9506 ± 0.0049** | **0.013526** |
| T1 | 9%  | 30.47 ± 0.35 | 0.8990 ± 0.0103 | 0.019432 | **30.61 ± 0.39** | **0.9130 ± 0.0092** | **0.018738** |
| T1 | 13% | 28.60 ± 0.36 | 0.8581 ± 0.0150 | 0.023677 | **28.69 ± 0.40** | **0.8773 ± 0.0131** | **0.023022** |
| T2 | 5%  | **36.69 ± 0.41** | 0.9495 ± 0.0064 | **0.008524** | 36.31 ± 0.40 | **0.9517 ± 0.0076** | 0.008686 |
| T2 | 9%  | **33.56 ± 0.38** | 0.9215 ± 0.0089 | 0.011569 | 33.50 ± 0.40 | **0.9350 ± 0.0100** | **0.011116** |
| T2 | 13% | 31.59 ± 0.38 | 0.9024 ± 0.0109 | 0.013846 | **31.63 ± 0.41** | **0.9178 ± 0.0119** | **0.013214** |

### Notes

- **T1, 5%:** Improved U-Net achieved the best overall performance.
- **T1, 9%:** Improved U-Net met both targets, while Standard U-Net was marginally below the SSIM threshold.
- **T1, 13%:** Both models were below the predefined targets, but Improved U-Net remained better across all metrics.
- **T2, 5%:** Standard U-Net had slightly higher PSNR and lower MAE, while Improved U-Net had slightly higher SSIM.
- **T2, 9%:** Standard U-Net had a small PSNR advantage, while Improved U-Net had higher SSIM and lower MAE.
- **T2, 13%:** Improved U-Net achieved the best overall performance across all three metrics.

---

## Summary

The combined results show that the Improved U-Net achieved the strongest overall structural performance across both MRI modalities. For T1-weighted MRI, the Improved U-Net outperformed the Standard U-Net at all three noise levels, although both models degraded noticeably at 13% noise. For T2-weighted MRI, both models performed strongly across all tested settings; the Standard U-Net showed a slight PSNR advantage at 5% and 9%, while the Improved U-Net consistently achieved higher SSIM and the best overall performance at 13%.

Overall, these findings suggest that the proposed Improved U-Net is more robust to increasing Rician noise, especially when structural preservation is considered together with pixel-wise fidelity.

---

## Short Message for Teacher

I combined the T1 and T2 results into one table for easier comparison. The final results show that for T1, the Improved U-Net performed better than the Standard U-Net at all tested noise levels. For T2, both models performed strongly; the Standard U-Net had a slight PSNR advantage at 5% and 9%, while the Improved U-Net consistently achieved higher SSIM and the best overall performance at 13%. Overall, the proposed model demonstrated stronger structural preservation and better robustness as noise increased.
