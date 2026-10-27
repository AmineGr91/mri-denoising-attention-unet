# Code Explanation Guide for Defense

This guide explains the key parts of your code that teachers might ask about during defense.

---

## 1. Rician Noise Implementation

**File:** `src/dataset.py`

**What it does:** Adds realistic MRI noise to clean images.

**Key code:**

```python
def add_rician_noise(image, noise_level):
    sigma = noise_level * image.max()
    noise_real = np.random.normal(0, sigma, image.shape)
    noise_imag = np.random.normal(0, sigma, image.shape)
    noisy = np.sqrt((image + noise_real) ** 2 + noise_imag ** 2)
    return np.clip(noisy, 0, 1)
```

**If asked to explain:**
> Rician noise arises from taking the magnitude of a complex MRI signal. I add independent Gaussian noise to both real and imaginary channels, then compute the magnitude using the square root of the sum of squares. This is the correct noise model for MRI magnitude images.

---

## 2. Attention Gate

**File:** `src/models.py`

**What it does:** Filters skip connections to focus on relevant regions.

**Key concept:**
- Takes decoder features (g) and encoder features (x)
- Computes attention map showing which spatial regions are important
- Multiplies encoder features by attention map
- Result: suppresses background, emphasizes brain tissue

**If asked to explain:**
> The attention gate learns which spatial regions in the skip connection are relevant for denoising. It uses the decoder features as a gating signal to compute an attention map, then multiplies the encoder features by this map. This suppresses irrelevant background regions and focuses on the brain tissue, which is especially helpful at high noise levels.

**Key formula:**
```
attention_map = sigmoid(W_g(decoder) + W_x(encoder))
output = encoder * attention_map
```

---

## 3. Multi-Scale Block

**File:** `src/models.py`

**What it does:** Captures features at multiple spatial scales simultaneously.

**Architecture:**
- 4 parallel branches:
  - 1×1 conv (point-wise features)
  - 3×3 conv (local features)
  - 5×5 conv (larger context)
  - Global average pooling (global context)
- Concatenate all branches
- Fuse with 1×1 conv
- Apply channel attention (SE block)
- Add residual connection

**If asked to explain:**
> The multi-scale block processes features through four parallel branches with different receptive fields. This captures noise patterns and anatomical structures at multiple spatial scales simultaneously. The residual connection ensures the block never hurts performance—if the multi-scale features aren't helpful, the network can bypass them.

---

## 4. Combined MSE + SSIM Loss

**File:** `src/train.py`

**What it does:** Optimizes for both pixel accuracy and structural similarity.

**Key code:**

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
    
    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.ssim(pred, target)
```

**If asked to explain:**
> The combined loss directly optimizes for what we measure. MSE ensures pixel-level accuracy, while SSIM preserves structural similarity. Using alpha=0.5 gives equal weight to both objectives. This is why the Improved U-Net achieves higher SSIM than the baseline, which only uses MSE loss.

---

## 5. SSIM Calculation

**File:** `src/train.py` (loss) and `src/metrics.py` (evaluation)

**What it does:** Measures structural similarity between images.

**Key concept:**
- Compares luminance, contrast, and structure
- Uses sliding window (11×11 Gaussian)
- Returns value between 0 and 1 (1 = identical)

**Formula:**
```
SSIM = (2*μ1*μ2 + C1)(2*σ12 + C2) / ((μ1² + μ2² + C1)(σ1² + σ2² + C2))
```

**If asked to explain:**
> SSIM measures structural similarity by comparing local luminance, contrast, and structure using a sliding Gaussian window. Unlike PSNR which only measures pixel-wise error, SSIM better reflects how humans perceive image quality and whether anatomical structures are preserved.

---

## 6. Volume-Level Data Split

**File:** `src/dataset.py`

**What it does:** Splits data at patient/volume level, not slice level.

**If asked to explain:**
> I split the dataset at the volume level to prevent data leakage. If I split at the slice level, slices from the same patient could appear in both training and test sets, allowing the model to memorize patient-specific anatomy rather than learning general denoising. Volume-level splitting ensures the test set contains patients the model has never seen during training.

---

## 7. Training Loop Structure

**File:** `src/train.py`

**Key components:**

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(noisy_input)
        loss = criterion(output, clean_target)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(noisy_input)
            metrics = evaluate(output, clean_target)
    
    # Learning rate scheduling
    scheduler.step()
    
    # Early stopping check
    if val_ssim > best_ssim:
        save_checkpoint()
```

**If asked to explain:**
> The training loop alternates between training and validation phases. During training, I compute the loss, backpropagate gradients, and update weights. During validation, I evaluate metrics without updating weights. I use cosine annealing to gradually reduce the learning rate, and early stopping based on validation SSIM to prevent overfitting.

---

## 8. Model Architecture Comparison

**Standard U-Net:**
- Encoder-decoder with skip connections
- MSE loss
- No attention mechanisms

**Improved U-Net:**
- Same encoder-decoder structure
- Attention gates on all 4 skip connections
- Multi-scale blocks at enc3, enc4, bottleneck
- Channel attention (SE blocks)
- Combined MSE + SSIM loss

**If asked why these specific improvements:**
> I chose attention gates because they let the model focus on relevant spatial regions. Multi-scale blocks capture features at different scales, which is important because MRI structures range from fine cortical edges to broad tissue regions. Channel attention helps the model emphasize informative feature channels. The combined loss directly optimizes for structural similarity, which is what we care about in medical imaging.

---

## 9. Metrics Calculation

**PSNR:**
```python
mse = np.mean((img1 - img2) ** 2)
psnr = 10 * np.log10(1.0 / mse)
```

**SSIM:**
- Uses scikit-image implementation
- Sliding window comparison
- Returns structural similarity score

**MAE:**
```python
mae = np.mean(np.abs(img1 - img2))
```

**If asked why these metrics:**
> PSNR measures pixel-level reconstruction accuracy. SSIM measures structural preservation, which is more relevant for medical images. MAE provides an intuitive measure of average pixel error. Together, these three metrics give a comprehensive evaluation of denoising quality.

---

## 10. Data Augmentation

**File:** `src/dataset.py`

**Augmentations used:**
- Random horizontal flips
- Random vertical flips
- Random rotations (90° multiples)

**If asked why these specific augmentations:**
> I used only geometric augmentations that preserve MRI image characteristics. Horizontal and vertical flips are valid because brain anatomy is roughly symmetric. 90-degree rotations are safe because they don't introduce interpolation artifacts. I avoided intensity-based augmentations because they could interfere with the noise simulation.

---

## Common Code Questions & Answers

### Q: Why residual learning in U-Net?

**Answer:**
> The U-Net predicts the noise residual, not the clean image directly. The final output is `input - predicted_noise`. This is easier to learn because the network only needs to model the noise, not the entire image structure.

### Q: Why batch normalization?

**Answer:**
> Batch normalization stabilizes training by normalizing activations. It allows higher learning rates and acts as a regularizer. It's standard in modern CNN architectures for medical imaging.

### Q: Why Adam optimizer?

**Answer:**
> Adam adapts the learning rate for each parameter and includes momentum. It's robust and works well for medical image tasks without extensive hyperparameter tuning. It's the standard choice for U-Net training.

### Q: Why cosine annealing scheduler?

**Answer:**
> Cosine annealing gradually reduces the learning rate from 1e-4 to 1e-7 over training. This helps the model converge to a better minimum. The smooth decay prevents sudden jumps in the loss landscape.

### Q: Why 50 epochs?

**Answer:**
> I monitored validation SSIM during training. The models typically converged within 30-40 epochs, so 50 epochs with early stopping ensures full convergence without wasting computation. I used early stopping to prevent overfitting.

### Q: Why batch size 16?

**Answer:**
> Batch size 16 balances GPU memory usage and training stability. Larger batches would exceed GPU memory for 256×256 images. Smaller batches would make training unstable. 16 is a standard choice for medical image segmentation and denoising.

---

## If Asked to Show Code During Defense

**Be ready to navigate to:**
1. `src/models.py` - Show AttentionGate or MultiScaleBlock
2. `src/train.py` - Show CombinedLoss
3. `src/dataset.py` - Show add_rician_noise
4. `src/metrics.py` - Show PSNR/SSIM calculation

**How to explain code quickly:**
1. State what the function/class does (one sentence)
2. Explain the key idea or formula
3. Mention why it's important for your thesis

---

## Key Points to Remember

1. **Attention gates** = spatial filtering of skip connections
2. **Multi-scale blocks** = parallel convolutions with different kernel sizes
3. **Combined loss** = MSE for pixels + SSIM for structure
4. **Rician noise** = magnitude of complex signal with Gaussian noise
5. **Volume-level split** = prevents data leakage
6. **Residual learning** = predict noise, not clean image

---

**You understand your code. Stay confident and explain clearly.**
