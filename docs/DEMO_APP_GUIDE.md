# MRI Denoising Demo — Beginner Guide

This guide explains how to run the demo app from scratch, even if you are a beginner.

## What this app does

The demo app lets you:
- upload an MRI file in `.nii` or `.nii.gz` format
- choose a trained denoising model
- view the noisy input and denoised output
- compare two models on the same MRI slice
- export a denoised NIfTI file
- download a comparison image for your thesis or defense slides

---

## Before you start

Make sure your project folder contains these folders/files:

- `src/`
- `models/`
- `app/demo_app.py`
- `requirements.txt`

The app automatically looks for trained model files named `best_model.pth` inside the `models/` subfolders.

---

## Step 1 — Open a terminal in the project folder

Example:

```bash
cd d:\Thesis\mri_denoising
```

---

## Step 2 — Install the required packages

Run:

```bash
pip install -r requirements.txt
```

If you already installed them before, it is okay to run this again.

---

## Step 3 — Start the demo app

Run:

```bash
streamlit run app/demo_app.py
```

After a few seconds, Streamlit will show a local URL in the terminal, usually something like:

```bash
Local URL: http://localhost:8501
```

Open that link in your browser.

---

## Step 4 — Use the app

### 4.1 Upload an MRI file
Click **Upload MRI volume** and select a file:
- `.nii`
- `.nii.gz`

### 4.2 Choose the input type
In the left sidebar, choose one of these:

- **Already noisy MRI**
  - use this if your uploaded MRI is already noisy
- **Clean MRI + simulate noise**
  - use this if you want the app to add synthetic Rician noise first

### 4.3 Choose the model
In the sidebar, select:
- a **Primary model**
- and optionally a **Comparison model**

If **Compare two models on the same scan** is checked, the app will show both outputs side by side.

### 4.4 Choose the slice
The app extracts central slices from the uploaded MRI volume.
Use the **Slice index** slider to choose which slice to display.

---

## Step 5 — Understand the output

The app shows:
- the uploaded/reference slice
- the input sent to the model
- the denoised output from the primary model
- optionally the denoised output from a comparison model

This is useful for:
- screenshots
- presentation slides
- demo during defense

---

## Step 6 — Download results

### Download comparison image
Click:

- **Download comparison image**

This saves a PNG image showing the side-by-side outputs.

### Export denoised MRI volume
Click:

- **Export denoised NIfTI using primary model**

Then the app will:
- denoise all extracted slices
- save a NIfTI file in `results/demo_exports/`
- give you a button to download the denoised file

---

## Best settings for defense

For your thesis defense, I recommend:

### Option A — practical demo
- Input type: **Already noisy MRI**
- Compare models: **ON**
- Primary model: **Improved U-Net**
- Comparison model: **U-Net**

This is the best option if you want to show a realistic workflow.

### Option B — controlled demo
- Input type: **Clean MRI + simulate noise**
- Noise level: choose `0.05`, `0.09`, or `0.13`
- Compare models: **ON**

This is the best option if you want to explain your thesis experiment setup.

---

## If the app does not open

Try this:

```bash
python -m streamlit run app/demo_app.py
```

---

## If Streamlit is not installed

Run:

```bash
pip install streamlit
```

or simply run again:

```bash
pip install -r requirements.txt
```

---

## If no models appear in the app

That means the app did not find `best_model.pth` files in your `models/` folder.

Check that your folders look like this:

```text
models/
  improved_t1_5/
    best_model.pth
  improved_t2_13/
    best_model.pth
  unet_t2_9/
    best_model.pth
```

---

## If the app is too slow

Possible reasons:
- using CPU instead of GPU
- very large MRI volume
- comparing two big models at once

If needed, turn off model comparison and run only one model.

---

## What to say in defense

You can say:

> This demo shows the practical use of the trained denoising model. A noisy MRI volume is uploaded, a trained model is selected, and the system produces a denoised output. The interface can also compare two models on the same scan and export denoised results.

---

## Quick start summary

Run these commands:

```bash
cd d:\Thesis\mri_denoising
pip install -r requirements.txt
streamlit run app/demo_app.py
```

Then open the browser link, upload an MRI file, and choose the model.
