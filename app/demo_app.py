import sys
import tempfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import add_rician_noise, extract_slices_from_volume
from metrics import calculate_mae, calculate_psnr, calculate_ssim
from models import get_model

st.set_page_config(page_title='MRI Denoising Demo', layout='wide')


SKIPPED_PRESETS = []


def infer_architecture_from_name(folder_name: str):
    name = folder_name.lower()
    if 'improved' in name:
        return 'improved_unet'
    if 'resunet' in name:
        return 'resunet'
    return 'unet'


def discover_model_presets():
    presets = {}
    models_dir = PROJECT_ROOT / 'models'
    if not models_dir.exists():
        return presets

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        best_model = model_dir / 'best_model.pth'
        if not best_model.exists():
            continue

        folder_name = model_dir.name
        if 'sanity' in folder_name.lower():
            SKIPPED_PRESETS.append(folder_name)
            continue

        pretty_name = folder_name.replace('_', ' ')
        presets[pretty_name] = {
            'model_name': infer_architecture_from_name(folder_name),
            'weights': best_model,
            'label': folder_name,
        }
    return presets


MODEL_PRESETS = discover_model_presets()
DEFAULT_PRESET_NAMES = list(MODEL_PRESETS.keys())


@st.cache_resource
def load_model(model_name: str, weights_path: str, device_name: str):
    device = torch.device(device_name)
    model = get_model(model_name)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def denoise_slice(model, image_2d: np.ndarray, device_name: str) -> np.ndarray:
    tensor = torch.from_numpy(image_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device_name)
    with torch.no_grad():
        output = model(tensor)
        output = torch.clamp(output, 0, 1)
    return output.squeeze().detach().cpu().numpy().astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def make_strip(images):
    return np.concatenate([to_uint8(img) for img in images], axis=1)


def save_nifti_like(original_nii, denoised_volume: np.ndarray, save_path: str):
    out_nii = nib.Nifti1Image(denoised_volume.astype(np.float32), affine=original_nii.affine, header=original_nii.header)
    nib.save(out_nii, save_path)


def get_model_output(model_info, image_2d, device_name):
    model = load_model(model_info['model_name'], str(model_info['weights']), device_name)
    return denoise_slice(model, image_2d, device_name)


def compute_metrics(reference, prediction):
    return {
        'PSNR': calculate_psnr(reference, prediction),
        'SSIM': calculate_ssim(reference, prediction),
        'MAE': calculate_mae(reference, prediction),
    }


st.markdown(
    """
    <style>
    .hero {
        padding: 1rem 1.25rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #164e63 100%);
        color: #e2e8f0;
        border: 1px solid rgba(148,163,184,0.25);
        margin-bottom: 1rem;
        box-shadow: 0 16px 40px rgba(2, 6, 23, 0.22);
    }
    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    </style>
    <div class="hero">
        <h1 style="margin-bottom:0.2rem;">MRI Denoising Demo</h1>
        <p style="margin:0; font-size:1.05rem;">Upload a noisy or clean MRI volume, run a trained model, compare outputs, and export denoised results for your thesis demonstration.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PRESETS:
    st.error('No trained checkpoints with `best_model.pth` were found in the `models/` folder.')
    st.stop()

if SKIPPED_PRESETS:
    st.info('Skipped helper checkpoints: ' + ', '.join(SKIPPED_PRESETS))

with st.sidebar:
    st.header('Demo Settings')
    mode = st.radio('Input type', ['Already noisy MRI', 'Clean MRI + simulate noise'])
    noise_level = st.slider('Noise level for simulation mode', min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    compare_models = st.checkbox('Compare two models on the same scan', value=True)
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.caption(f'Running on: {device_name}')

    primary_model_name = st.selectbox('Primary model', DEFAULT_PRESET_NAMES, index=0)
    primary_info = MODEL_PRESETS[primary_model_name]

    secondary_info = None
    secondary_model_name = None
    if compare_models:
        secondary_options = [name for name in DEFAULT_PRESET_NAMES if name != primary_model_name]
        if secondary_options:
            secondary_model_name = st.selectbox('Comparison model', secondary_options, index=0)
            secondary_info = MODEL_PRESETS[secondary_model_name]

uploaded_file = st.file_uploader('Upload MRI volume (.nii or .nii.gz)', type=['nii', 'gz'])

if uploaded_file is None:
    st.info('Upload a NIfTI MRI volume to start the demo.')
    st.stop()

suffix = '.nii.gz' if uploaded_file.name.endswith('.nii.gz') else Path(uploaded_file.name).suffix or '.nii'
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_file.getbuffer())
    tmp_path = tmp.name

try:
    nii = nib.load(tmp_path)
    volume = nii.get_fdata().astype(np.float32)
    slices = extract_slices_from_volume(volume, num_slices=30, target_size=256)
finally:
    Path(tmp_path).unlink(missing_ok=True)

if len(slices) == 0:
    st.error('No usable slices were found in this volume.')
    st.stop()

slice_idx = st.slider('Slice index from extracted central slices', 0, len(slices) - 1, len(slices) // 2)
base_slice = slices[slice_idx]

if mode == 'Clean MRI + simulate noise':
    np.random.seed(42)
    model_input = add_rician_noise(base_slice, noise_level)
    reference_slice = base_slice
    reference_title = 'Clean reference slice'
else:
    model_input = base_slice
    reference_slice = base_slice
    reference_title = 'Uploaded noisy slice'

try:
    primary_output = get_model_output(primary_info, model_input, device_name)
    secondary_output = None
    if secondary_info is not None:
        secondary_output = get_model_output(secondary_info, model_input, device_name)
except RuntimeError as exc:
    st.error(f'Failed to load selected checkpoint: {exc}')
    st.stop()

summary_cols = st.columns(3 if secondary_output is not None else 2)
with summary_cols[0]:
    st.markdown(f"<div class='metric-card'><b>Primary model</b><br>{primary_model_name}</div>", unsafe_allow_html=True)
with summary_cols[1]:
    st.markdown(f"<div class='metric-card'><b>Slice used</b><br>{slice_idx + 1} / {len(slices)}</div>", unsafe_allow_html=True)
if secondary_output is not None:
    with summary_cols[2]:
        st.markdown(f"<div class='metric-card'><b>Comparison model</b><br>{secondary_model_name}</div>", unsafe_allow_html=True)

if mode == 'Clean MRI + simulate noise':
    st.subheader('Quantitative metrics for this slice')
    metric_cols = st.columns(2 if secondary_output is not None else 1)

    primary_metrics = compute_metrics(reference_slice, primary_output)
    with metric_cols[0]:
        st.markdown(f"**{primary_model_name}**")
        st.write(f"PSNR: {primary_metrics['PSNR']:.2f} dB")
        st.write(f"SSIM: {primary_metrics['SSIM']:.4f}")
        st.write(f"MAE: {primary_metrics['MAE']:.6f}")

    if secondary_output is not None:
        secondary_metrics = compute_metrics(reference_slice, secondary_output)
        with metric_cols[1]:
            st.markdown(f"**{secondary_model_name}**")
            st.write(f"PSNR: {secondary_metrics['PSNR']:.2f} dB")
            st.write(f"SSIM: {secondary_metrics['SSIM']:.4f}")
            st.write(f"MAE: {secondary_metrics['MAE']:.6f}")

if secondary_output is not None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(reference_slice, caption=reference_title, clamp=True)
    with c2:
        st.image(model_input, caption='Input to model', clamp=True)
    with c3:
        st.image(primary_output, caption=f'Primary output\n{primary_model_name}', clamp=True)
    with c4:
        st.image(secondary_output, caption=f'Comparison output\n{secondary_model_name}', clamp=True)

    strip = make_strip([reference_slice, model_input, primary_output, secondary_output])
    st.subheader('Side-by-side comparison')
    st.image(strip, caption='Reference / Input / Primary output / Comparison output')
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(reference_slice, caption=reference_title, clamp=True)
    with c2:
        st.image(model_input, caption='Input to model', clamp=True)
    with c3:
        st.image(primary_output, caption=f'Denoised output\n{primary_model_name}', clamp=True)

    strip = make_strip([reference_slice, model_input, primary_output])
    st.subheader('Side-by-side comparison')
    st.image(strip, caption='Reference / Input / Denoised output')

png_bytes = cv2.imencode('.png', strip)[1].tobytes()
st.download_button('Download comparison image', data=png_bytes, file_name='comparison.png', mime='image/png')

if st.button('Export denoised NIfTI using primary model'):
    denoised_slices = []
    for s in slices:
        if mode == 'Clean MRI + simulate noise':
            noisy_s = add_rician_noise(s, noise_level)
        else:
            noisy_s = s
        denoised_slices.append(get_model_output(primary_info, noisy_s, device_name))

    denoised_volume = np.stack(denoised_slices, axis=-1)
    out_dir = PROJECT_ROOT / 'results' / 'demo_exports'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"denoised_{Path(uploaded_file.name).stem}_{primary_info['label']}.nii.gz"
    save_nifti_like(nii, denoised_volume, str(out_path))

    st.success(f'Denoised file saved to: {out_path}')
    with open(out_path, 'rb') as f:
        st.download_button('Download denoised NIfTI', data=f.read(), file_name=out_path.name)

st.caption('Tip for defense: use “Already noisy MRI” for a practical demo, or “Clean MRI + simulate noise” to show controlled noise removal with PSNR/SSIM/MAE on screen.')
