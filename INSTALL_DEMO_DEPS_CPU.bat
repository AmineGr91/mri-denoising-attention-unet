@echo off

cd /d %~dp0

echo Installing demo dependencies for MRI Denoising Demo...

echo This may take several minutes, especially PyTorch.

python -m pip install --upgrade pip

python -m pip install streamlit nibabel numpy opencv-python scikit-image matplotlib Pillow tqdm scipy

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.

echo Installation finished. Now run START_DEMO.bat

pause

