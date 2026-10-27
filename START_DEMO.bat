@echo off

cd /d %~dp0

echo Starting MRI Denoising Streamlit Demo...

echo Project folder: %CD%

python -m streamlit run app\demo_app.py

pause

