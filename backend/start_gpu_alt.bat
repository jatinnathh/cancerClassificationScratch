@echo off
REM Cancer Classification API - GPU Accelerated
REM Alternative startup for Anaconda installations

echo ========================================
echo  Cancer Classification API Server
echo  GPU-Accelerated (torch_gpu)
echo ========================================
echo.

cd /d "%~dp0"

REM Try to find and initialize conda
echo Initializing Conda...

REM Common Anaconda/Miniconda installation paths
set "CONDA_PATHS=C:\ProgramData\anaconda3 C:\Users\%USERNAME%\anaconda3 C:\Users\%USERNAME%\miniconda3 C:\Anaconda3 C:\Miniconda3"

set "CONDA_FOUND="
for %%P in (%CONDA_PATHS%) do (
    if exist "%%P\Scripts\activate.bat" (
        set "CONDA_FOUND=%%P"
        echo Found Conda at: %%P
        goto :CondaFound
    )
)

:CondaFound
if not defined CONDA_FOUND (
    echo [ERROR] Conda not found in common locations
    echo.
    echo Please install Anaconda or Miniconda from:
    echo https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo Or run this command in Anaconda Prompt:
    echo   cd %~dp0
    echo   conda activate torch_gpu
    echo   python app.py
    echo.
    pause
    exit /b 1
)

REM Initialize conda for this session
call "%CONDA_FOUND%\Scripts\activate.bat" "%CONDA_FOUND%"

REM Activate torch_gpu environment
echo Activating torch_gpu environment...
call conda activate torch_gpu

if errorlevel 1 (
    echo.
    echo [ERROR] torch_gpu environment not found
    echo.
    echo Please create it first by opening Anaconda Prompt and running:
    echo.
    echo   conda create -n torch_gpu python=3.11 -y
    echo   conda activate torch_gpu
    echo   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    echo   pip install flask flask-cors pillow opencv-python scikit-image werkzeug
    echo.
    pause
    exit /b 1
)

echo [OK] torch_gpu environment activated
echo.

REM Check GPU
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Running on CPU')" 2>nul

if errorlevel 1 (
    echo [WARNING] Could not check GPU status
)
echo.

REM Start server
echo Starting Flask API server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python app.py

pause
