@echo off
REM Quick Start Script for Cancer Classification API
REM Uses torch_gpu conda environment

echo ========================================
echo  Cancer Classification API Server
echo  GPU-Accelerated (torch_gpu)
echo ========================================
echo.

cd /d "%~dp0"

REM Activate torch_gpu environment
call conda activate torch_gpu

REM Check if activation was successful
if errorlevel 1 (
    echo [ERROR] Could not activate torch_gpu environment
    echo.
    echo Please ensure you have created the torch_gpu environment with:
    echo conda create -n torch_gpu python=3.11
    echo conda activate torch_gpu
    echo conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    echo pip install flask flask-cors pillow opencv-python scikit-image werkzeug
    echo.
    pause
    exit /b 1
)

echo ✓ torch_gpu environment activated
echo.

REM Display GPU info
echo Checking GPU...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}' if torch.cuda.is_available() else 'Running on CPU')"
echo.

REM Start Flask server
echo Starting Flask API server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python app.py

pause
