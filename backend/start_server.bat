@echo off
echo ========================================
echo  Cancer Classification API Server
echo  Using CUDA GPU Environment
echo ========================================
echo.

cd /d "%~dp0"

REM Use torch_gpu conda environment
echo Activating torch_gpu environment...
call conda activate torch_gpu
if errorlevel 1 (
    echo [ERROR] Failed to activate torch_gpu environment
    echo Please ensure Anaconda/Miniconda is installed and torch_gpu environment exists
    echo.
    echo To create the environment, run:
    echo conda create -n torch_gpu python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    echo.
    pause
    exit /b 1
)
echo [OK] torch_gpu environment activated
echo.

echo [OK] torch_gpu environment activated
echo.

REM Check if required packages are installed in conda environment
echo Checking dependencies...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask dependencies in torch_gpu environment...
    pip install flask flask-cors pillow opencv-python scikit-image werkzeug
    echo.
)

python -c "import flask_cors" 2>nul
if errorlevel 1 (
    echo Installing additional dependencies...
    pip install flask-cors
    echo.
)

REM Check if model files exist
echo Checking model files...
set "MODELS_OK=1"

if not exist "src\brain\brain_tumor_classifier_v1.pth" (
    echo [WARNING] Brain tumor model not found: src\brain\brain_tumor_classifier_v1.pth
    set "MODELS_OK=0"
)

if not exist "src\lungs\lung_cnn_checkpoint.pth" (
    echo [WARNING] Lung cancer model not found: src\lungs\lung_cnn_checkpoint.pth
    set "MODELS_OK=0"
)

if not exist "src\lungs\lung_class_names.pkl" (
    echo [WARNING] Lung class names not found: src\lungs\lung_class_names.pkl
    set "MODELS_OK=0"
)

if not exist "src\skin\skin_cnn_full_model.pth" (
    echo [WARNING] Skin cancer model not found: src\skin\skin_cnn_full_model.pth
    set "MODELS_OK=0"
)

if not exist "src\skin\class_names.pkl" (
    echo [WARNING] Skin class names not found: src\skin\class_names.pkl
    set "MODELS_OK=0"
)

if "%MODELS_OK%"=="0" (
    echo.
    echo [ERROR] Some model files are missing!
    echo Please ensure all model files are in their correct locations.
    echo.
    pause
    exit /b 1
)

echo [OK] All model files found!
echo.

REM Check GPU availability
echo Checking GPU availability...
python check_gpu.py
if errorlevel 1 (
    echo.
    echo [WARNING] GPU check failed or no GPU detected
    echo Server will run on CPU (slower)
    echo Press any key to continue or Ctrl+C to exit...
    pause >nul
)
echo.

REM Start the Flask server
echo Starting Flask API server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
