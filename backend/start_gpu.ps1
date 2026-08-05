# Cancer Classification API - GPU Startup Script for VS Code
# Run this directly in VS Code terminal

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Cancer Classification API Server" -ForegroundColor Cyan
Write-Host " GPU-Accelerated (torch_gpu)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find Anaconda/Miniconda installation
$condaPaths = @(
    "C:\ProgramData\anaconda3",
    "$env:USERPROFILE\anaconda3",
    "$env:USERPROFILE\miniconda3",
    "C:\Anaconda3",
    "C:\Miniconda3"
)

$condaPath = $null
foreach ($path in $condaPaths) {
    if (Test-Path "$path\Scripts\conda.exe") {
        $condaPath = $path
        Write-Host "✓ Found Conda at: $path" -ForegroundColor Green
        break
    }
}

if (-not $condaPath) {
    Write-Host "❌ Conda not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Anaconda/Miniconda or use Anaconda Prompt" -ForegroundColor Yellow
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Check if torch_gpu environment exists
$torchGpuPython = Join-Path $condaPath "envs\torch_gpu\python.exe"

if (-not (Test-Path $torchGpuPython)) {
    Write-Host "❌ torch_gpu environment not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please open Anaconda Prompt and run:" -ForegroundColor Yellow
    Write-Host "  conda create -n torch_gpu python=3.11 -y" -ForegroundColor White
    Write-Host "  conda activate torch_gpu" -ForegroundColor White
    Write-Host "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y" -ForegroundColor White
    Write-Host "  pip install flask flask-cors pillow opencv-python scikit-image werkzeug" -ForegroundColor White
    exit 1
}

Write-Host "✓ Using Python from: $torchGpuPython" -ForegroundColor Green
Write-Host ""

# Check GPU
Write-Host "Checking GPU..." -ForegroundColor Cyan
& $torchGpuPython -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Running on CPU')"
Write-Host ""

# Start server
Write-Host "Starting Flask server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run app.py with torch_gpu python
& $torchGpuPython app.py
