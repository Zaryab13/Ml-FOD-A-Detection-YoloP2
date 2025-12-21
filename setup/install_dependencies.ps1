# FOD Detection Project - Dependency Installation Script
# Run this script to set up the complete environment with virtual environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FOD Detection Pipeline - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Define venv path
$venvPath = "..\venv"
$venvPython = "$venvPath\Scripts\python.exe"
$venvPip = "$venvPath\Scripts\pip.exe"

# Check Python version
Write-Host "[1/7] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

if ($pythonVersion -notmatch "Python 3\.(9|10|11|12|13)") {
    Write-Host "WARNING: Python 3.9+ required. Current version may cause issues." -ForegroundColor Red
    Write-Host "Please install Python 3.9 or higher from python.org" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/7] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists at: $venvPath" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
        python -m venv $venvPath
        Write-Host "Virtual environment recreated!" -ForegroundColor Green
    } else {
        Write-Host "Using existing virtual environment." -ForegroundColor Green
    }
} else {
    python -m venv $venvPath
    Write-Host "Virtual environment created at: $venvPath" -ForegroundColor Green
}

# Upgrade pip in venv
Write-Host ""
Write-Host "[3/7] Upgrading pip in virtual environment..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip
Write-Host "pip upgraded successfully!" -ForegroundColor Green

# Install PyTorch first (with CUDA support if available)
Write-Host ""
Write-Host "[4/7] Installing PyTorch..." -ForegroundColor Yellow
Write-Host "Detecting CUDA availability..." -ForegroundColor Cyan

# Try to detect NVIDIA GPU
$nvidiaGpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }

if ($nvidiaGpu) {
    Write-Host "NVIDIA GPU detected: $($nvidiaGpu.Name)" -ForegroundColor Green
    Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Cyan
    & $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-Host "CUDA 12.1 failed, trying CPU version..." -ForegroundColor Yellow
        & $venvPip install torch torchvision torchaudio
    }
} else {
    Write-Host "No NVIDIA GPU detected. Installing CPU-only PyTorch..." -ForegroundColor Yellow
    Write-Host "WARNING: Training will be VERY slow without GPU!" -ForegroundColor Red
    & $venvPip install torch torchvision torchaudio
}

# Install remaining dependencies
Write-Host ""
Write-Host "[5/7] Installing remaining Python dependencies..." -ForegroundColor Yellow
& $venvPip install -r ..\..\requirements.txt --upgrade
Write-Host "All dependencies installed!" -ForegroundColor Green

# Verify CUDA availability
Write-Host ""
Write-Host "[6/7] Verifying PyTorch CUDA setup..." -ForegroundColor Yellow
$cudaCheck = & $venvPython -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
Write-Host $cudaCheck -ForegroundColor $(if ($cudaCheck -match "CUDA Available: True") { "Green" } else { "Yellow" })

if ($cudaCheck -match "CUDA Available: True") {
    Write-Host "✓ GPU acceleration enabled - Training will be fast!" -ForegroundColor Green
} else {
    Write-Host "✗ No CUDA detected. Training will use CPU (very slow)." -ForegroundColor Yellow
    Write-Host "Consider using Google Colab or Kaggle for GPU access." -ForegroundColor Cyan
}

# Verify Ultralytics installation
Write-Host ""
Write-Host "[7/7] Verifying installations..." -ForegroundColor Yellow
$ultraCheck = & $venvPython -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully!')" 2>&1
$sahiCheck = & $venvPython -c "import sahi; print(f'SAHI version: {sahi.__version__}')" 2>&1
Write-Host "✓ $ultraCheck" -ForegroundColor Green
Write-Host "✓ $sahiCheck" -ForegroundColor Green

# Setup complete
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Virtual Environment Created at: venv\" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Activate venv: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Place FOD-A dataset in: .\data\FOD-A\" -ForegroundColor White
Write-Host "  3. Run: jupyter notebook .\notebooks\01_dataset_exploration.ipynb" -ForegroundColor White
Write-Host "  4. Check README.md for dataset structure requirements" -ForegroundColor White
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor Cyan
Write-Host "  Activate venv:   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  Deactivate:      deactivate" -ForegroundColor White
Write-Host "  Run Jupyter:     jupyter notebook" -ForegroundColor White
Write-Host ""
