# Manual Installation Script - Run this if automatic script has issues
# This script installs packages step-by-step with better error handling

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Manual Dependency Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$venvPath = "..\venv"
$venvPython = "$venvPath\Scripts\python.exe"
$venvPip = "$venvPath\Scripts\pip.exe"

# Check if venv exists
if (-not (Test-Path $venvPath)) {
    Write-Host "❌ Virtual environment not found at: $venvPath" -ForegroundColor Red
    Write-Host "Creating it now..." -ForegroundColor Yellow
    python -m venv $venvPath
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

Write-Host "[1/6] Upgrading pip..." -ForegroundColor Yellow
& $venvPip install --upgrade pip
Write-Host ""

Write-Host "[2/6] Installing PyTorch (this may take a few minutes)..." -ForegroundColor Yellow
Write-Host "Trying CPU version first (safest option)..." -ForegroundColor Cyan
& $venvPip install torch torchvision torchaudio
Write-Host ""

Write-Host "[3/6] Installing Ultralytics YOLO..." -ForegroundColor Yellow
& $venvPip install ultralytics
Write-Host ""

Write-Host "[4/6] Installing SAHI..." -ForegroundColor Yellow
& $venvPip install sahi
Write-Host ""

Write-Host "[5/6] Installing data science packages..." -ForegroundColor Yellow
& $venvPip install numpy pandas matplotlib seaborn plotly scikit-learn opencv-python Pillow
Write-Host ""

Write-Host "[6/6] Installing Jupyter..." -ForegroundColor Yellow
& $venvPip install jupyter ipywidgets
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking installations..." -ForegroundColor Yellow
& $venvPython -c "import torch; print('✓ PyTorch:', torch.__version__)"
& $venvPython -c "import ultralytics; print('✓ Ultralytics: OK')"
& $venvPython -c "import sahi; print('✓ SAHI:', sahi.__version__)"
& $venvPython -c "import numpy; print('✓ NumPy:', numpy.__version__)"
& $venvPython -c "import pandas; print('✓ Pandas:', pandas.__version__)"
& $venvPython -c "import cv2; print('✓ OpenCV:', cv2.__version__)"

Write-Host ""
Write-Host "Checking CUDA support..." -ForegroundColor Yellow
& $venvPython -c "import torch; cuda = torch.cuda.is_available(); print('✓ CUDA Available:', cuda); print('  GPU:', torch.cuda.get_device_name(0) if cuda else 'N/A')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
