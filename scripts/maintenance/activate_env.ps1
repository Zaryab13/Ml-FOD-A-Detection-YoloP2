# Quick activation script for the virtual environment
# Just run: .\activate_env.ps1

$venvPath = ".\venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & $venvPath
    Write-Host ""
    Write-Host "Virtual environment activated!" -ForegroundColor Green
    Write-Host "Python location: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: .\setup\install_dependencies.ps1" -ForegroundColor Yellow
}
