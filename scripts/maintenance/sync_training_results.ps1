# Sync Training Results Between PCs
# This script prepares training results for Git sync while excluding heavy files

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Results Sync Preparation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = $PSScriptRoot

# Create results sync directory
$syncDir = Join-Path $projectRoot "results_sync"
New-Item -ItemType Directory -Force -Path $syncDir | Out-Null

# Function to copy training results (excluding heavy weights)
function Sync-TrainingRun {
    param($runPath)
    
    $runName = Split-Path $runPath -Leaf
    Write-Host "📦 Processing: $runName" -ForegroundColor Yellow
    
    # Copy key files only
    $filesToCopy = @(
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png"
    )
    
    foreach ($file in $filesToCopy) {
        $srcFile = Join-Path $runPath $file
        if (Test-Path $srcFile) {
            $destDir = Join-Path $syncDir $runName
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
            Copy-Item $srcFile -Destination $destDir -Force
            Write-Host "   ✓ $file" -ForegroundColor Green
        }
    }
    
    # Copy best.pt and last.pt ONLY (exclude epoch checkpoints)
    $weightsDir = Join-Path $runPath "weights"
    if (Test-Path $weightsDir) {
        $destWeights = Join-Path $syncDir "$runName\weights"
        New-Item -ItemType Directory -Force -Path $destWeights | Out-Null
        
        $bestPt = Join-Path $weightsDir "best.pt"
        $lastPt = Join-Path $weightsDir "last.pt"
        
        if (Test-Path $bestPt) {
            Copy-Item $bestPt -Destination $destWeights -Force
            $size = [math]::Round((Get-Item $bestPt).Length / 1MB, 2)
            Write-Host "   ✓ best.pt ($size MB)" -ForegroundColor Green
        }
        
        if (Test-Path $lastPt) {
            Copy-Item $lastPt -Destination $destWeights -Force
            $size = [math]::Round((Get-Item $lastPt).Length / 1MB, 2)
            Write-Host "   ✓ last.pt ($size MB)" -ForegroundColor Green
        }
    }
}

# Find all training runs
$fodDetectionDir = Join-Path $projectRoot "fod_detection"
if (Test-Path $fodDetectionDir) {
    $runs = Get-ChildItem -Path $fodDetectionDir -Directory
    
    Write-Host "Found $($runs.Count) training run(s)" -ForegroundColor Cyan
    Write-Host ""
    
    foreach ($run in $runs) {
        Sync-TrainingRun -runPath $run.FullName
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Sync Preparation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Results prepared in: results_sync/" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. git add ." -ForegroundColor White
Write-Host "2. git commit -m 'Update training results'" -ForegroundColor White
Write-Host "3. git push" -ForegroundColor White
Write-Host ""
Write-Host "On other PC:" -ForegroundColor Cyan
Write-Host "1. git pull" -ForegroundColor White
Write-Host "2. Copy results from results_sync/ to fod_detection/" -ForegroundColor White
Write-Host ""
