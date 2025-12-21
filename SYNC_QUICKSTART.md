# Quick Start: Multi-PC Training Sync

## ✅ Configuration Complete!

Your repository syncs **EVERYTHING** except the large dataset files.

---

## What Gets Synced

### ✅ SHARED (via Git):
```
Everything including all training outputs!
```

**Code & Configs:**
- All Python scripts (`.py`)
- YAML configs (`configs/*.yaml`)
- Requirements (`requirements.txt`)
- Documentation (`.md` files)

**Training Results & Models (ALL OF THEM):**
- `args.yaml` (training config)
- `results.csv` (metrics per epoch)
- All plots (`.png`, `.jpg` files)
- **ALL model weights** (`*.pt` files):
  - `best.pt`
  - `last.pt`
  - `epoch0.pt`, `epoch10.pt`, `epoch20.pt`, etc. (ALL checkpoints)
- Complete `fod_detection/` folder with all runs
- Complete `runs/` folder
- Pre-trained models in `models/` folder

### ❌ EXCLUDED (ONLY the dataset):
- Dataset images (`data/FOD-A/images/`) - ~10 GB
- Dataset labels (`data/FOD-A/labels/`) - ~500 MB
- Pascal VOC format dataset (`data/FODPascalVOCFormat-V.2.1/`)
- Virtual environment (`venv/`)
- W&B cache (`wandb/`)

---

## First-Time Setup

### On Current PC (PC 1):

1. **Initialize Git repo (if not done):**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: FOD detection project"
   ```

2. **Connect to remote (GitHub/GitLab):**
   ```powershell
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

### On New PC (PC 2):

1. **Clone the repository:**
   ```powershell
   git clone <your-repo-url>
   cd "ML Project\Code"
   ```

2. **Setup environment:**
   ```powershell
   # Install dependencies
   .\setup\install_dependencies.ps1
   
   # Activate environment
   .\activate_env.ps1
   ```

3. **Copy dataset MANUALLY (one-time):**
   ```powershell
   # Option A: Copy from PC 1 via network share/USB
   # Copy entire data/FOD-A/ folder to PC 2
   
   # Option B: Keep dataset in shared network location
   # Update data/FOD-A/data.yaml with network path
   ```

4. **Verify setup:**
   ```powershell
   python utils/check_status.py
   ```

---

## Daily Workflow

### Before Starting Work:
```powershell
# Pull latest changes from other PC
git pull
```

### After Training Session:
```powershell
# Add new results
git add .

# Check what will be committed
git status

# Commit with descriptive message
git commit -m "YOLOv8-P2: Completed 100 epochs (mAP50: 0.82)"

# Push to share with other PC
git push
```

---

## Commands Cheat Sheet

### Check Status:
```powershell
git status              # See changed files
git diff                # See detailed changes
git log --oneline       # View commit history
```

### Sync Changes:
```powershell
git pull                # Get latest from other PC
git add .               # Stage all changes
git commit -m "msg"     # Commit changes
git push                # Upload to remote
```

### Handle Conflicts:
```powershell
git pull                # May show conflicts
# Edit conflicted files manually
git add <file>          # Mark as resolved
git commit              # Complete merge
git push                # Upload resolution
```

---

## Example Scenario

**PC 1 (Current):**
```powershell
# Continue your current training
python train_yolov8_p2.py

# After training completes:
git add fod_detection/yolov8m-p2_*/
git commit -m "Complete: YOLOv8-P2 training 100 epochs"
git push
```

**PC 2 (New):**
```powershell
# Get ALL results including all weights
git pull

# Now you have:
# - ALL model weights (best.pt, last.pt, epoch*.pt)
# - All training metrics and plots
# - Code and configs
# - Pre-trained models

# Setup environment and copy dataset
.\setup\install_dependencies.ps1
.\activate_env.ps1

# Copy ONLY the dataset from PC 1 (one-time)
# Option 1: Network share
# Option 2: External drive
# Option 3: Re-convert from Pascal VOC

# Start transfer learning (uses yolov8m.pt from models/)
python train_yolov8_p2_transfer.py

# After training:
git add .
git commit -m "Complete: Transfer learning training"
git push
```

**Back on PC 1:**
```powershell
# Get transfer learning results
git pull

# Now both PCs have all results!
```

---

## File Size Guidelines

✅ **Safe to commit:**
- Results CSV files: < 1 MB
- PNG plots: < 1 MB each
- Config YAML: < 100 KB
- best.pt/last.pt: 50-165 MB

⚠️ **Too large (excluded):**
- Epoch checkpoints: 50-165 MB each (10+ files)
- Dataset images: 5-10 GB
- Virtual environment: 2-5 GB

---

## Troubleshooting

### "Repository is too large"
Check for accidentally committed large files:
```powershell
git ls-files | ForEach-Object { 
    if (Test-Path $_) {
        $size = (Get-Item $_).Length / 1MB
        if ($size -gt 10) {
            Write-Host "$_ : $size MB"
        }
    }
}
```

### "Dataset not found on new PC"
Dataset is NOT synced via git. Copy manually:
```powershell
# From PC 1, copy to USB/network share:
Copy-Item "data\FOD-A" -Destination "E:\Backup\" -Recurse

# On PC 2, copy from USB/network share:
Copy-Item "E:\Backup\FOD-A" -Destination "data\" -Recurse
```

### "Merge conflict in results.csv"
Keep both versions:
```powershell
# View conflict
git diff results.csv

# Keep both (manually merge)
# Or keep one version:
git checkout --ours results.csv    # Keep your version
# OR
git checkout --theirs results.csv  # Keep other PC's version

git add results.csv
git commit
```

---

## Next Steps

1. **On PC 1 (now):**
   ```powershell
   git add .
   git commit -m "Setup multi-PC sync configuration"
   git push
   ```

2. **On PC 2:**
   ```powershell
   git clone <repo-url>
   # Follow "First-Time Setup" above
   ```

3. **Start training on both PCs!**
   - PC 1: Continue current training
   - PC 2: Start transfer learning

**You're all set! Training results will sync automatically via Git.** 🚀

---

**Created:** December 22, 2025  
**Status:** Ready for multi-PC training workflow
