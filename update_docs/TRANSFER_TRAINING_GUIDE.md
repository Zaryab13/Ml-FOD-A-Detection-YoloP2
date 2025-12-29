# Transfer Learning Training on New PC - Setup Guide

## Overview

This guide shows how to start YOLOv8-P2 **transfer learning** training on a new PC using the models and results from your first PC.

---

## What You'll Get from Git

When you clone/pull the repository on the new PC, you'll get **EVERYTHING except the dataset**:

✅ All code and scripts
✅ All training results from PC 1
✅ **All model weights** including:
  - Pre-trained models (`models/yolov8m.pt`, etc.)
  - All epoch checkpoints from previous training
  - Best/last weights from all training runs
✅ All plots, metrics, and configs

❌ Dataset images and labels (too large - copy manually)

---

## Step-by-Step Setup on New PC

### Step 1: Clone the Repository

```powershell
# Clone from GitHub/GitLab
git clone <your-repo-url>
cd "ML Project\Code"
```

### Step 2: Setup Python Environment

```powershell
# Install all dependencies
.\setup\install_dependencies.ps1

# Activate environment
.\activate_env.ps1

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Copy Dataset (One-Time Manual Transfer)

The dataset is NOT in Git (too large). You need to copy it manually:

#### Option A: Network Share (Recommended if PCs are on same network)
```powershell
# On PC 1, share the data folder
# On PC 2, copy from network share
Copy-Item "\\PC1\SharedFolder\FOD-A" -Destination "data\" -Recurse
```

#### Option B: External Drive
```powershell
# On PC 1, copy to USB/external drive
Copy-Item "data\FOD-A" -Destination "E:\Backup\" -Recurse

# On PC 2, copy from drive
Copy-Item "E:\Backup\FOD-A" -Destination "data\" -Recurse
```

#### Option C: Re-convert from Pascal VOC
```powershell
# If you have Pascal VOC format on PC 2
python utils\convert_voc_to_yolo.py `
  --voc-root "data\FODPascalVOCFormat-V.2.1\VOC2007" `
  --yolo-root "data\FOD-A"
```

### Step 4: Verify Dataset Structure

```powershell
# Check dataset
python utils\check_status.py

# Should show:
# ✅ Train images: 25,298
# ✅ Val images: 8,429
```

Expected structure:
```
data/FOD-A/
├── images/
│   ├── train/  (25,298 images)
│   └── val/    (8,429 images)
├── labels/
│   ├── train/  (25,298 .txt files)
│   └── val/    (8,429 .txt files)
└── data.yaml   (already in Git)
```

### Step 5: Start Transfer Learning Training

```powershell
# Run transfer learning script
python train_yolov8_p2_transfer.py
```

**What this does:**
- Uses pre-trained `models/yolov8m.pt` (already in repo from Git)
- Builds YOLOv8-P2 architecture
- Initializes with COCO pre-trained weights
- Fine-tunes on FOD-A dataset
- Lower learning rate (0.001 vs 0.01) for transfer learning

### Step 6: Push Results Back to Git

After training completes:

```powershell
# Add all new results (checkpoints, plots, metrics)
git add .

# Commit
git commit -m "Complete: YOLOv8-P2 transfer learning on PC 2"

# Push to share with PC 1
git push
```

---

## Training Script Details

### `train_yolov8_p2_transfer.py` Configuration:

```python
MODEL_CONFIG = "configs/yolov8-p2.yaml"          # P2 architecture
PRETRAINED_WEIGHTS = "./models/yolov8m.pt"       # Pre-trained from Git
IMG_SIZE = 1280                                   # High res for small objects
BATCH_SIZE = 6                                    # Adjust for your GPU
EPOCHS = 100
PATIENCE = 20
```

### Key Differences from Scratch Training:

| Setting | From Scratch | Transfer Learning |
|---------|-------------|-------------------|
| Learning Rate | 0.01 | 0.001 (10x lower) |
| Pretrained | False | True |
| Weights Init | Random | COCO weights |
| Convergence | Slower | Faster |

---

## Monitoring Training

### Real-time Progress:

Training will show:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/100      22.1G      1.234      0.567      0.891        128       1280
```

### Check Results:

```powershell
# View training curves
explorer fod_detection\yolov8m-p2_transfer_*\

# Check metrics
type fod_detection\yolov8m-p2_transfer_*\results.csv
```

---

## After Training Completes

### Compare with PC 1 Results:

```powershell
# Your repo now has results from BOTH PCs:
# - PC 1: YOLOv8-P2 (from scratch)
# - PC 2: YOLOv8-P2 Transfer (with pre-trained weights)

# Compare mAP scores
python -c "
import pandas as pd
scratch = pd.read_csv('fod_detection/yolov8m-p2_*/results.csv')
transfer = pd.read_csv('fod_detection/yolov8m-p2_transfer_*/results.csv')
print('Scratch best mAP50:', scratch['metrics/mAP50(B)'].max())
print('Transfer best mAP50:', transfer['metrics/mAP50(B)'].max())
"
```

### Share Results:

```powershell
# Commit and push
git add .
git commit -m "Results: Transfer learning vs from-scratch comparison"
git push

# Pull on PC 1 to get the comparison
```

---

## Syncing Between Both PCs

### PC 1 → PC 2 (Get latest from PC 1):
```powershell
# On PC 2
git pull
```

### PC 2 → PC 1 (Get transfer learning results):
```powershell
# On PC 1
git pull
```

### Both PCs Training Simultaneously:
```powershell
# Before starting work
git pull

# After each milestone (every few hours)
git add .
git commit -m "Progress: Epoch 50/100"
git push
```

---

## Troubleshooting

### "Dataset not found"
**Solution:** You need to copy the dataset manually (see Step 3)
```powershell
# Check if dataset exists
Test-Path "data\FOD-A\images\train"
```

### "CUDA out of memory"
**Solution:** Reduce batch size in `train_yolov8_p2_transfer.py`:
```python
BATCH_SIZE = 4  # Instead of 6
```

### "Model weights not found"
**Solution:** Make sure you pulled from Git first:
```powershell
git pull
ls models\*.pt  # Should show yolov8m.pt
```

### "Git says repository is too large"
**Solution:** Dataset should NOT be in Git. Verify:
```powershell
# Check .gitignore is excluding dataset
git check-ignore data/FOD-A/images/train/000000.jpg
# Should output the filename (means it's ignored)
```

---

## Quick Reference

### Essential Commands:

```powershell
# 1. Initial setup (one-time)
git clone <repo-url>
.\setup\install_dependencies.ps1
.\activate_env.ps1
# Copy dataset manually

# 2. Start training
python train_yolov8_p2_transfer.py

# 3. Share results
git add .
git commit -m "Training update"
git push

# 4. Get updates from other PC
git pull
```

---

## Expected Timeline

| Task | Duration |
|------|----------|
| Git clone | 5-10 min (includes all weights) |
| Environment setup | 10-15 min |
| Dataset copy | 30-60 min (depends on transfer method) |
| Training (100 epochs) | 6-12 hours (GPU dependent) |

---

## Summary

✅ **Code, models, results** → Automatically synced via Git  
❌ **Dataset only** → Copy manually once  
🚀 **Transfer learning** → Uses pre-trained weights from Git  
🔄 **Both PCs** → Can train simultaneously and sync results  

**You're ready to start transfer learning on the new PC!** 🎯

---

**Last Updated:** December 22, 2025  
**Status:** Ready for multi-PC transfer learning workflow
