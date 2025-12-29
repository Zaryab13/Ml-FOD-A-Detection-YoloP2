# Multi-PC Training Synchronization Guide

## Setup Overview

You want to train models on **2 different PCs** while sharing:
✅ Code, configs, scripts
✅ Training results (CSV, plots)
✅ Best model weights (best.pt, last.pt)

**EXCLUDE from sync:**
❌ Dataset images/labels (too large)
❌ All epoch checkpoints (epoch0.pt, epoch10.pt, etc.)
❌ Intermediate weights

---

## Git Configuration (Already Set Up!)

Your `.gitignore` is configured to:

### ✅ SHARE (lightweight):
- All Python code (`*.py`)
- All configs (`*.yaml`, `*.json`)
- Training results (`results.csv`)
- Training plots (`*.png` in `runs/`)
- Args/configs (`args.yaml`)
- **Best weights only** (`best.pt`, `last.pt`)

### ❌ EXCLUDE (heavy):
- All dataset images (`*.jpg`, `*.png` in `data/`)
- All labels (`data/*/labels/`)
- Epoch checkpoints (`epoch*.pt`)
- Large model files (>100MB)

---

## Workflow for 2 PCs

### **PC 1 (Current - Training YOLOv8-P2)**

1. **Commit your current training results:**
   ```powershell
   # Add all shareable files (heavy files auto-excluded)
   git add .
   
   # Check what will be committed (verify no large files)
   git status
   
   # Commit
   git commit -m "Add YOLOv8-P2 training results"
   
   # Push to remote
   git push origin main
   ```

2. **Continue training:**
   ```powershell
   # Your current training keeps running
   python train_yolov8_p2.py
   ```

### **PC 2 (New PC - Training YOLOv8-P2 Transfer)**

1. **First time setup:**
   ```powershell
   # Clone the repo
   git clone <your-repo-url>
   cd "ML Project/Code"
   
   # Setup environment
   .\setup\install_dependencies.ps1
   
   # Activate environment
   .\activate_env.ps1
   ```

2. **Copy ONLY the dataset (manual, one-time):**
   ```powershell
   # Option A: Copy from PC 1 via network/USB
   # Copy data/FOD-A/ folder to PC 2
   
   # Option B: Re-convert from Pascal VOC
   python utils/convert_voc_to_yolo.py
   ```

3. **Start transfer learning training:**
   ```powershell
   python train_yolov8_p2_transfer.py
   ```

4. **Push your results:**
   ```powershell
   git add .
   git commit -m "Add YOLOv8-P2 transfer learning results"
   git push
   ```

---

## Continuous Sync Between PCs

### Before starting work (on either PC):
```powershell
# Pull latest changes
git pull

# Check for conflicts
git status
```

### After training session (on either PC):
```powershell
# Add new results
git add .

# Commit with descriptive message
git commit -m "Update: YOLOv8-P2 epoch 50 results (mAP: 0.75)"

# Push to share with other PC
git push
```

---

## File Size Management

### What gets synced (typical sizes):

| File | Size | Sync? |
|------|------|-------|
| `results.csv` | ~50 KB | ✅ Yes |
| `args.yaml` | ~5 KB | ✅ Yes |
| `confusion_matrix.png` | ~500 KB | ✅ Yes |
| `best.pt` | ~50-100 MB | ✅ Yes (only best) |
| `last.pt` | ~50-100 MB | ✅ Yes (only last) |
| `epoch10.pt` | ~50-100 MB | ❌ No (excluded) |
| `epoch20.pt` | ~50-100 MB | ❌ No (excluded) |
| Dataset images | ~5-10 GB | ❌ No (manual copy) |

**Total per training run:** ~200-300 MB (manageable for Git)

---

## Handling Large Files (Optional: Git LFS)

If `best.pt` files are still too large (>100MB), use Git LFS:

```powershell
# Install Git LFS (one-time)
git lfs install

# Track large model files
git lfs track "**/*.pt"
git lfs track "fod_detection/**/weights/best.pt"
git lfs track "fod_detection/**/weights/last.pt"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

---

## Common Scenarios

### Scenario 1: PC 1 finishes training, share with PC 2

**PC 1:**
```powershell
git add fod_detection/yolov8m-p2_*/
git commit -m "Complete: YOLOv8-P2 training (mAP50: 0.82)"
git push
```

**PC 2:**
```powershell
git pull
# Now you have the results and best weights!
```

### Scenario 2: Both PCs training simultaneously

**Both PCs** - Push/pull frequently:
```powershell
# Every few hours or after key milestones
git pull  # Get other PC's updates
git add .
git commit -m "Update: Training progress"
git push
```

### Scenario 3: Conflict resolution

If both PCs edit the same file:
```powershell
git pull
# If conflict occurs:
# 1. Check conflicted files: git status
# 2. Manually resolve conflicts
# 3. git add <resolved-files>
# 4. git commit
# 5. git push
```

---

## Directory Structure (Synced vs Local)

```
Code/
├── 📤 SYNCED (Git):
│   ├── *.py (all Python code)
│   ├── configs/ (YAML configs)
│   ├── utils/ (helper scripts)
│   ├── notebooks/ (.ipynb files)
│   ├── requirements.txt
│   ├── README.md
│   ├── fod_detection/
│   │   ├── yolov8m-p2_*/
│   │   │   ├── args.yaml ✅
│   │   │   ├── results.csv ✅
│   │   │   ├── *.png ✅
│   │   │   └── weights/
│   │   │       ├── best.pt ✅
│   │   │       ├── last.pt ✅
│   │   │       ├── epoch0.pt ❌ (excluded)
│   │   │       └── epoch10.pt ❌ (excluded)
│
├── 💾 LOCAL ONLY (Not synced):
│   ├── data/FOD-A/ (copy manually once)
│   ├── venv/ (recreate on each PC)
│   └── fod_detection/**/weights/epoch*.pt
```

---

## Quick Reference Commands

### Daily Workflow:
```powershell
# 1. Start work session
git pull

# 2. Do your work (training, analysis, etc.)
python train_yolov8_p2.py

# 3. Share results
git add .
git commit -m "Descriptive message"
git push
```

### Check what will be committed:
```powershell
git status
git diff --cached
```

### Undo uncommitted changes:
```powershell
git restore <file>
git restore .  # Restore all
```

### View commit history:
```powershell
git log --oneline
git log --graph --oneline --all
```

---

## Best Practices

✅ **DO:**
- Pull before starting work
- Commit frequently (every milestone)
- Use descriptive commit messages
- Check `git status` before committing
- Keep dataset on local disk (not in git)

❌ **DON'T:**
- Commit dataset images
- Commit all epoch checkpoints
- Force push (`git push -f`) - can lose work
- Ignore merge conflicts
- Commit large files without Git LFS

---

## Troubleshooting

### Issue: "Repo too large"
**Solution:** Check for accidentally committed large files:
```powershell
git ls-files --cached | ForEach-Object { 
    Write-Host "$_ : $((Get-Item $_).Length / 1MB) MB" 
}
```

### Issue: "Dataset missing on new PC"
**Solution:** 
- Datasets are NOT synced via git
- Copy manually from PC 1 or re-download

### Issue: "Merge conflicts in results.csv"
**Solution:**
```powershell
# Keep both results (append)
git checkout --ours results.csv
git add results.csv
git commit
```

---

## Summary

✅ **Code & configs:** Synced automatically  
✅ **Training results:** Synced (CSV, plots)  
✅ **Best weights:** Synced (best.pt, last.pt only)  
❌ **Dataset:** Copy manually (one-time)  
❌ **All epoch checkpoints:** Excluded (too large)  

**You can now train on both PCs simultaneously and share results seamlessly!** 🚀
