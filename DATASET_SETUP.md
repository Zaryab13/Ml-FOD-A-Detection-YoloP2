# Dataset Setup Guide

## Current Status

✅ **Pascal VOC Format Dataset Found**
- Location: `data/FODPascalVOCFormat-V.2.1/VOC2007/`
- Contains: Annotations (XML), JPEGImages, ImageSets

✅ **YOLO Format Dataset Found**
- Location: `data/FOD-A/`
- Already converted and ready to use!

---

## Option 1: Use Existing YOLO Dataset (RECOMMENDED)

Your dataset is **already converted** and ready! Just verify it:

```powershell
# Check dataset structure
python utils/check_status.py
```

If everything looks good, you can **start training immediately**:

```powershell
# Train baseline model
python train_yolov8_base.py

# Or train P2 model for small objects
python train_yolov8_p2.py
```

---

## Option 2: Re-convert from Pascal VOC (if needed)

If you need to re-convert the Pascal VOC dataset to YOLO format:

```powershell
# Run the conversion script
python utils/convert_voc_to_yolo.py --voc-root "data/FODPascalVOCFormat-V.2.1/VOC2007" --yolo-root "data/FOD-A-NEW"
```

This will:
1. ✅ Read XML annotations from Pascal VOC format
2. ✅ Convert bounding boxes to YOLO format (normalized coordinates)
3. ✅ Split into train/val sets
4. ✅ Create `data.yaml` configuration file
5. ✅ Copy all images to YOLO structure

**Expected output:**
```
data/FOD-A-NEW/
├── images/
│   ├── train/  (training images)
│   └── val/    (validation images)
├── labels/
│   ├── train/  (YOLO .txt annotations)
│   └── val/    (YOLO .txt annotations)
└── data.yaml   (dataset configuration)
```

---

## Git Workflow (Already Configured!)

Your `.gitignore` is properly set up to **exclude large files**:

### What WILL be tracked by Git:
✅ Code files (`*.py`)
✅ Configs (`*.yaml`)
✅ Documentation (`*.md`)
✅ Scripts (`*.ps1`)
✅ Notebooks (`*.ipynb`)
✅ `data.yaml` (dataset config - small file, safe to commit)

### What WON'T be tracked (too large):
❌ All images (`*.jpg`, `*.png`)
❌ All labels in `data/*/labels/`
❌ All images in `data/*/images/`
❌ Entire `data/FOD-A/` and `data/FODPascalVOCFormat-V.2.1/` folders
❌ Model weights (`*.pt`)
❌ Training runs (`runs/`)

### Safe Git Commands:

```powershell
# Add all files (large files automatically excluded)
git add .

# Commit
git commit -m "Initial project setup"

# Check what will be committed (verify no large files)
git status
```

---

## Verification Steps

### 1. Check Dataset Structure
```powershell
python -c "from utils.dataset_loader import validate_dataset; validate_dataset('data/FOD-A')"
```

### 2. Count Images and Labels
```powershell
# Count training images
(Get-ChildItem "data/FOD-A/images/train" -Include *.jpg,*.png -Recurse).Count

# Count training labels
(Get-ChildItem "data/FOD-A/labels/train" -Include *.txt -Recurse).Count

# Count validation images
(Get-ChildItem "data/FOD-A/images/val" -Include *.jpg,*.png -Recurse).Count

# Count validation labels
(Get-ChildItem "data/FOD-A/labels/val" -Include *.txt -Recurse).Count
```

### 3. Visualize Sample
```powershell
# Open Jupyter notebook
jupyter notebook

# Then open: notebooks/01_dataset_exploration.ipynb
```

---

## Understanding Pascal VOC vs YOLO Format

### Pascal VOC Format (XML):
```xml
<object>
  <name>bolt</name>
  <bndbox>
    <xmin>100</xmin>
    <ymin>200</ymin>
    <xmax>150</xmax>
    <ymax>250</ymax>
  </bndbox>
</object>
```
- Absolute pixel coordinates
- One XML file per image

### YOLO Format (TXT):
```
0 0.5625 0.4375 0.125 0.075
```
- Format: `class_id x_center y_center width height`
- All values normalized (0-1)
- One TXT file per image

The conversion script (`utils/convert_voc_to_yolo.py`) handles this automatically!

---

## Common Issues & Solutions

### Issue: "Dataset not found"
**Solution:** Check that `data/FOD-A/data.yaml` exists and has correct paths

### Issue: "No images in training set"
**Solution:** Run the conversion script to populate the dataset

### Issue: "Git tracking too many files"
**Solution:** Already fixed! `.gitignore` properly excludes large files

### Issue: "Want to backup dataset"
**Solution:** 
- Dataset is NOT in git (too large)
- Keep original Pascal VOC format as backup
- Can always re-convert using `convert_voc_to_yolo.py`

---

## Recommended Next Steps

1. **Verify current dataset:**
   ```powershell
   python utils/check_status.py
   ```

2. **Add code to Git (safe):**
   ```powershell
   git add .
   git commit -m "FOD detection project setup"
   ```

3. **Explore dataset:**
   ```powershell
   jupyter notebook notebooks/01_dataset_exploration.ipynb
   ```

4. **Start training:**
   ```powershell
   python train_yolov8_p2.py
   ```

---

## Summary

✅ Your dataset is **ready to use**  
✅ `.gitignore` **properly configured**  
✅ **Safe to run** `git add .` (large files excluded)  
✅ Can **re-convert anytime** from Pascal VOC format  
✅ **No manual work needed** - everything is automated!

**You're all set to start training!** 🚀
