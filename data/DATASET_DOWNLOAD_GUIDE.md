# FOD-A Dataset Download and Setup Guide

## 📥 Step-by-Step Download Instructions

### Option 1: Manual Download (Recommended for Beginners)

#### 1. Download the Dataset

**Choose ONE version** (we'll use Pascal VOC for YOLO compatibility):

**Pascal VOC Format (412 MB)** - RECOMMENDED
- Link: https://drive.google.com/file/d/1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x/view?usp=sharing
- Size: 412 MB (300×300 images)
- Format: Pascal VOC (XML annotations)
- Includes: Train/Val splits

**Original Format (8.3 GB)** - For Dataset Extension
- Link: https://drive.google.com/file/d/1lLBJXXaQCWaFa-1MeLAANPpSwMhCJqGh/view?usp=sharing
- Size: 8.3 GB (400×400 images)
- Format: Original format

#### 2. Extract the Downloaded File

```powershell
# After downloading, extract to a temporary location
# Example: Extract to D:\Downloads\FOD-A-VOC\
```

#### 3. Move to Project Data Directory

```powershell
# Move the extracted folder to your project
# Target: D:\Zaryab\Course Work\Machine Learning\ML Project\Code\data\FOD-A\
```

---

### Option 2: Automated Download with gdown (Faster)

We can use Python's `gdown` library to download directly from Google Drive.

#### Install gdown

```powershell
.\activate_env.ps1
pip install gdown
```

#### Download Pascal VOC Format (412 MB)

```powershell
# Create download script
python data\download_fod_dataset.py
```

---

## 🔄 Convert Pascal VOC to YOLO Format

After downloading, we need to convert from Pascal VOC (XML) to YOLO format (TXT).

### Step 1: Inspect the Downloaded Structure

```powershell
.\activate_env.ps1
python -c "import os; print('\n'.join(os.listdir('data/FOD-A')))"
```

Expected structure (Pascal VOC):
```
FOD-A/
├── JPEGImages/         # Images
├── Annotations/        # XML files (Pascal VOC format)
├── ImageSets/
│   └── Main/
│       ├── train.txt   # List of training images
│       └── val.txt     # List of validation images
└── README.md
```

### Step 2: Convert to YOLO Format

We'll create a conversion script that:
1. Reads Pascal VOC XML files
2. Converts bounding boxes to YOLO format
3. Organizes into YOLO directory structure

```powershell
python utils\convert_voc_to_yolo.py
```

---

## 📋 Quick Setup Commands

### Complete Setup Sequence

```powershell
# 1. Activate environment
.\activate_env.ps1

# 2. Install gdown for Google Drive downloads
pip install gdown

# 3. Download dataset (Pascal VOC - 412MB)
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x', 'data/FOD-A-VOC.zip', quiet=False)"

# 4. Extract (if you have 7zip or use Windows Explorer)
# Right-click FOD-A-VOC.zip → Extract All → data/FOD-A/

# 5. Convert to YOLO format
python utils\convert_voc_to_yolo.py

# 6. Validate dataset
python utils\check_status.py

# 7. Explore in Jupyter
jupyter notebook notebooks\01_dataset_exploration.ipynb
```

---

## 🎯 Which Version Should You Use?

| Version | Size | Image Size | Use Case |
|---------|------|------------|----------|
| **Pascal VOC** | 412 MB | 300×300 | ✅ **Start here** - Training & experiments |
| Original | 8.3 GB | 400×400 | Dataset expansion/custom resolution |

**Recommendation**: Start with **Pascal VOC** (smaller, pre-split, easier to work with).

---

## 🔧 Troubleshooting

### Issue: Google Drive Download Quota Exceeded

If direct download fails due to quota:
1. Open link in browser
2. Add to your Google Drive ("Add to My Drive")
3. Download from your own Drive

### Issue: gdown Not Working

```powershell
# Try alternative download method
pip install gdown --upgrade

# Or use browser download + manual extract
```

### Issue: Extraction Taking Too Long

```powershell
# Use 7-Zip for faster extraction (if installed)
# Or just use Windows Explorer: Right-click → Extract All
```

---

## ✅ Verification After Setup

Once downloaded and converted:

```powershell
.\activate_env.ps1
python utils\check_status.py
```

**Expected output**:
```
✅ FOD-A root
✅ Train images: ~X images
✅ Train labels: ~X labels  
✅ Val images: ~X images
✅ Val labels: ~X labels
```

---

## 📊 Dataset Specifications

From the paper (FOD-A v2.1):
- **Classes**: 31 object categories
- **Images**: Thousands of annotated images
- **Splits**: Pre-defined train/val split
- **Resolution**: 300×300 (Pascal VOC) or 400×400 (Original)
- **Format**: RGB images with runway/taxiway backgrounds
- **Annotations**: Bounding boxes + Light level + Weather conditions

---

## 🚀 Next Steps After Download

1. **Validate**: Run `python utils\check_status.py`
2. **Explore**: Open `notebooks\01_dataset_exploration.ipynb`
3. **Verify**: Check class distribution and object sizes
4. **Ready**: Proceed to Week 1 training!

---

**Need Help?** Just ask and I'll create the download/conversion scripts for you!
