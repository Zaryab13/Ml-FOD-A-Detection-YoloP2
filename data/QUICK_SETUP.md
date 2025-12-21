# 📥 FOD-A Dataset Setup - Simple 3-Step Process

## Step 1: Download Dataset (5-10 minutes)

### Option A: Automatic Download (Recommended)

```powershell
# Activate environment
.\activate_env.ps1

# Install gdown (Google Drive downloader)
pip install gdown

# Download FOD-A Pascal VOC format (412 MB)
python data\download_fod_dataset.py --version voc
```

**What this does:**
- Downloads from Google Drive automatically
- Extracts the dataset
- Places it in `data/FOD-A/` directory

---

### Option B: Manual Download (If automatic fails)

1. **Open this link in your browser:**
   https://drive.google.com/file/d/1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x/view

2. **Click "Download"** button

3. **Save to**: `D:\Zaryab\Course Work\Machine Learning\ML Project\Code\data\`

4. **Extract the zip file** (Right-click → Extract All)

5. **Rename** the extracted folder to `FOD-A-VOC` (if needed)

---

## Step 2: Convert to YOLO Format (2-3 minutes)

The downloaded dataset is in **Pascal VOC format** (XML files). We need to convert it to **YOLO format** (TXT files).

```powershell
# Make sure venv is activated
.\activate_env.ps1

# Run converter (auto-detects dataset location)
python utils\convert_voc_to_yolo.py
```

**What this does:**
- Reads XML annotation files
- Converts bounding boxes to YOLO format
- Creates proper directory structure:
  ```
  data/FOD-A/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── data.yaml
  ```
- Generates `data.yaml` config file

---

## Step 3: Validate Dataset (1 minute)

Verify everything is set up correctly:

```powershell
# Check dataset status
python utils\check_status.py
```

**Expected output:**
```
✅ FOD-A root
✅ Train images: ~X images
✅ Train labels: ~X labels
✅ Val images: ~X images
✅ Val labels: ~X labels
```

---

## Step 4: Explore Dataset (Optional but Recommended)

Run the exploration notebook to see statistics and visualizations:

```powershell
jupyter notebook
# Open: notebooks/01_dataset_exploration.ipynb
# Click: Cell → Run All
```

**What you'll see:**
- Class distribution (31 FOD categories)
- Object size analysis (small/medium/large)
- Sample images with bounding boxes
- Dataset statistics and recommendations

---

## 🎯 All-in-One Command Sequence

Just copy-paste this entire block:

```powershell
# Navigate to project
cd "D:\Zaryab\Course Work\Machine Learning\ML Project\Code"

# Activate environment
.\activate_env.ps1

# Install downloader
pip install gdown

# Download dataset
python data\download_fod_dataset.py --version voc

# Convert to YOLO
python utils\convert_voc_to_yolo.py

# Validate
python utils\check_status.py

# Explore
jupyter notebook
```

---

## ⏱️ Time Estimate

| Step | Time |
|------|------|
| Download (412 MB) | 5-10 min |
| Extract | 1-2 min |
| Convert to YOLO | 2-3 min |
| Validation | < 1 min |
| **Total** | **~10-15 minutes** |

---

## 🔧 Troubleshooting

### Issue: "gdown: command not found"
```powershell
# Make sure venv is activated
.\activate_env.ps1
pip install gdown
```

### Issue: "Google Drive quota exceeded"
Use **Option B (Manual Download)** instead.

### Issue: "Dataset not found during conversion"
```powershell
# Specify path manually
python utils\convert_voc_to_yolo.py --voc-root "data/YOUR_FOLDER_NAME"
```

### Issue: "No module named 'tqdm'"
```powershell
pip install tqdm pyyaml
```

---

## ✅ Success Checklist

After setup, verify:
- ☑️ `data/FOD-A/images/train/` contains images
- ☑️ `data/FOD-A/images/val/` contains images
- ☑️ `data/FOD-A/labels/train/` contains .txt files
- ☑️ `data/FOD-A/labels/val/` contains .txt files
- ☑️ `data/FOD-A/data.yaml` exists
- ☑️ `python utils\check_status.py` shows all ✅

---

## 🚀 What's Next?

Once dataset is validated:
1. **Explore**: Run Jupyter notebook for dataset analysis
2. **Train**: Start with Week 1 baseline models
3. **Experiment**: Try YOLOv8-P2 and YOLOv11 architectures

**You're ready to start training!** 🎉
