# FOD Detection Pipeline - Quick Start Guide

## 📋 Setup Checklist

### Step 1: Install Dependencies (5-10 minutes)

```powershell
# Navigate to project directory
cd "d:\Zaryab\Course Work\Machine Learning\ML Project\Code"

# Run installation script (creates venv and installs all packages)
.\setup\install_dependencies.ps1
```

**What this does:**
- ✅ Creates isolated Python virtual environment at `./venv/`
- ✅ Installs PyTorch (with CUDA if GPU available)
- ✅ Installs Ultralytics YOLO (v8 + v11)
- ✅ Installs SAHI for sliced inference
- ✅ Installs all data science libraries

**Expected output:**
```
✓ Virtual Environment Created
✓ PyTorch Version: 2.x.x
✓ CUDA Available: True (if GPU present)
✓ Ultralytics YOLO installed
✓ SAHI installed
```

---

### Step 2: Activate Virtual Environment

**Every time you work on this project:**
```powershell
.\activate_env.ps1
```

Or manually:
```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

---

### Step 3: Obtain FOD-A Dataset

**Option A: Download from Source**
- Paper: https://arxiv.org/abs/2110.03072
- Contact authors or check paper for dataset links

**Option B: Request Access**
- Email authors listed in the paper
- Specify academic/research purpose

**Place dataset at:**
```
Code/data/FOD-A/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

### Step 4: Validate Dataset

```powershell
# Activate venv first
.\activate_env.ps1

# Launch Jupyter
jupyter notebook

# Open: notebooks/01_dataset_exploration.ipynb
# Run all cells (Cell → Run All)
```

**This notebook will:**
- ✅ Validate dataset structure
- ✅ Generate `data.yaml` config file
- ✅ Analyze class distribution
- ✅ Show object size statistics
- ✅ Visualize samples with bounding boxes
- ✅ Create plots in `results/` folder

**If successful, you'll see:**
```
✅ Dataset validation complete!
📊 Training: X images, Y annotations
📊 Validation: X images, Y annotations
⚠️ Small objects comprise 85%+ of data
```

---

## 🚀 Training Pipeline (After Setup)

### Week 1: YOLOv8-P2 Baseline

```powershell
# Train vanilla YOLOv8m (reference)
python -m ultralytics.train model=yolov8m.pt data=data/FOD-A/data.yaml epochs=100

# Train YOLOv8-P2 (improved architecture)
python -m ultralytics.train model=configs/yolov8-p2.yaml data=data/FOD-A/data.yaml epochs=100
```

### Week 2: YOLOv11 Experiments

```powershell
# Train YOLOv11m baseline
python -m ultralytics.train model=yolo11m.pt data=data/FOD-A/data.yaml epochs=100

# Train YOLOv11-P2 (custom)
python -m ultralytics.train model=configs/yolov11-p2.yaml data=data/FOD-A/data.yaml epochs=100
```

### Week 3: SAHI Evaluation

```python
# See notebooks/04_sahi_inference.ipynb
```

---

## 🔧 Common Commands

### Check GPU Status
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### List Installed Packages
```powershell
pip list
```

### Update Ultralytics
```powershell
pip install ultralytics --upgrade
```

### Deactivate venv
```powershell
deactivate
```

---

## 📊 Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Setup & Dataset Validation | 1 day | ✅ READY |
| YOLOv8-P2 Training | 2-3 days | ⏳ TODO |
| YOLOv11 Training | 2-3 days | ⏳ TODO |
| SAHI Implementation | 1 day | ⏳ TODO |
| Adversarial Testing | 2 days | ⏳ TODO |
| Analysis & Report | 2 days | ⏳ TODO |

**Total: 10-14 days** (active work, excluding training time)

---

## ⚠️ Troubleshooting

### Issue: "python not recognized"
**Solution:** Install Python 3.9+ from python.org and add to PATH

### Issue: "CUDA Out of Memory"
**Solution:** 
```python
# Reduce batch size in training
model.train(batch=8)  # Instead of 16
```

### Issue: "Dataset not found"
**Solution:** Check path in `data/FOD-A/data.yaml` matches your structure

### Issue: "Module not found"
**Solution:** Ensure venv is activated:
```powershell
.\activate_env.ps1
```

---

## 📁 Project Structure

```
Code/
├── venv/                    # Virtual environment (created by setup)
├── setup/
│   └── install_dependencies.ps1  # Main setup script
├── data/
│   └── FOD-A/              # Dataset goes here (you provide)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb  # START HERE after setup
│   ├── 02_yolov8_training.ipynb
│   ├── 03_yolov11_training.ipynb
│   └── 04_sahi_inference.ipynb
├── configs/                # Model YAML configs (P2 variants)
├── utils/
│   └── dataset_loader.py   # Custom data loader
├── models/                 # Saved checkpoints (.pt files)
├── results/               # Plots, metrics, logs
├── activate_env.ps1       # Quick venv activation
└── requirements.txt       # All dependencies
```

---

## 🎯 Success Criteria

**Setup Complete When:**
- ✅ Virtual environment created
- ✅ PyTorch with CUDA installed (if GPU available)
- ✅ FOD-A dataset validated
- ✅ Jupyter notebook runs successfully
- ✅ Sample visualizations generated

**Ready for Training When:**
- ✅ All above completed
- ✅ `data.yaml` file exists
- ✅ GPU memory checked (≥8GB VRAM recommended)

---

## 📞 Next Steps

1. **Run setup script** → `.\setup\install_dependencies.ps1`
2. **Activate venv** → `.\activate_env.ps1`
3. **Open Jupyter** → `jupyter notebook`
4. **Run notebook** → `01_dataset_exploration.ipynb`
5. **Check TODO list** → See project root for tracking

---

**Last Updated:** December 18, 2025  
**Status:** Setup infrastructure complete, ready for dataset integration
