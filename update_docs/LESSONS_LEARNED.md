# Critical Lessons Learned - FOD Detection Project

**Date:** December 26, 2025  
**Status:** Mistakes Identified & Corrected

---

## 🚨 Critical Mistake #1: Wrong Image Resolution

### The Problem:
**Used `imgsz=1280` instead of paper's `imgsz=640`**

### Impact:
- **4× more pixels** per image (640×640 = 409K vs 1280×1280 = 1.64M)
- **4× more VRAM** usage at same batch size
- **4× slower** training per epoch
- **Artificially limited batch sizes** due to VRAM constraints

### Evidence:
| Model | @ 1280, Batch 16 | @ 640, Batch 16 | VRAM Difference |
|-------|------------------|-----------------|-----------------|
| YOLOv8m | 24.3GB (tight) | ~6GB | **-75% VRAM!** |
| YOLOv8m-P2 | OOM ❌ | ~13GB ✅ | Works at 640! |
| YOLOv11m | 26.9GB @ batch 14 ❌ | ~11GB ✅ | Works at 640! |
| YOLOv11m-P2 | OOM ❌ | ~13GB ✅ | Works at 640! |

### Root Cause:
- FOD-A dataset images are 1280×1280 resolution
- **Assumed** we must train at native resolution
- **Didn't check** Springer paper used standard 640×640
- YOLO models resize images anyway - no need for native resolution!

### Lesson Learned:
✅ **Always verify paper's exact training resolution**  
✅ **Higher resolution ≠ better results** (diminishing returns)  
✅ **640×640 is YOLO standard** for good reason (efficiency vs accuracy trade-off)

---

## 🚨 Critical Mistake #2: Inconsistent Batch Sizes

### The Problem:
**Each model trained with different batch sizes:**
- YOLOv8m: batch=16
- YOLOv8m-P2: batch=6
- YOLOv11m: batch=8
- YOLOv11m-P2: batch=6

### Impact:
- ❌ **Unfair comparison** (batch size affects training dynamics)
- ❌ **Different gradient stability** per model
- ❌ **Confounding variable** (can't isolate architecture effects)
- ❌ **Not paper-compliant** (paper used unified settings)

### Root Cause:
- Trained at 1280×1280, hit VRAM limits
- Reduced batch sizes reactively without unified strategy
- AdamW optimizer crashes led to conservative batch choices
- Didn't realize resolution was the real problem

### Lesson Learned:
✅ **Unified batch size = fair comparison**  
✅ **Test resolution impact BEFORE full training**  
✅ **Batch size consistency > maximum utilization**

---

## 🚨 Critical Mistake #3: YOLOv11m-P2 Config Bug

### The Problem:
**configs/yolo11-p2.yaml defaulted to NANO scale instead of MEDIUM**

### Evidence:
```yaml
# WRONG (what we had):
scales:
  n: [0.25, 0.50, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.50, 1.00, 512]   # Defined but NOT selected!
  # Missing: scale: m
  
# Result: Ultralytics assumed scale='n'
# Model: 2.7M params instead of 20M params!
```

### Impact:
- ❌ **All YOLOv11m-P2 training was INVALID** (90.62% mAP was nano, not medium!)
- ❌ **Compared 2.7M nano against 20M+ medium models** (apples to oranges)
- ❌ **Wasted 60+ hours** training wrong model
- ❌ **Incorrect batch size tests** (nano uses way less VRAM)

### Fix Applied:
```yaml
# CORRECT (fixed):
scale: m                # Explicit scale selection
depth_multiple: 0.50    # Medium depth
width_multiple: 1.00    # Medium width
max_channels: 512       # Medium channels
```

### Lesson Learned:
✅ **Always verify parameter count** after loading custom configs  
✅ **Print model summary** before training  
✅ **Explicit > implicit** in YAML configs  
✅ **Compare params against known baselines** (20M for yolo11m)

---

## 🚨 Mistake #4: Gradient Accumulation Assumption

### The Problem:
**Assumed Ultralytics YOLO supports gradient accumulation**

### Reality:
```python
# REMOVED in Ultralytics 8.3.x
model.train(
    batch=8,
    accumulate=2,  # ❌ NOT SUPPORTED
    ...
)
# SyntaxError: 'accumulate' is not a valid YOLO argument
```

### Impact:
- Spent time researching workarounds (modify source, custom training loop)
- All alternatives too complex or break features
- **Should have tested resolution first** instead

### Lesson Learned:
✅ **Check API docs for supported parameters**  
✅ **Test simple solutions first** (resolution change vs complex code)  
✅ **Don't over-engineer** when simple fix exists

---

## 🚨 Mistake #5: Transfer Learning Terminology Confusion

### The Problem:
**Called COCO-pretrained training "From Scratch"**

### What Actually Happened:
| Our Name | What It Was | Paper's Method? |
|----------|-------------|-----------------|
| "From Scratch" | COCO pretrained + P2 config | ✅ YES - Standard! |
| "Transfer Learning" | FOD-baseline weights → P2 | ❌ NO - Experimental |

### Evidence:
- "From Scratch" 92.88% > "Transfer" 91.10%
- Transfer used lr=0.0001 (too low for P2 head)
- Transfer loaded wrong weights (FOD-tuned, not COCO)

### Lesson Learned:
✅ **"From scratch" means random init, not pretrained**  
✅ **Paper's "transfer learning" = COCO pretrained (standard practice)**  
✅ **Use consistent terminology** to avoid confusion

---

## 🚨 Mistake #6: Incorrect File Paths

### The Problem:
**Used `./models/yolov8m.pt` which was corrupted**

### Evidence:
```python
weights: './models/yolov8m.pt'
# Error: invalid load key, 'v'
```

### Fix:
```python
weights: 'yolov8m.pt'  # Let Ultralytics auto-download
```

### Lesson Learned:
✅ **Use Ultralytics standard paths** (auto-download is reliable)  
✅ **Verify file integrity** if using local weights  
✅ **Test loading before full training**

---

## ✅ Corrected Configuration (Final)

### Paper-Compliant Settings:
```python
# Model Loading (All models)
YOLOv8m:     YOLO('yolov8m.pt')              # COCO pretrained
YOLOv8m-P2:  YOLO('configs/yolov8-p2.yaml')  # P2 arch
             .load('yolov8m.pt')              # + COCO weights
YOLOv11m:    YOLO('yolo11m.pt')              # COCO pretrained
YOLOv11m-P2: YOLO('configs/yolo11-p2.yaml')  # P2 arch (FIXED!)
             .load('yolo11m.pt')              # + COCO weights

# Training Hyperparameters (Springer Paper)
epochs = 300                    # Paper's standard
imgsz = 640                     # ✅ FIXED (was 1280)
batch = 16                      # ✅ UNIFIED (was 6-16 mixed)
lr0 = 0.01                      # Paper's LR (no scaling needed!)
optimizer = 'SGD'               # Paper's optimizer
momentum = 0.937                # Paper's momentum
weight_decay = 0.0005           # Paper's weight decay
warmup_epochs = 3.0             # Paper's warmup
workers = 8                     # ✅ INCREASED (was 4)

# Augmentation (Paper's settings)
mosaic = 1.0
mixup = 0.1                     # Paper uses 0.1
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 0.0
translate = 0.1
scale = 0.5
shear = 0.0
perspective = 0.0
flipud = 0.0
fliplr = 0.5
```

### VRAM Usage @ 640×640, Batch 16:
- YOLOv8m Baseline: ~6GB (75% reduction from 1280!)
- YOLOv8m-P2: ~13GB (was OOM at 1280)
- YOLOv11m Baseline: ~11GB (was OOM at 1280)
- YOLOv11m-P2: ~13GB (was OOM at 1280)

**All models fit comfortably with 11-18GB headroom!**

---

## 📊 Impact Summary

### Wasted Effort:
- ❌ 60+ hours training YOLOv11m-P2 with wrong config (nano instead of medium)
- ❌ 40+ hours training at 1280 resolution (4× slower than needed)
- ❌ Multiple batch size tests at wrong resolution
- ❌ Gradient accumulation research (unnecessary with correct resolution)

### Time Saved Going Forward:
- ✅ 640×640 = **4× faster training** per epoch
- ✅ Batch 16 unified = **no more OOM troubleshooting**
- ✅ Workers=8 = **faster data loading**
- ✅ Paper-compliant = **reproducible results**

### Estimated Training Time (100 epochs):
- **@ 1280, batch 6:** ~18-20 hours per model
- **@ 640, batch 16:** ~4-6 hours per model
- **Speedup:** ~4× faster!

---

## 🎯 Key Takeaways

1. **Always verify paper's EXACT hyperparameters** (especially resolution!)
2. **Test quick before committing** to 100-300 epoch training
3. **Validate model configs** (check parameter count)
4. **Unified settings = fair comparison**
5. **Standard resolution (640) is standard for a reason**
6. **Higher resolution ≠ better results** (diminishing returns, huge cost)
7. **Let framework handle downloads** (don't fight the tooling)
8. **Use correct terminology** (COCO pretrained, not "from scratch")

---

## 📝 Corrective Actions Taken

- ✅ Fixed configs/yolo11-p2.yaml (explicit scale='m')
- ✅ Changed imgsz from 1280 → 640
- ✅ Unified batch size to 16 for all models
- ✅ Using paper's exact hyperparameters
- ✅ Increased workers from 4 → 8
- ✅ Using Ultralytics standard weight paths
- ✅ Created quick_batch_test.py for rapid validation
- ✅ Documented all mistakes for future reference

---

**Status:** Ready to retrain all models with corrected settings  
**Expected:** 4× faster training, fair comparison, paper-compliant results  
**Confidence:** High (all tests passed at 640, batch 16)
