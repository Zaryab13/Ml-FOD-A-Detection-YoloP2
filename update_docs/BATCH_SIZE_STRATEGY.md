# Batch Size Optimization Strategy
**Goal:** Find optimal batch size for fair comparison across all models on 24GB GPU

---

## Problem Statement

Current models were trained with **inconsistent batch sizes**, making fair comparison impossible:

| Model | Batch Size | mAP@50-95 | Issue |
|-------|-----------|-----------|-------|
| YOLOv8m Baseline | 16 | 93.31% | ✓ Good GPU utilization |
| YOLOv8m-P2 | 6 | 92.88% | ⚠️ Low GPU utilization |
| YOLOv11m Baseline | 8 | 86.87% | ⚠️ Low GPU utilization |
| YOLOv11m-P2 | 6 | 90.62% | ⚠️ Low GPU utilization |

**Confounding Variable:** Batch size affects:
- Gradient noise/stability
- Learning rate scaling
- Training dynamics
- Final convergence

**Cannot determine if performance differences are due to:**
- Architecture improvements (P2 head)
- Model capacity (YOLOv8 vs YOLOv11)
- **OR batch size differences**

---

## Solution: Unified Batch Size

### Step 1: Find Maximum Stable Batch
**Script:** `find_optimal_batch.py`

Test YOLOv11m-P2 (most memory-intensive) at:
- Batch 12 (expected ~12-13GB)
- Batch 14 (expected ~14-15GB)
- Batch 16 (expected ~16-17GB)
- Batch 18 (expected ~18-19GB)
- Batch 20 (expected ~20-21GB)

**Safety Criterion:** Maintain 2-4GB free VRAM for stability

---

### Step 2: Current VRAM Evidence

From batch=8 test (YOLOv11m-P2):
```
Epoch 1/100    GPU_mem 8.1G    ...
```

**Extrapolation:**
- Batch 8 → 8.1 GB
- Batch 12 → **~12.2 GB** (1.5× scaling)
- Batch 16 → **~16.3 GB** (2× scaling)
- Batch 20 → **~20.4 GB** (2.5× scaling)

**Prediction:** Batch 16 is likely maximum stable size (16.3GB + 4GB overhead ≈ 20GB < 24GB)

---

### Step 3: Retrain All Models

Once optimal batch found (likely **batch=12 or 16**), retrain ALL models:

#### Training Scripts to Create:
1. `train_yolov8m_baseline_batchX.py`
2. `train_yolov8m_p2_batchX.py`
3. `train_yolov11m_baseline_batchX.py`
4. `train_yolov11m_p2_batchX.py`

Where X = optimal batch size

#### Hyperparameters (Unified):
```python
batch = X                    # Same for ALL models
lr0 = 0.01 × (X / 16)       # Scale learning rate proportionally
momentum = 0.937             # Standard
weight_decay = 0.0005        # Standard
optimizer = 'SGD'            # Stable for all models
imgsz = 1280                 # Small object detection
epochs = 100                 # Full training
workers = 4                  # Windows optimization
cache = False                # Windows optimization
```

**Loss Weights (Optimized for Small Objects):**
```python
box = 10.0                   # High weight for small bounding boxes
cls = 0.5                    # Standard classification
dfl = 2.0                    # Distribution Focal Loss for localization
```

---

## Expected Outcomes

### Scenario A: Optimal Batch = 16
**Implications:**
- All models can train at batch=16 (fair comparison)
- YOLOv8m Baseline retrain may **improve** (already at batch=16)
- YOLOv8m-P2 will **significantly improve** (6→16, +167% batch increase)
- YOLOv11m Baseline will **improve** (8→16, +100% batch increase)
- YOLOv11m-P2 will **significantly improve** (6→16, +167% batch increase)

**Expected Performance Shifts:**
- YOLOv8m Baseline: 93.31% → **93.5-94.0%** (minor improvement)
- YOLOv8m-P2: 92.88% → **93.5-94.5%** (major improvement from larger batch)
- YOLOv11m Baseline: 86.87% → **88.0-89.0%** (major improvement)
- YOLOv11m-P2: 90.62% → **91.5-92.5%** (major improvement)

### Scenario B: Optimal Batch = 12
**Implications:**
- Conservative choice (safer for long training)
- YOLOv8m Baseline may **decrease slightly** (16→12, -25% batch)
- Others will improve from larger batch

**Expected Performance Shifts:**
- YOLOv8m Baseline: 93.31% → **92.8-93.2%** (slight decrease)
- YOLOv8m-P2: 92.88% → **93.2-93.8%** (improvement from 6→12)
- YOLOv11m Baseline: 86.87% → **87.5-88.5%** (improvement from 8→12)
- YOLOv11m-P2: 90.62% → **91.0-91.8%** (improvement from 6→12)

---

## Decision Criteria

### Prioritize Batch 16 IF:
- ✅ VRAM usage ≤ 20GB during test
- ✅ No OOM errors during 1-epoch test
- ✅ At least 2GB safety margin
- ✅ Training speed acceptable

### Use Batch 12 IF:
- ⚠️ Batch 16 uses >21GB VRAM
- ⚠️ OOM errors occur
- ⚠️ Safety margin <2GB
- ⚠️ Need stability over maximum performance

---

## Timeline

| Step | Task | Duration | Output |
|------|------|----------|--------|
| 1 | Run `find_optimal_batch.py` | ~30 min | Optimal batch size (12, 14, or 16) |
| 2 | Create 4 training scripts | ~30 min | Unified training configs |
| 3 | Train YOLOv8m Baseline | ~30 hours | Baseline at optimal batch |
| 4 | Train YOLOv8m-P2 | ~90 hours | P2 at optimal batch |
| 5 | Train YOLOv11m Baseline | ~16 hours | Baseline at optimal batch |
| 6 | Train YOLOv11m-P2 | ~12 hours | P2 at optimal batch |
| 7 | Update comparison report | ~1 hour | Fair performance analysis |

**Total Time:** ~150 hours (can parallelize on multiple GPUs if available)

---

## Fair Comparison Checklist

Once all models retrained at unified batch size:

- [ ] All models use **same batch size**
- [ ] All models use **same image size** (1280)
- [ ] All models use **same optimizer** (SGD)
- [ ] All models use **same epochs** (100)
- [ ] All models use **same loss weights** (box=10, dfl=2)
- [ ] All models use **same augmentation** settings
- [ ] All models trained on **same hardware** (RTX 4090 24GB)
- [ ] All models trained on **same dataset split** (FOD-A)

**Only then can we make valid conclusions about:**
1. YOLOv8 vs YOLOv11 architecture differences
2. P2 head effectiveness for small object detection
3. Model capacity vs performance trade-offs

---

## Current Action

**Run batch size finder:**
```bash
python find_optimal_batch.py
```

This will test YOLOv11m-P2 at progressively larger batches and recommend the optimal size.

---

**Last Updated:** December 25, 2025  
**Status:** Ready to find optimal batch size
