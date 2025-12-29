# Final Batch Size Strategy & Training Plan

**Date:** December 25, 2025  
**Status:** Ready to Execute  
**Hardware:** RTX 4090 24GB VRAM

---

## Recommended Batch Sizes (Based on Testing & Evidence)

| Model | Optimal Batch | VRAM Usage | Parameters | Evidence |
|-------|---------------|------------|------------|----------|
| **YOLOv8m Baseline** | **16** | ~17GB | 25.9M | ✅ Tested successfully |
| **YOLOv8m-P2** | **8** | ~22GB | 42.9M | Predicted (OOM @ 16) |
| **YOLOv11m Baseline** | **12** | ~24GB | 20.1M | Predicted (OOM @ 16, 30.8GB) |
| **YOLOv11m-P2** | **8** | ~22GB | ~20M | Needs retest with corrected config |

---

## Batch Size Determination Logic

### YOLOv8m Baseline: batch=16 ✅
- **Evidence:** Trained successfully at batch 16
- **VRAM:** Comfortable margin
- **Status:** Confirmed optimal

### YOLOv8m-P2: batch=8 📊
- **Evidence:** OOM at batch 16 (42.9M params, 205.9 GFLOPs)
- **Previous:** Trained at batch 6 successfully
- **Estimate:** batch 8 likely optimal (42M params = 1.6× YOLOv8m baseline)
- **Calculation:** 16 / 1.6 ≈ 10, conservative = 8

### YOLOv11m Baseline: batch=12 📊
- **Evidence:** OOM at batch 16 (30.8GB VRAM)
- **Previous:** Trained at batch 8 (after AdamW crash at 16)
- **Estimate:** batch 12-14 likely optimal
- **Calculation:** 30.8GB / 24GB = 1.28×, so 16/1.28 ≈ 12

### YOLOv11m-P2: batch=8 ⚠️
- **Evidence:** Previous batch 6 was WRONG MODEL (was nano!)
- **Prediction:** Similar to YOLOv8m-P2 (~20M params + P2 head)
- **Estimate:** batch 8 likely optimal
- **Status:** **MUST RETEST** with corrected config

---

## Learning Rate Scaling

Scale learning rate proportionally to batch size:

```python
base_lr = 0.01  # For batch 16
scaled_lr = base_lr × (actual_batch / 16)
```

| Model | Batch | Scaled LR |
|-------|-------|-----------|
| YOLOv8m Baseline | 16 | 0.01000 |
| YOLOv8m-P2 | 8 | 0.00500 |
| YOLOv11m Baseline | 12 | 0.00750 |
| YOLOv11m-P2 | 8 | 0.00500 |

---

## Why Different Batch Sizes?

### Technical Reasons:
1. **Parameter count varies:** 20M (YOLOv11m) to 43M (YOLOv8m-P2)
2. **Architecture overhead:** P2 head adds extra detection layer
3. **Memory patterns:** C2PSA (YOLOv11) vs C2f (YOLOv8) have different VRAM profiles
4. **GPU limitation:** 24GB VRAM is fixed constraint

### Is This Fair?
**Yes, with caveats:**
- ✅ Each model operates at optimal efficiency
- ✅ Maximizes GPU utilization
- ✅ Real-world scenario (you'd optimize each model individually)
- ⚠️ Batch size IS a variable (but documented)
- ⚠️ Focus on architecture comparison, not absolute mAP

### Mitigation:
1. **Document clearly:** State batch sizes in all results
2. **Scale LR:** Compensate for batch differences
3. **Acknowledge:** Batch size affects training dynamics
4. **Focus:** Compare architecture gains (P2 vs baseline), not absolute numbers

---

## Gradient Accumulation Alternative

### Problem:
Ultralytics doesn't support native gradient accumulation.

### Solution:
Use maximum stable batch per model instead:
- **YOLOv8m Baseline:** batch=16 (no accumulation needed)
- **YOLOv8m-P2:** batch=8 (≈50% of ideal batch 16)
- **YOLOv11m Baseline:** batch=12 (≈75% of ideal batch 16)
- **YOLOv11m-P2:** batch=8 (≈50% of ideal batch 16)

**Impact:** Slight suboptimal gradient stability, but minimal effect on final mAP.

---

## Training Configuration Summary

### YOLOv8m Baseline
```python
model.train(
    data='data/FOD-A/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=16,           # ✅ Optimal
    lr0=0.01,          # Base LR
    optimizer='SGD',
    device=0,
    workers=8
)
```

### YOLOv8m-P2
```python
model.train(
    data='data/FOD-A/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,           # ✅ Predicted optimal
    lr0=0.005,         # Scaled: 0.01 × (8/16)
    optimizer='SGD',
    device=0,
    workers=8
)
```

### YOLOv11m Baseline
```python
model.train(
    data='data/FOD-A/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=12,          # ✅ Predicted optimal
    lr0=0.0075,        # Scaled: 0.01 × (12/16)
    optimizer='SGD',
    device=0,
    workers=8
)
```

### YOLOv11m-P2
```python
model.train(
    data='data/FOD-A/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=8,           # ⚠️ Needs confirmation
    lr0=0.005,         # Scaled: 0.01 × (8/16)
    optimizer='SGD',
    device=0,
    workers=8
)
```

---

## Validation Plan

### Step 1: Quick Memory Test (5 minutes)
Test predicted batches with 1 epoch:
```bash
python test_yolov8m_p2_memory.py  # Test batch 8, 10
python test_yolo11m_baseline_memory.py  # Test batch 12, 14
python test_yolo11m_p2_memory.py  # Test batch 6, 8, 10 (CORRECTED config)
```

### Step 2: Confirm Optimal Batches
- YOLOv8m-P2: Aim for batch 8-10 with <2GB VRAM margin
- YOLOv11m Baseline: Aim for batch 12-14 with <2GB margin
- YOLOv11m-P2: Aim for batch 8 (conservative after config fix)

### Step 3: Update Training Scripts
Modify all training scripts with confirmed batches.

### Step 4: Full Training (100 epochs each)
Train all 4 models with optimized settings.

---

## Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Memory testing (3 models) | 15 min | ⏳ Pending |
| Script updates | 10 min | ⏳ Pending |
| YOLOv8m Baseline (batch 16) | Already done | ✅ Complete |
| YOLOv8m-P2 (batch 8) | 12-15 hours | ⏳ Pending |
| YOLOv11m Baseline (batch 12) | 12-15 hours | ⏳ Pending |
| YOLOv11m-P2 (batch 8) | 12-15 hours | ⏳ Pending |

**Total:** ~40-45 hours training time

---

## Next Actions

1. ✅ **Fix YOLOv11-P2 config** - Done
2. ✅ **Gradient accumulation research** - Done (not supported)
3. ⏳ **Quick memory tests** - Run 3 test scripts
4. ⏳ **Update training scripts** - Set confirmed batches
5. ⏳ **Train models** - 100 epochs each
6. ⏳ **Compare results** - Analyze architecture effects

---

## Success Criteria

### Training Success:
- ✅ All 4 models train to completion (100 epochs)
- ✅ No OOM errors
- ✅ VRAM usage within safe margin (<22GB)
- ✅ Convergence observed (mAP plateaus)

### Comparison Success:
- ✅ Batch sizes documented and justified
- ✅ Learning rates scaled appropriately
- ✅ Architecture effects isolated (P2 vs baseline)
- ✅ YOLOv8 vs YOLOv11 comparison meaningful

---

**Current Status:** Strategy defined, awaiting execution  
**Blocker:** Need YOLOv11m-P2 memory test with corrected config  
**Recommendation:** Run quick memory tests, then proceed with training
