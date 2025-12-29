# YOLOv11m-P2 Optimization Plan
**Goal:** Improve mAP@50-95 from 90.62% to 92%+ while maximizing 24GB VRAM utilization

---

## Phase 1: Gradient Accumulation (Low Risk) ⭐ START HERE

### Objective
Simulate larger batch size (18) while maintaining VRAM usage at batch 6

### Implementation
**Script:** `train_yolo11m_p2_optimized.py`

### Key Changes
```python
batch=6              # Physical batch (fits in VRAM)
accumulate=3         # Effective batch = 6 × 3 = 18
lr0=0.011           # Scaled from 0.01 for batch 18
box=10.0            # Increased from 7.5 (small object focus)
dfl=2.0             # Increased from 1.5 (better localization)
scale=0.7           # More aggressive (was 0.9)
translate=0.15      # More aggressive (was 0.1)
```

### Expected Results
- **Performance:** +1-2% mAP@50-95 (target: 91.5-92.5%)
- **Training Time:** Similar (~12 hours for 100 epochs)
- **VRAM Usage:** 16-18GB (same as current)
- **Stability:** High (no architecture changes)

### Success Criteria
- mAP@50-95 ≥ 91.5%
- No NaN/Inf divergence
- Training completes 100 epochs

### Decision Point
- **If successful (≥91.5%):** Document and deploy
- **If <91.5%:** Proceed to Phase 2

---

## Phase 2: Architecture Optimization (Medium Risk)

### Objective
Reduce memory by removing P5 head, increase batch size to 8-10

### Implementation
**Config:** Create `configs/yolo11-p2-lite.yaml`

### Architecture Changes
```yaml
# Remove P5 detection head (stride 32)
# Keep only P2 (stride 4), P3 (stride 8), P4 (stride 16)

head:
  # ... existing P2, P3, P4 blocks ...
  - [[19, 22, 25], 1, Detect, [nc]]  # 3 heads instead of 4
  # Remove layer 28 (P5 head)
```

### Rationale
- FOD objects are typically **8-64 pixels**
- P2 (stride 4) handles small objects (4-16px)
- P3 (stride 8) handles medium objects (16-32px)
- P4 (stride 16) handles large objects (32-64px)
- **P5 (stride 32) is redundant** for objects <64px

### Expected Results
- **Memory Saved:** ~15-20% (can increase batch to 8-10)
- **Performance:** -0.5 to +0.5% (minimal impact if objects <64px)
- **Training Speed:** +25-30% faster (larger batch, fewer feature maps)

### Training Settings
```python
batch=8              # Increased from 6
accumulate=2         # Effective batch = 16
lr0=0.01            # Standard for batch 16
```

### Success Criteria
- mAP@50-95 ≥ 90.5% (acceptable loss)
- VRAM usage ≤ 20GB
- Training speed ≥ 25% faster

### Risk Mitigation
- Test on validation set before full training
- Keep P5 config as fallback

---

## Phase 3: Hyperparameter Fine-Tuning (Low Risk)

### Objective
Squeeze final 0.5-1% performance through hyperparameter search

### Parameters to Tune
1. **Learning Rate Schedule**
   ```python
   lr0: [0.008, 0.01, 0.012]  # Test around baseline
   warmup_epochs: [3, 5, 7]   # Longer warmup for stability
   ```

2. **Loss Weights**
   ```python
   box: [8.0, 10.0, 12.0]     # Test higher weights
   dfl: [1.5, 2.0, 2.5]       # Test higher localization focus
   ```

3. **Augmentation Intensity**
   ```python
   scale: [0.6, 0.7, 0.8]     # Test more/less scale variation
   copy_paste: [0.3, 0.4, 0.5] # Test occlusion robustness
   ```

### Method
- Grid search (9 combinations max)
- Train for 20 epochs each
- Select best → full 100 epoch run

### Expected Results
- **Performance:** +0.5-1% mAP@50-95
- **Time Investment:** ~50 hours total (9 × 5.5 hours)

---

## Alternative: AdamW with Gradient Clipping

### Status
**NOT RECOMMENDED** based on previous crashes

### If Attempted Anyway
```python
optimizer='AdamW'
lr0=0.001            # 10× lower than SGD
weight_decay=0.01    # Higher for AdamW
warmup_epochs=7.0    # Longer warmup

# Manual gradient clipping (not supported in Ultralytics)
# Would require source code modification
```

### Why SGD is Better
- **Proven stable** at batch 6-8
- **Current results are good** (90.62% mAP)
- **AdamW crashed twice** (epochs 24, 29) on this dataset
- **Risk >> Reward**

---

## Execution Timeline

| Phase | Action | Duration | Risk | Expected Gain |
|-------|--------|----------|------|---------------|
| 1 | Gradient Accumulation | 12 hours | Low | +1-2% mAP |
| 2 | Remove P5 Head (if needed) | 10 hours | Medium | +25% speed |
| 3 | Hyperparameter Search | 50 hours | Low | +0.5-1% mAP |

**Total Time:** 22-72 hours depending on Phase 1 success

---

## Current Baseline Performance

| Model | mAP@50-95 | Batch | Training Time |
|-------|-----------|-------|---------------|
| YOLOv8m Baseline | **93.31%** | 16 | 29.4 hours |
| YOLOv11m-P2 (current) | 90.62% | 6 | 12.1 hours |
| YOLOv11m-P2 (target) | **92.0%+** | 6 (eff. 18) | ~12 hours |

---

## Decision Tree

```
Start
  ↓
Phase 1: Gradient Accumulation
  ↓
mAP ≥ 91.5%?
  ├─ YES → Deploy & Document ✓
  └─ NO → Phase 2: Remove P5 Head
           ↓
         mAP ≥ 90.5% AND 25%+ faster?
           ├─ YES → Deploy lite version ✓
           └─ NO → Phase 3: Hyperparameter Tuning
                    ↓
                  mAP ≥ 91.5%?
                    ├─ YES → Deploy optimized config ✓
                    └─ NO → Stick with YOLOv8m Baseline (93.31%)
```

---

## Files Created

1. **Training Scripts**
   - `train_yolo11m_p2_optimized.py` (Phase 1 - Gradient Accumulation)

2. **Configs** (To be created if needed)
   - `configs/yolo11-p2-lite.yaml` (Phase 2 - Remove P5)

3. **Documentation**
   - `OPTIMIZATION_PLAN.md` (this file)

---

## Next Immediate Action

**Run Phase 1:**
```bash
python fod_detection/train_yolo11m_p2_optimized.py
```

Monitor for:
- VRAM usage stays ≤ 18GB ✓
- No NaN/Inf losses ✓
- mAP@50 > 99.0% by epoch 50 (good sign)
- Final mAP@50-95 ≥ 91.5% (success threshold)

---

**Last Updated:** December 25, 2025  
**Status:** Phase 1 Ready for Execution
