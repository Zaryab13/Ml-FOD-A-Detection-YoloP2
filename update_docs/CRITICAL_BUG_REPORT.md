# CRITICAL BUG REPORT: YOLOv11-P2 Configuration Error

**Date:** December 25, 2025  
**Severity:** CRITICAL  
**Impact:** All YOLOv11-P2 training was done with WRONG model size

---

## Summary

**All YOLOv11m-P2 training runs used YOLOv11n-P2 (nano) instead of YOLOv11m-P2 (medium)!**

This explains why:
- YOLOv11m-P2 used only 16.1GB VRAM at batch 16 (should be ~25-30GB)
- YOLOv11m-P2 had only 2.7M parameters (should be ~20M)
- YOLOv11m-P2 trained faster than expected
- Batch size tests were completely invalid

---

## Evidence

### Model Parameter Comparison:

| Model | Expected Params | Actual Params | Ratio | Issue |
|-------|----------------|---------------|-------|-------|
| **YOLOv11m Baseline** | 20.1M | 20.1M | ✅ 100% | Correct |
| **YOLOv8m-P2** | ~43M | 42.9M | ✅ 99.8% | Correct |
| **YOLOv11m-P2** | **~20M** | **2.7M** | ❌ **13%** | **NANO SIZE!** |

### VRAM Usage @ Batch 16:

| Model | VRAM Usage | Expected | Status |
|-------|-----------|----------|--------|
| YOLOv11m Baseline | 30.8GB | ~30GB | ✅ Correct (OOM on 24GB) |
| YOLOv8m-P2 | OOM | ~35-40GB | ✅ Correct (OOM) |
| **YOLOv11m-P2** | **16.1GB** | **~25-30GB** | ❌ **Was using nano!** |

### Console Warnings:

```
WARNING no model scale passed. Assuming scale='n'.
YOLO11-p2 summary: 216 layers, 2,681,064 parameters  ← NANO SIZE!
```

**YOLOv11m should have ~20M parameters, not 2.7M!**

---

## Root Cause

### Problem in configs/yolo11-p2.yaml:

The YAML defined multiple scales but didn't specify which to use:

```yaml
scales: # model compound scaling constants
  n: [0.50, 0.25, 1024]  # YOLOv11n
  s: [0.50, 0.50, 1024]  # YOLOv11s
  m: [0.50, 1.00, 512]   # YOLOv11m  ← We wanted this!
  l: [1.00, 1.00, 512]   # YOLOv11l
  x: [1.00, 1.50, 512]   # YOLOv11x
```

**Ultralytics defaults to scale='n' when not explicitly specified!**

---

## Fix Applied

Updated `configs/yolo11-p2.yaml` to explicitly use medium scale:

```yaml
# NEW (FIXED):
nc: 41
scale: m              # EXPLICIT: Use medium scale
depth_multiple: 0.50  # YOLOv11m depth
width_multiple: 1.00  # YOLOv11m width  
max_channels: 512     # YOLOv11m max channels
```

---

## Impact on Previous Results

### All YOLOv11m-P2 Training Runs Are INVALID:

| Run | Date | Batch | Result | Issue |
|-----|------|-------|--------|-------|
| Original YOLOv11m-P2 | Dec 24 | 6 | 90.62% mAP | ❌ Was nano, not medium |
| Batch size tests | Dec 25 | 12,14,16 | All passed | ❌ Tests used nano model |
| Batch 16 script | Dec 25 | Not run | - | ❌ Would have used nano |

**None of these results are comparable to YOLOv11m baseline (20M params)!**

---

## Required Actions

### 1. ✅ Fix Configuration
- [x] Update `configs/yolo11-p2.yaml` with explicit scale='m'
- [x] Verify model loads with ~20M parameters

### 2. 🔄 Re-test Batch Sizes
- [ ] Run `test_yolo11m_p2_memory.py` with CORRECTED config
- [ ] Find actual maximum batch size for TRUE YOLOv11m-P2
- [ ] Expected: Much lower than batch 16 (probably 6-10)

### 3. 🔄 Update Training Scripts
- [ ] Verify all scripts load corrected config
- [ ] Update batch sizes based on new memory tests
- [ ] Update expected VRAM usage in documentation

### 4. 🔄 Retrain Models
- [ ] Retrain YOLOv11m-P2 with CORRECT medium scale
- [ ] Use corrected optimal batch size
- [ ] Compare with YOLOv11m baseline (both 20M params now)

### 5. 🔄 Update Documentation
- [ ] Update BATCH_SIZE_AUDIT.md with corrected findings
- [ ] Update MODEL_COMPARISON_REPORT.md (invalidate old YOLOv11m-P2 results)
- [ ] Document lessons learned

---

## Corrected Expectations

### True YOLOv11m-P2 Memory Profile:

| Batch Size | Estimated VRAM | Status | Reasoning |
|-----------|---------------|--------|-----------|
| 6 | ~18-20GB | ✅ Likely safe | Was 16.1GB with nano, medium adds ~25% |
| 8 | ~24-26GB | ⚠️ Risky | Might fit, minimal margin |
| 10 | ~28-30GB | ❌ OOM | Will exceed 24GB |
| 12+ | >30GB | ❌ OOM | Definitely too large |

**Predicted optimal batch: 6-8 (not 16!)**

---

## Comparison: Nano vs Medium Scale

### YOLOv11n-P2 (What We Accidentally Trained):
- Parameters: 2.7M
- GFLOPs: 10.6
- Batch 16 VRAM: 16.1GB
- Training time: Fast
- Performance: 90.62% mAP (but unfair comparison!)

### YOLOv11m-P2 (What We Should Train):
- Parameters: ~20M (7.4× larger!)
- GFLOPs: ~68-70 (6.5× more compute)
- Batch 16 VRAM: ~30GB (OOM)
- Optimal batch: 6-8 (estimated)
- Performance: Unknown (needs retraining)

---

## Lessons Learned

1. **Always verify model size** after loading custom configs
2. **Check parameter counts** match expected architecture
3. **Warnings matter** - "Assuming scale='n'" was a red flag
4. **Memory usage should match model size** - 16GB for "medium" was suspicious
5. **Compare apples to apples** - nano vs medium is not a fair comparison

---

## Next Steps

1. ✅ Configuration fixed
2. **NOW:** Run `python test_yolo11m_p2_memory.py` to find real optimal batch
3. Update all training scripts with corrected batch sizes
4. Retrain YOLOv11m-P2 with correct configuration
5. Update all documentation and comparison reports

---

**Status:** Config fixed, awaiting memory retesting  
**Priority:** CRITICAL - blocks all YOLOv11-P2 work  
**Owner:** Model Training Team
