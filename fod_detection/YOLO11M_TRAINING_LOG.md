# YOLOv11m Baseline Training Log & Research Notes

**Date:** December 23, 2025  
**Model:** YOLOv11m (Medium)  
**Dataset:** FOD-A (Foreign Object Debris)  
**Hardware:** NVIDIA RTX 4090 (24GB)

---

## 1. Final Configuration (Stable)
These are the settings used for the successful training run (after fixing initial issues).

*   **Image Size:** 1280 x 1280
*   **Batch Size:** 8
*   **Optimizer:** SGD (Switched from AdamW due to instability)
*   **Learning Rate:** 0.005 (Conservative SGD start)
*   **System Settings:** `workers=4`, `cache=False` (Windows Optimized)

---

## 2. Critical Research Notes (For Paper)

### A. Batch Size Discrepancy
*   **Observation:** We used **Batch 8** for YOLOv11m, whereas YOLOv8m Baseline used **Batch 16**.
*   **Reason:** The YOLOv11m architecture is more memory-intensive. At Batch 16, VRAM usage exceeded 24GB, causing the OS to swap to system RAM (Shared GPU Memory). This slowed training from **0.2s/it** to **21s/it**.
*   **Justification:** Hyperparameters were optimized per-architecture to maximize hardware utilization. The impact on generalization is considered minor compared to architectural differences.

### B. Stability & Optimizer Choice
*   **Observation:** YOLOv11m was unstable with `AdamW` optimizer at Batch 8.
*   **Event:** Training diverged (Loss = NaN/Inf) multiple times (Epoch 24, then Epoch 29).
*   **Analysis:** `AdamW` calculates adaptive learning rates based on gradient variance. With a small batch size (8), the variance estimates are noisy, leading to "exploding gradients" where weights shoot to Infinity.
*   **Fix:** Switched to **SGD (Stochastic Gradient Descent)**. SGD is less sensitive to batch noise and generally more robust for YOLO training, though it may converge slightly slower.

---

## 3. Training History & Troubleshooting Log

### Phase 1: Performance Optimization
*   **Initial Attempt:** Batch 16, `cache=True`, `workers=20`.
*   **Issue:** Extremely slow training (~12.6s/iteration).
*   **Diagnosis:** VRAM spillover + Windows shared memory limits.
*   **Solution:** Reduced Batch to 8, `workers` to 4, disabled RAM caching.

### Phase 2: The Crash (Epoch 24)
*   **Run ID:** `yolo11m_baseline_20251222_223346`
*   **Event:** Loss became `NaN` (Not a Number).
*   **Cause:** `AdamW` optimizer instability with high LR (0.01).

### Phase 3: The Second Crash (Epoch 29)
*   **Attempt:** Resumed with `AdamW` and reduced LR (0.001).
*   **Event:** Loss became `NaN` again after 9 epochs.
*   **Conclusion:** `AdamW` is fundamentally unstable for this specific Model/Batch/Dataset combination.

### Phase 4: The Final Fix (Switch to SGD)
*   **Strategy:** Switched optimizer to `SGD` with `lr0=0.005`.
*   **Reasoning:** SGD is the industry standard for stable Object Detection training when batch sizes are small.
