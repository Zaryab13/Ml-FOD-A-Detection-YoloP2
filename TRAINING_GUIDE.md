# FOD Detection Training Guide

## Training Setup Complete! 🎉

You now have everything ready to train YOLOv8 models on the FOD-A dataset.

## Available Training Scripts

### 1. YOLOv8 Base Model (Baseline)
```powershell
# Train standard YOLOv8n
.\venv\Scripts\Activate.ps1
python train_yolov8_base.py
```

**Features:**
- Standard YOLOv8n architecture (3 detection heads: P3/8, P4/16, P5/32)
- Pretrained weights on COCO dataset
- Fast training (~2-3 hours on RTX 4090)
- Expected mAP: ~92-93%

### 2. YOLOv8-P2 Model (Optimized for Small Objects)
```powershell
# Train YOLOv8-P2 with extra P2 detection head
.\venv\Scripts\Activate.ps1
python train_yolov8_p2.py
```

**Features:**
- Custom architecture with 4 detection heads (P2/4, P3/8, P4/16, P5/32)
- P2 head specifically for small objects (<32px)
- Training from scratch with custom config
- Expected mAP: ~94-95% (target: beat 93.8% baseline)

## Training Configuration

### Key Parameters
- **Image Size:** 1280×1280 (higher resolution for small objects)
- **Batch Size:** 16 (adjust based on GPU memory)
- **Epochs:** 100 (with early stopping patience=20)
- **Augmentation:** Heavy (Mosaic, MixUp, CopyPaste)
- **Device:** GPU 0 (CUDA 12.4)

### Hyperparameters (configs/hyp_fod.yaml)
- Learning rate: 0.01 (SGD)
- Weight decay: 0.0005
- Box loss weight: 7.5 (higher for small objects)
- Cosine LR scheduler
- Warmup: 3 epochs

## Expected Training Time

**On NVIDIA RTX 4090 (24GB):**
- YOLOv8n Base: ~2-3 hours (100 epochs)
- YOLOv8n-P2: ~3-4 hours (100 epochs, more complex architecture)

**Note:** Training will automatically use GPU if available. Check GPU usage with:
```powershell
nvidia-smi
```

## Output Structure

Training results will be saved to:
```
fod_detection/
├── yolov8n_baseline_YYYYMMDD_HHMMSS/
│   ├── weights/
│   │   ├── best.pt      # Best model by mAP
│   │   └── last.pt      # Last epoch checkpoint
│   ├── results.csv      # Training metrics per epoch
│   ├── results.png      # Training curves
│   ├── confusion_matrix.png
│   ├── F1_curve.png
│   ├── PR_curve.png
│   └── val_batch*.jpg   # Validation predictions
│
└── yolov8n-p2_YYYYMMDD_HHMMSS/
    └── ... (same structure)
```

## Monitoring Training

### Real-time Metrics
The training script will display:
- Current epoch and batch progress
- Loss values (box, cls, dfl)
- mAP50 and mAP50-95
- Precision and Recall
- GPU memory usage

### TensorBoard (Optional)
```powershell
tensorboard --logdir fod_detection
```

## Quick Start Commands

### Option 1: Train Baseline First (Recommended)
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Train baseline YOLOv8n
python train_yolov8_base.py

# Wait for completion (~2-3 hours)
# Check results in fod_detection/yolov8n_baseline_*/
```

### Option 2: Train P2 Model Directly
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Train YOLOv8-P2
python train_yolov8_p2.py

# Wait for completion (~3-4 hours)
# Check results in fod_detection/yolov8n-p2_*/
```

### Option 3: Train Both (Sequential)
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Train baseline
python train_yolov8_base.py

# Then train P2
python train_yolov8_p2.py
```

## Model Sizes (Advanced)

You can also train larger models for better accuracy:

**Edit the training scripts to change MODEL_SIZE:**
- `yolov8n` - Nano (fastest, ~3M params)
- `yolov8s` - Small (~11M params)
- `yolov8m` - Medium (~25M params)
- `yolov8l` - Large (~43M params)
- `yolov8x` - Extra Large (~68M params)

## Tips for Best Results

1. **Monitor GPU temperature and utilization:**
   ```powershell
   nvidia-smi -l 1  # Update every 1 second
   ```

2. **Adjust batch size if out of memory:**
   - Edit `BATCH_SIZE` in training scripts
   - Try: 16 → 8 → 4
   - Lower batch size = slower training but same accuracy

3. **Use early stopping:**
   - Training stops if no improvement for 20 epochs
   - Saves time and prevents overfitting

4. **Check validation images:**
   - Look at `val_batch*.jpg` in results folder
   - Verify model is detecting small objects

## Next Steps After Training

1. **Evaluate models:**
   ```python
   from ultralytics import YOLO
   model = YOLO('fod_detection/yolov8n_baseline_*/weights/best.pt')
   results = model.val()
   ```

2. **Compare baseline vs P2:**
   - Compare mAP50-95 scores
   - Check AP_small specifically
   - Analyze confusion matrices

3. **Run inference:**
   ```python
   model = YOLO('fod_detection/yolov8n-p2_*/weights/best.pt')
   results = model.predict('path/to/image.jpg', imgsz=1280)
   ```

4. **Week 2: Implement SAHI:**
   - Sliced inference for even better small object detection
   - Coming next after baseline training!

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce BATCH_SIZE (16 → 8 → 4)

### Issue: Training too slow
**Solution:** Check GPU is being used (`nvidia-smi`), reduce image size (1280 → 960)

### Issue: mAP not improving
**Solution:** Train longer, check data augmentation settings, verify dataset quality

### Issue: Model not detecting small objects
**Solution:** Use YOLOv8-P2 instead of base, increase image size to 1280+

## Ready to Start?

```powershell
# Let's go! 🚀
.\venv\Scripts\Activate.ps1
python train_yolov8_base.py
```

Expected target: **mAP50 > 94%** (beating paper's 93.8% baseline)
