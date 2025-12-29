"""
YOLOv8m-P2 Training - Paper-Compliant Configuration
Foreign Object Debris (FOD) Detection on FOD-A Dataset

Configuration:
- Model: YOLOv8m-P2 (P2 architecture + COCO pretrained backbone)
- Resolution: 640×640 (Springer paper standard)
- Batch: 16 (unified across all models)
- Epochs: 100 (paper standard)
- Optimizer: SGD (paper standard)
- Workers: 12 (optimized for RTX 4090)

VRAM Usage: ~13GB @ batch 16, 640×640
Training Time: ~20-24 hours (100 epochs)
"""

from ultralytics import YOLO
from datetime import datetime
import torch
import yaml
from pathlib import Path

if __name__ == '__main__':
    # Verify CUDA availability
    print(f"\n{'='*70}")
    print("SYSTEM CHECK")
    print(f"{'='*70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*70}\n")
    # Dataset configuration
    DATASET_CONFIG = "data/FOD-A/data.yaml"

    
# --- Springer Paper Training Hyperparameters ---
    EPOCHS = 100                  # Standard research duration for convergence
    IMGSZ = 640                   # Paper's standard input resolution (NOT 1280!)
    BATCH = 16                    # Unified batch size for fair comparison
    OPTIMIZER = 'SGD'             # Stochastic Gradient Descent with momentum
    LR0 = 0.01                    # Initial learning rate (no scaling needed!)
    LRF = 0.01                    # Final learning rate (cosine decay)
    MOMENTUM = 0.937              # SGD momentum factor
    WEIGHT_DECAY = 0.0005         # Optimizer weight decay
    WARMUP_EPOCHS = 3.0           # Initial stabilization period
    WARMUP_MOMENTUM = 0.8         # Momentum during warmup
    WARMUP_BIAS_LR = 0.1          # Learning rate for bias layers during warmup
    WORKERS = 12                 # Data loading workers (increased from 4)

    # --- Loss Function Weights (YOLOv8/v11 defaults) ---
    BOX = 7.5                     # Bounding box loss gain
    CLS = 0.5                     # Classification loss gain
    DFL = 1.5                     # Distribution Focal Loss gain

    # --- Data Augmentation (Standard FOD-A settings) ---
    MOSAIC = 1.0                  # Combine 4 images into 1
    MIXUP = 0.1                   # Alpha for image mixing (paper uses 0.1)
    HSV_H = 0.015                 # Image HSV-Hue augmentation
    HSV_S = 0.7                   # Image HSV-Saturation augmentation
    HSV_V = 0.4                   # Image HSV-Value augmentation
    DEGREES = 0.0                 # Image rotation (set to 0 for runway perspective)
    TRANSLATE = 0.1               # Image translation
    SCALE = 0.5                   # Image scale (+/- gain)
    SHEAR = 0.0                   # Image shear (+/- deg)
    PERSPECTIVE = 0.0             # Image perspective (+/- fraction)
    FLIPUD = 0.0                  # Image flip up-down (prob)
    FLIPLR = 0.5                  # Image flip left-right (prob)

    # Training configuration
    PROJECT_NAME = "fod_detection"
    RUN_NAME = f"yolov8m_p2_640_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*70}")
    print("YOLOV8M-P2 - PAPER-COMPLIANT TRAINING")
    print(f"{'='*70}")
    print(f"Model:        YOLOv8m-P2 (P2 architecture + COCO pretrained)")
    print(f"Dataset:      FOD-A (41 classes)")
    print(f"Resolution:   {IMGSZ}×{IMGSZ} (paper standard)")
    print(f"Batch Size:   {BATCH} (unified)")
    print(f"Epochs:       {EPOCHS}")
    print(f"Optimizer:    {OPTIMIZER}")
    print(f"LR:           {LR0} (base, no scaling)")
    print(f"Workers:      {WORKERS}")
    print(f"Save to:      {PROJECT_NAME}/{RUN_NAME}")
    print(f"{'='*70}\n")

    # Prepare YOLOv8-P2 config with correct nc
    print("Preparing YOLOv8-P2 configuration...")
    config_path = Path('configs/yolov8-p2.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(DATASET_CONFIG, 'r') as f:
        data_config = yaml.safe_load(f)

    config['nc'] = data_config.get('nc', 41)

    temp_config = Path('configs/yolov8-p2_temp.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ P2 config prepared: {temp_config}")
    print(f"  Number of classes: {config['nc']}\n")

    # Load P2 architecture
    print("Loading YOLOv8-P2 architecture...")
    model = YOLO(str(temp_config))

    # Load COCO pretrained weights
    print("Loading COCO pretrained weights (yolov8m.pt)...")
    model.load('yolov8m.pt')

    # Verify model
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")
    print(f"  Expected: ~42.9M parameters (P2 head adds extra layer)\n")

    # Start training
    print(f"{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")

    results = model.train(
        # Dataset
        data=DATASET_CONFIG,
        
        # Training duration
        epochs=EPOCHS,
        
        # Image settings
        imgsz=IMGSZ,
        
        # Batch & workers
        batch=BATCH,
        workers=WORKERS,
        
        # Device
        device=0,
        
        # Optimizer settings
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        
        # Loss weights
        box=BOX,
        cls=CLS,
        dfl=DFL,
        
        # Data augmentation
        mosaic=MOSAIC,
        mixup=MIXUP,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        
        # Performance
        cache=False,
        
        # Output
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        
        # Validation & saving
        val=True,
        save=True,
        save_period=10,
        plots=True,
        
        # Misc
        verbose=True,
        seed=0,
        deterministic=True,
        patience=100
    )

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {PROJECT_NAME}/{RUN_NAME}")
    print(f"Best weights: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
    print(f"Last weights: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
    print(f"{'='*70}\n")

    # Print final metrics
    print("Final Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print()
