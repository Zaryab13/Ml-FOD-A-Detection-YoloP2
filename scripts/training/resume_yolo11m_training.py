"""
Resume YOLOv11m Training Script
Resumes training from Epoch 20 checkpoint to fix NaN/Inf divergence issue.
Uses lower learning rate to prevent further divergence.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Configuration
DATASET_CONFIG = "data/FOD-A/data.yaml"
# Point to the good checkpoint before divergence
CHECKPOINT_PATH = "fod_detection/yolo11m_baseline_20251222_223346/weights/epoch20.pt"

IMG_SIZE = 1280
BATCH_SIZE = 8
EPOCHS = 80  # Remaining epochs (100 - 20)
PATIENCE = 20
DEVICE = 0

# Hyperparameters - Modified for stability
HYPERPARAMS = {
    'lr0': 0.005,  # SGD is more stable, but we use conservative LR
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,  # Increased warmup
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.9,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.15,
    'copy_paste': 0.3,
    'close_mosaic': 10,
}

# Output paths
PROJECT_NAME = "fod_detection"
RUN_NAME = "yolo11m_baseline_20251222_223346"  # Save in original folder

def train_resume():
    print("="*70)
    print("RESUMING YOLOv11m TRAINING (STABILIZED)")
    print("="*70)
    
    # Verify checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        
    print(f"\n🔄 Loading checkpoint: {CHECKPOINT_PATH}")
    print("📉 Reducing Learning Rate: 0.01 -> 0.001 (to fix NaN/Inf)")
    
    # Load model
    model = YOLO(CHECKPOINT_PATH)
    
    # Start training
    print("\n🚀 Starting stabilized training...")
    
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        cache=False,
        workers=4,
        patience=PATIENCE,
        save=True,
        save_period=10,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',  # Switched to SGD for maximum stability
        verbose=True,
        seed=42,
        deterministic=False,
        single_cls=False,
        rect=False,
        cos_lr=True,
        label_smoothing=0.0,
        val=True,
        plots=True,
        amp=True,
        **HYPERPARAMS
    )
    
    print("\n✅ Training Completed!")
    return model, results

if __name__ == "__main__":
    try:
        train_resume()
    except Exception as e:
        print(f"\n❌ Error: {e}")
