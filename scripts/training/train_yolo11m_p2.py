"""
YOLOv11m-P2 Training Script
Trains YOLOv11m with P2 detection head (4 detection layers) on FOD-A dataset.
Optimized for 24GB GPU.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Configuration
DATASET_CONFIG = "data/FOD-A/data.yaml"
MODEL_CONFIG = "configs/yolo11-p2.yaml"
PRETRAINED_WEIGHTS = "models/yolo11m.pt"

IMG_SIZE = 1280
BATCH_SIZE = 6  # Reduced for P2 architecture (more memory intensive)
EPOCHS = 100
PATIENCE = 20
DEVICE = 0

# Hyperparameters (SGD optimized)
HYPERPARAMS = {
    'lr0': 0.01,     # Standard SGD learning rate
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
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
RUN_NAME = f"yolo11m-p2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def check_gpu():
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        return True
    return False

def train_yolo11m_p2():
    print("="*70)
    print("YOLOv11m-P2 TRAINING (Stride 4)")
    print("="*70)
    
    if not check_gpu():
        return

    # Initialize model from config (defines architecture)
    print(f"\n🏗️  Building YOLOv11m-P2 from {MODEL_CONFIG}...")
    model = YOLO(MODEL_CONFIG)
    
    # Load pretrained weights (transfer learning)
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"🔄 Loading pretrained weights from {PRETRAINED_WEIGHTS}...")
        try:
            # Load weights into the new architecture
            # This will transfer backbone weights and ignore mismatched head weights
            model.load(PRETRAINED_WEIGHTS)
            print("✓ Weights transferred successfully (Backbone initialized)")
        except Exception as e:
            print(f"⚠️ Warning: Could not load weights: {e}")
            print("   Starting training from scratch.")
    else:
        print(f"⚠️ Pretrained weights not found at {PRETRAINED_WEIGHTS}")
        print("   Starting training from scratch.")

    print(f"\n📊 Dataset: {DATASET_CONFIG}")
    print(f"📐 Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"💾 Output: {PROJECT_NAME}/{RUN_NAME}")
    
    # Start training
    print("\n🚀 Starting training...")
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        cache=False,  # Windows optimization
        workers=4,    # Windows optimization
        patience=PATIENCE,
        save=True,
        save_period=10,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',  # Using SGD for stability
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
    
    print("\n✅ Training Complete!")
    return model, results

if __name__ == "__main__":
    try:
        train_yolo11m_p2()
    except Exception as e:
        print(f"\n❌ Error: {e}")
