"""
YOLOv11m Baseline Training Script - BATCH 16 (Fair Comparison)
Trains YOLOv11m baseline model on FOD-A dataset at batch 16 for fair comparison.

UPDATES:
- Batch size: 8 → 16 (matches YOLOv8m baseline)
- SGD optimizer (stable, proven)
- Optimized hyperparameters for small object detection
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Configuration
DATASET_CONFIG = "data/FOD-A/data.yaml"
MODEL_SIZE = "models/yolo11m.pt"
IMG_SIZE = 1280
BATCH_SIZE = 16  # ⬆️ UPDATED from 8 for fair comparison
EPOCHS = 100
PATIENCE = 25
DEVICE = 0

# Hyperparameters (optimized for small objects + batch 16)
HYPERPARAMS = {
    'lr0': 0.01,        # Standard for batch 16
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,
    'box': 10.0,        # ⬆️ Higher weight for small bounding boxes
    'cls': 0.5,
    'dfl': 2.0,         # ⬆️ Better localization for small objects
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 5.0,     # ⬆️ Small rotation augmentation
    'translate': 0.15,  # ⬆️ More spatial variation
    'scale': 0.7,       # ⬆️ Aggressive scale augmentation
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.2,       # ⬆️ Increased
    'copy_paste': 0.4,  # ⬆️ Increased
    'close_mosaic': 15, # ⬆️ Keep mosaic longer
}

# Output paths
PROJECT_NAME = "fod_detection"
RUN_NAME = f"yolo11m_baseline_batch16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def check_gpu():
    """Check GPU availability and clear cache"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        return True
    else:
        print("⚠️ No GPU detected! Training will be slow.")
        return False


def train_yolo11m_baseline_batch16():
    """Train YOLOv11m baseline model at batch 16"""
    
    print("="*70)
    print("YOLOv11m BASELINE TRAINING (BATCH 16 - FAIR COMPARISON)")
    print("="*70)
    
    if not check_gpu():
        return
    
    # Verify dataset
    dataset_path = Path(DATASET_CONFIG)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_CONFIG}")
    
    # Check if model exists locally
    model_path = Path("models") / MODEL_SIZE
    if not model_path.exists():
        print(f"\n⏳ Pretrained model not found locally, will download from Ultralytics...")
        model_to_load = MODEL_SIZE
    else:
        print(f"\n✓ Using local pretrained model: {model_path}")
        model_to_load = str(model_path)
    
    print(f"\n📊 Configuration:")
    print(f"   Dataset: {DATASET_CONFIG}")
    print(f"   Model: YOLOv11m (latest YOLO architecture)")
    print(f"   Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE} ⬆️ (was 8)")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Optimizer: SGD (stable)")
    print(f"   Output: {PROJECT_NAME}/{RUN_NAME}")
    
    print(f"\n🎯 Optimizations:")
    print(f"   • High box/DFL loss weights (10.0/2.0)")
    print(f"   • Aggressive augmentation (scale=0.7)")
    print(f"   • Batch 16 for fair comparison")
    print(f"   • SGD optimizer (AdamW was unstable)")
    
    # Load pretrained model
    print(f"\n⏳ Loading pretrained {MODEL_SIZE}...")
    model = YOLO(model_to_load)
    
    # Start training
    print("\n🚀 Starting training...")
    print("\n⚡ Expected improvements over batch=8 run:")
    print("   • Better gradient estimates (100% more samples)")
    print("   • More stable convergence")
    print("   • +1.0-2.0% mAP improvement expected\n")
    
    results = model.train(
        # Data & Model
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        
        # Hardware
        device=DEVICE,
        cache=False,
        workers=4,
        
        # Training Control
        patience=PATIENCE,
        save=True,
        save_period=10,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        
        # Optimizer
        pretrained=True,
        optimizer='SGD',  # ⚠️ CRITICAL: SGD stable, AdamW diverged
        verbose=True,
        seed=42,
        deterministic=False,
        
        # Training Settings
        single_cls=False,
        rect=False,
        cos_lr=True,
        label_smoothing=0.0,
        val=True,
        plots=True,
        amp=True,
        
        # Hyperparameters
        **HYPERPARAMS
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Print final metrics
    print(f"\n📈 Final Results:")
    metrics = results.results_dict
    print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"   mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    best_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    print(f"\n💾 Best Model: {best_model}")
    
    print("\n📊 Comparison:")
    print("   Previous (batch=8): 86.87% mAP@50-95")
    print("   Current (batch=16): Check results.csv for final score")
    print("   Expected: 87.9-88.9% mAP@50-95\n")
    
    return model, results


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("YOLOv11m BASELINE RETRAINING - FAIR COMPARISON (BATCH 16)")
        print("="*70)
        print("\nReason for retraining:")
        print("  • Original training used batch=8 (after AdamW crash)")
        print("  • YOLOv8m baseline used batch=16")
        print("  • Batch size affects gradient quality and convergence")
        print("  • Need same batch for fair architectural comparison")
        print("\nExpected VRAM usage: ~16-17GB (safe for 24GB GPU)")
        print("="*70 + "\n")
        
        model, results = train_yolo11m_baseline_batch16()
        print("\n✅ Training completed successfully!\n")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
