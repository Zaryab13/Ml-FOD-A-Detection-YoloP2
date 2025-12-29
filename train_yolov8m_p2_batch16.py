"""
YOLOv8m-P2 Training Script - BATCH 16 (Fair Comparison)
Trains YOLOv8-P2 with P2 detection head on FOD-A dataset at batch 16 for fair comparison.

UPDATES:
- Batch size: 6 → 16 (matches YOLOv8m baseline)
- Optimized hyperparameters for small object detection
- SGD optimizer (proven stable)
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

# Configuration
DATASET_CONFIG = "data/FOD-A/data.yaml"
MODEL_CONFIG = "configs/yolov8-p2.yaml"
MODEL_SIZE = "yolov8m"
IMG_SIZE = 1280
BATCH_SIZE = 16  # ⬆️ UPDATED from 6 for fair comparison
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
RUN_NAME = f"yolov8m-p2_batch16_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


def prepare_p2_config():
    """Prepare YOLOv8-P2 configuration with correct nc value"""
    config_path = Path(MODEL_CONFIG)
    
    if not config_path.exists():
        raise FileNotFoundError(f"P2 config not found: {MODEL_CONFIG}")
    
    # Load and verify config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset config to get nc
    with open(DATASET_CONFIG, 'r') as f:
        data_config = yaml.safe_load(f)
    
    nc = data_config.get('nc', 41)
    config['nc'] = nc
    
    # Save updated config
    temp_config = config_path.parent / f"yolov8-p2_temp.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(temp_config)


def train_yolov8m_p2_batch16():
    """Train YOLOv8m-P2 model at batch 16"""
    
    print("="*70)
    print("YOLOv8m-P2 TRAINING (BATCH 16 - FAIR COMPARISON)")
    print("="*70)
    
    if not check_gpu():
        return
    
    # Verify dataset
    dataset_path = Path(DATASET_CONFIG)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_CONFIG}")
    
    # Prepare P2 config
    print(f"\n⏳ Preparing P2 configuration...")
    config_file = prepare_p2_config()
    print(f"✓ P2 config ready: {config_file}")
    
    print(f"\n📊 Configuration:")
    print(f"   Dataset: {DATASET_CONFIG}")
    print(f"   Model: YOLOv8m-P2 (P2/P3/P4/P5 detection heads)")
    print(f"   Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE} ⬆️ (was 6)")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Optimizer: SGD (stable)")
    print(f"   Output: {PROJECT_NAME}/{RUN_NAME}")
    
    print(f"\n🎯 Optimizations:")
    print(f"   • P2 detection head (stride=4) for small objects")
    print(f"   • High box/DFL loss weights (10.0/2.0)")
    print(f"   • Aggressive augmentation (scale=0.7)")
    print(f"   • Batch 16 for fair comparison with baseline")
    
    # Load model from config
    print(f"\n⏳ Initializing model from {config_file}...")
    model = YOLO(config_file)
    
    # Start training
    print("\n🚀 Starting training...")
    print("\n⚡ Expected improvements over batch=6 run:")
    print("   • Better gradient estimates (167% more samples)")
    print("   • More stable convergence")
    print("   • +0.5-1.5% mAP improvement expected\n")
    
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
        optimizer='SGD',
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
    print("   Previous (batch=6): 92.88% mAP@50-95")
    print("   Current (batch=16): Check results.csv for final score")
    print("   Expected: 93.4-94.4% mAP@50-95\n")
    
    return model, results


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("YOLOv8m-P2 RETRAINING - FAIR COMPARISON (BATCH 16)")
        print("="*70)
        print("\nReason for retraining:")
        print("  • Original training used batch=6")
        print("  • YOLOv8m baseline used batch=16")
        print("  • Batch size affects gradient quality and convergence")
        print("  • Need same batch for fair architectural comparison")
        print("\nExpected VRAM usage: ~16-17GB (safe for 24GB GPU)")
        print("="*70 + "\n")
        
        model, results = train_yolov8m_p2_batch16()
        print("\n✅ Training completed successfully!\n")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
