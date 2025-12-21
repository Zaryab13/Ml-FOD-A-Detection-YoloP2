"""
YOLOv8-P2 Model Training Script
Trains YOLOv8-P2 (with P2 detection head) on FOD-A dataset for improved small object detection
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

# Configuration
DATASET_CONFIG = "data/FOD-A/data.yaml"
MODEL_CONFIG = "configs/yolov8-p2.yaml"  # Custom P2 architecture
MODEL_SIZE = "yolov8m"  # Base model to initialize from (n, s, m, l, x)
IMG_SIZE = 1280  # Higher resolution crucial for small objects
BATCH_SIZE = 6  # Optimized for 24GB GPU (using ~22-23GB total)
EPOCHS = 100
PATIENCE = 20  # Early stopping patience
DEVICE = 0  # GPU device

# Hyperparameters (optimized for small objects)
HYPERPARAMS = {
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'box': 7.5,  # Higher box loss weight for small objects
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
RUN_NAME = f"yolov8m-p2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def check_gpu():
    """Check GPU availability and CUDA version"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ No GPU detected! Training will be slow.")
        return False
    return True


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


def train_yolov8_p2():
    """Train YOLOv8-P2 model with P2 detection head"""
    
    print("="*70)
    print("YOLOv8-P2 TRAINING (Small Object Detection)")
    print("="*70)
    
    # Check GPU
    if not check_gpu():
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Verify dataset
    dataset_path = Path(DATASET_CONFIG)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_CONFIG}")
    
    # Prepare P2 config
    print("\n📝 Preparing YOLOv8-P2 configuration...")
    p2_config = prepare_p2_config()
    
    print(f"\n📊 Dataset: {DATASET_CONFIG}")
    print(f"🏗️  Architecture: YOLOv8{MODEL_SIZE[-1].upper()}-P2 (4 detection heads: P2/4, P3/8, P4/16, P5/32)")
    print(f"📐 Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"💾 Output: {PROJECT_NAME}/{RUN_NAME}")
    
    # Initialize model with P2 architecture
    print(f"\n⏳ Building YOLOv8-P2 model from config...")
    model = YOLO(p2_config)
    
    # Start training
    print("\n🚀 Starting training...\n")
    print("⚡ P2 head will detect small objects (<32px) at stride=4")
    print("⚡ Using aggressive augmentation for small object robustness\n")
    
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        cache=True,
        workers=16,  # <-- INCREASE FROM 8 TO 12
        patience=PATIENCE,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=False,  # Training from scratch with custom architecture
        optimizer='SGD',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        label_smoothing=0.0,
        val=True,
        plots=True,
        **HYPERPARAMS
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Print results summary
    print(f"\n📈 Final Metrics:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    # Save paths
    best_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    last_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "last.pt"
    
    print(f"\n💾 Model Weights Saved:")
    print(f"   Best: {best_model}")
    print(f"   Last: {last_model}")
    
    # Validation
    print("\n🔍 Running final validation...")
    val_results = model.val()
    
    print(f"\n📊 Validation Results:")
    print(f"   Precision: {val_results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   Recall: {val_results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"   mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    # Cleanup temp config
    if Path(p2_config).exists():
        Path(p2_config).unlink()
    
    return model, results


if __name__ == "__main__":
    try:
        model, results = train_yolov8_p2()
        print("\n✅ Script completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
