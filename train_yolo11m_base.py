"""
YOLOv11m Base Model Training Script
Trains YOLOv11m model on FOD-A dataset as comparison to YOLOv8 models
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Configuration
DATASET_CONFIG = "data/FOD-A/data.yaml"
MODEL_SIZE = "yolo11m.pt"  # YOLOv11m pretrained model
IMG_SIZE = 1280  # Higher resolution for small objects
BATCH_SIZE = 8  # Reduced from 16 to prevent VRAM spillover (30GB+ usage on 24GB card)
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
RUN_NAME = f"yolo11m_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def check_gpu():
    """Check GPU availability and CUDA version"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Clear cache before training
        torch.cuda.empty_cache()
        print(f"✓ GPU cache cleared")
    else:
        print("⚠️ No GPU detected! Training will be slow.")
        return False
    return True


def train_yolo11m_base():
    """Train YOLOv11m base model"""
    
    print("="*70)
    print("YOLOv11m BASELINE TRAINING")
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
    
    # Check if model exists locally
    model_path = Path("models") / MODEL_SIZE
    if not model_path.exists():
        print(f"\n⏳ Pretrained model not found locally, will download from Ultralytics...")
        model_to_load = MODEL_SIZE
    else:
        print(f"\n✓ Using local pretrained model: {model_path}")
        model_to_load = str(model_path)
    
    print(f"\n📊 Dataset: {DATASET_CONFIG}")
    print(f"🏗️  Model: YOLOv11m (latest YOLO architecture)")
    print(f"📐 Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"💾 Output: {PROJECT_NAME}/{RUN_NAME}")
    
    # Load pretrained model
    print(f"\n⏳ Loading pretrained {MODEL_SIZE}...")
    model = YOLO(model_to_load)
    
    # Start training
    print("\n🚀 Starting training...\n")
    print("⚡ YOLOv11 improvements over v8:")
    print("   • Enhanced backbone architecture")
    print("   • Improved feature fusion")
    print("   • Better small object detection")
    print("   • Optimized anchor-free detection\n")
    
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        cache=False,  # Disable cache to avoid Windows shared memory issues
        workers=4,  # Reduced for Windows - prevents shared memory errors
        patience=PATIENCE,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',  # AdamW often better for larger batches
        verbose=True,
        seed=42,
        deterministic=False,  # Faster training, still reproducible with seed
        single_cls=False,
        rect=False,  # Rectangular training
        cos_lr=True,  # Cosine LR scheduler
        label_smoothing=0.0,
        val=True,
        plots=True,
        amp=True,  # Automatic Mixed Precision for faster training on 24GB GPU
        **HYPERPARAMS
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Print results summary
    print(f"\n📈 Final Metrics:")
    metrics = results.results_dict
    print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
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
    val_metrics = val_results.results_dict
    print(f"   Precision: {val_metrics.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   Recall: {val_metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"   mAP50: {val_metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95: {val_metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    print("\n" + "="*70)
    print("📊 COMPARISON READY")
    print("="*70)
    print("\nYou can now compare:")
    print("   • YOLOv8m baseline")
    print("   • YOLOv8m-P2")
    print("   • YOLOv11m baseline (this run)")
    
    return model, results


if __name__ == "__main__":
    try:
        model, results = train_yolo11m_base()
        print("\n✅ Script completed successfully!")
        
        print("\n💡 Next steps:")
        print("   1. Compare results with YOLOv8 models")
        print("   2. Check if YOLOv11 improved upon v8 baseline")
        print("   3. Push results to Git:")
        print("      git add .")
        print("      git commit -m 'Add YOLOv11m baseline training results'")
        print("      git push")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
