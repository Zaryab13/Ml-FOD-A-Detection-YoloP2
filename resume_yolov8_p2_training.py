"""
Resume YOLOv8-P2 Training from Checkpoint
Resumes training from epoch 98 to complete the final 2 epochs (99-100)
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# Configuration
RESUME_CHECKPOINT = "fod_detection/yolov8m-p2_20251220_185845/weights/last.pt"
DATASET_CONFIG = "data/FOD-A/data.yaml"
REMAINING_EPOCHS = 2  # Complete epochs 99-100

# Original training settings (must match original training)
IMG_SIZE = 1280
BATCH_SIZE = 6
DEVICE = 0

# Hyperparameters (same as original)
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

# Output paths (same as original)
PROJECT_NAME = "fod_detection"
RUN_NAME = "yolov8m-p2_20251220_185845"  # Same run name to continue in same folder


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Clear cache before resuming
        torch.cuda.empty_cache()
        print(f"✓ GPU cache cleared")
    else:
        print("⚠️ No GPU detected!")
        return False
    return True


def resume_training():
    """Resume training from last checkpoint"""
    
    print("="*70)
    print("RESUME YOLOv8-P2 TRAINING")
    print("="*70)
    
    # Check GPU
    if not check_gpu():
        response = input("Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Verify checkpoint exists
    checkpoint_path = Path(RESUME_CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {RESUME_CHECKPOINT}")
    
    # Verify dataset
    dataset_path = Path(DATASET_CONFIG)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_CONFIG}")
    
    print(f"\n📦 Resuming from: {RESUME_CHECKPOINT}")
    print(f"📊 Dataset: {DATASET_CONFIG}")
    print(f"🔄 Remaining Epochs: {REMAINING_EPOCHS}")
    print(f"📐 Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    
    # Load model from checkpoint
    print(f"\n⏳ Loading checkpoint from epoch 98...")
    model = YOLO(RESUME_CHECKPOINT)
    
    print(f"\n🚀 Resuming training for final {REMAINING_EPOCHS} epochs...\n")
    
    # Resume training - YOLO will automatically continue from saved epoch
    results = model.train(
        data=DATASET_CONFIG,
        epochs=100,  # Total epochs (will resume from epoch 98)
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        resume=True,  # CRITICAL: Resume from checkpoint
        cache=True,
        workers=16,
        patience=20,
        save=True,
        save_period=10,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,  # Continue in same folder
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
    print("✅ TRAINING RESUMED AND COMPLETED!")
    print("="*70)
    
    # Final checkpoint locations
    weights_dir = Path(PROJECT_NAME) / RUN_NAME / "weights"
    final_best = weights_dir / "best.pt"
    final_last = weights_dir / "last.pt"
    epoch100 = weights_dir / "epoch100.pt"
    
    print(f"\n💾 Updated Weights:")
    if final_best.exists():
        print(f"   Best: {final_best}")
    if final_last.exists():
        print(f"   Last (epoch 100): {final_last}")
    if epoch100.exists():
        print(f"   Epoch 100: {epoch100}")
    
    # Run final validation
    print("\n🔍 Running final validation on completed model...")
    val_results = model.val()
    
    print(f"\n📊 Final Validation Results (Epoch 100):")
    metrics = val_results.results_dict
    print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE - 100/100 EPOCHS")
    print("="*70)
    
    return model, results


if __name__ == "__main__":
    try:
        print("\n⚠️  IMPORTANT: This will resume training from epoch 98")
        print("   Last checkpoint: fod_detection/yolov8m-p2_20251220_185845/weights/last.pt")
        print("   Target: Complete epochs 99-100\n")
        
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            exit(0)
        
        model, results = resume_training()
        
        print("\n✅ Script completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Check updated results.csv for epochs 99-100")
        print("   2. Verify last.pt is now epoch 100 checkpoint")
        print("   3. Compare best.pt metrics with previous best")
        print("   4. Push updated results to Git:")
        print("      git add .")
        print("      git commit -m 'Complete YOLOv8-P2 training (100/100 epochs)'")
        print("      git push")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
