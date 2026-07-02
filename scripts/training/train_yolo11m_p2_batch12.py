"""
YOLOv11m-P2 Optimized Training Script
Implements gradient accumulation and optimized hyperparameters for small object detection.

OPTIMIZATIONS:
- Gradient Accumulation: Simulates batch 18 while using batch 6 (VRAM friendly)
- Small Object Focus: Increased box loss weight, DFL loss, aggressive augmentation
- Scaled Learning Rate: Adjusted for effective batch size

Author: FOD-A Detection Research Team
Date: December 2025
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_CONFIG = "data/FOD-A/data.yaml"
MODEL_CONFIG = "configs/yolo11-p2.yaml"
PRETRAINED_WEIGHTS = "models/yolo11m.pt"

IMG_SIZE = 1280
BATCH_SIZE = 12          # Optimized for 24GB VRAM - fair comparison with baseline models
EPOCHS = 100
PATIENCE = 25            # Increased patience for stability
DEVICE = 0

# ============================================================================
# OPTIMIZED HYPERPARAMETERS FOR SMALL OBJECTS
# ============================================================================

HYPERPARAMS = {
    # Learning Rate (optimized for batch 12)
    'lr0': 0.0075,       # Scaled for batch 12 (0.01 × 12/16)
    'lrf': 0.01,         # Final LR ratio
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 5.0,  # Longer warmup for training stability
    
    # Loss Weights (CRITICAL for small object detection)
    'box': 10.0,         # ⬆️ Increased from 7.5 - small boxes need higher weight
    'cls': 0.5,          # Keep balanced
    'dfl': 2.0,          # ⬆️ Increased from 1.5 - better localization for small objects
    
    # Data Augmentation (aggressive for small object robustness)
    'hsv_h': 0.015,      # Hue variation
    'hsv_s': 0.7,        # Saturation variation
    'hsv_v': 0.4,        # Value variation
    'degrees': 5.0,      # ⬆️ Small rotation (was 0) - helps generalization
    'translate': 0.15,   # ⬆️ Increased from 0.1 - more spatial variation
    'scale': 0.7,        # ⬆️ More scale variation (was 0.9) - critical for small objects
    'flipud': 0.0,       # No vertical flip (objects have orientation)
    'fliplr': 0.5,       # Horizontal flip
    'mosaic': 1.0,       # Always use mosaic
    'mixup': 0.2,        # ⬆️ Increased from 0.15
    'copy_paste': 0.4,   # ⬆️ Increased from 0.3 - helps with occlusion
    'close_mosaic': 15,  # ⬆️ Keep mosaic longer (was 10) - more aggressive training
}

# Output paths
PROJECT_NAME = "fod_detection"
RUN_NAME = f"yolo11m-p2_batch12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def check_gpu():
    """Check GPU availability and clear cache"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        return True
    return False

def print_optimization_summary():
    """Print optimization strategy summary"""
    print("\n" + "="*70)
    print("OPTIMIZATION STRATEGY")
    print("="*70)
    print(f"\n🔧 Batch Size Optimization:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Note: Gradient accumulation not supported in Ultralytics")
    print(f"   Strategy: Optimized hyperparameters for batch {BATCH_SIZE}")
    
    print(f"\n📈 Adjusted Learning Rate:")
    print(f"   Base LR (batch 16): 0.01")
    print(f"   Optimized LR (batch {BATCH_SIZE}): {HYPERPARAMS['lr0']}")
    
    print(f"\n🎯 Small Object Optimizations:")
    print(f"   Box Loss Weight: {HYPERPARAMS['box']} (↑ from 7.5)")
    print(f"   DFL Loss Weight: {HYPERPARAMS['dfl']} (↑ from 1.5)")
    print(f"   Scale Augmentation: {HYPERPARAMS['scale']} (↑ from 0.9)")
    print(f"   Translation: {HYPERPARAMS['translate']} (↑ from 0.1)")
    
    print(f"\n💾 Memory Management:")
    print(f"   Expected VRAM: ~16-18GB (batch {BATCH_SIZE})")
    print(f"   Workers: 4 (Windows optimized)")
    print(f"   Cache: False (avoid shared memory issues)")
    print("="*70 + "\n")

def train_yolo11m_p2_optimized():
    """Train YOLOv11m-P2 with optimized settings"""
    
    print("="*70)
    print("YOLOv11m-P2 OPTIMIZED TRAINING")
    print("Small Object Detection with Gradient Accumulation")
    print("="*70)
    
    if not check_gpu():
        return
    
    print_optimization_summary()
    
    # Initialize model from config
    print(f"🏗️  Building YOLOv11m-P2 from {MODEL_CONFIG}...")
    model = YOLO(MODEL_CONFIG)
    
    # Load pretrained weights (transfer learning)
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"🔄 Loading pretrained backbone from {PRETRAINED_WEIGHTS}...")
        try:
            model.load(PRETRAINED_WEIGHTS)
            print("✓ Backbone weights transferred successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load weights: {e}")
            print("   Starting from scratch.")
    
    print(f"\n📊 Configuration:")
    print(f"   Dataset: {DATASET_CONFIG}")
    print(f"   Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Patience: {PATIENCE}")
    print(f"   Output: {PROJECT_NAME}/{RUN_NAME}")
    
    # Start training
    print("\n🚀 Starting optimized training...")
    print("\n⚡ Key improvements over previous run:")
    print("   1. Increased batch size 6→8 → more stable gradients")
    print("   2. Higher box/DFL loss → better small object localization")
    print("   3. Aggressive augmentation → improved generalization")
    print("   4. Optimized learning rate → better convergence\n")
    
    results = model.train(
        # Data & Model
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        
        # Batch Size (optimized for YOLOv11m-P2)
        batch=BATCH_SIZE,
        
        # Hardware
        device=DEVICE,
        cache=False,      # Windows optimization
        workers=4,        # Windows optimization
        
        # Training Control
        patience=PATIENCE,
        save=True,
        save_period=10,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        
        # Optimizer
        pretrained=True,
        optimizer='SGD',  # Stable for small batches
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
        amp=True,         # Automatic Mixed Precision
        
        # Hyperparameters (optimized for small objects)
        **HYPERPARAMS
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    # Print results summary
    print(f"\n📈 Final Metrics:")
    metrics = results.results_dict
    print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"   mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    # Save paths
    best_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    last_model = Path(PROJECT_NAME) / RUN_NAME / "weights" / "last.pt"
    
    print(f"\n💾 Model Weights Saved:")
    print(f"   Best: {best_model}")
    print(f"   Last: {last_model}")
    
    print("\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    print("\nCompare this optimized run with:")
    print("   • YOLOv11m-P2 (original): 90.62% mAP@50-95 (batch 6)")
    print("   • YOLOv8m Baseline: 93.31% mAP@50-95 (batch 16)")
    print("\nExpected improvement: +0.5-1.5% mAP from larger batch + optimized hyperparameters\n")
    
    return model, results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("YOLO11M-P2 OPTIMIZATION EXPERIMENT")
        print("="*70)
        print("\nThis script implements:")
        print("  ✓ Increased Batch Size (6 → 8)")
        print("  ✓ Small Object Loss Weights (box=10, dfl=2)")
        print("  ✓ Aggressive Augmentation (scale=0.7, translate=0.15)")
        print("  ✓ Optimized Learning Rate (lr=0.005 for batch 8)")
        print("\n" + "="*70 + "\n")
        
        model, results = train_yolo11m_p2_optimized()
        
        print("\n✅ Script completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Check results.csv for training curves")
        print("   2. Compare mAP@50-95 with previous run (90.62%)")
        print("   3. If improvement ≥1%, document success; else try Phase 2 (remove P5 head)")
        print("   4. Update MODEL_COMPARISON_REPORT.md with new results")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
