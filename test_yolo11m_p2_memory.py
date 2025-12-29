"""
Test YOLOv11m-P2 Memory Usage with CORRECTED Configuration
Quick test to determine actual VRAM usage of TRUE YOLOv11m-P2 (medium scale)
"""

import torch
from ultralytics import YOLO
import gc

MODEL_CONFIG = "configs/yolo11-p2.yaml"
PRETRAINED_WEIGHTS = "models/yolo11m.pt"
DATASET_CONFIG = "data/FOD-A/data.yaml"
IMG_SIZE = 1280

# Test batch sizes for YOLOv11m-P2 (MEDIUM scale this time!)
TEST_BATCHES = [6, 8, 10, 12, 14]

def test_batch_size(batch_size):
    """Test a specific batch size and return VRAM usage"""
    print(f"\n{'='*70}")
    print(f"Testing YOLOv11m-P2 (MEDIUM) at Batch {batch_size}")
    print(f"{'='*70}\n")
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load model
        print(f"Loading YOLOv11m-P2 from {MODEL_CONFIG}...")
        model = YOLO(MODEL_CONFIG)
        
        # Load pretrained weights
        print(f"Loading weights from {PRETRAINED_WEIGHTS}...")
        model.load(PRETRAINED_WEIGHTS)
        
        # Check if it loaded correctly
        print(f"\n📊 Model Info:")
        print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        print(f"   Expected: ~20M for YOLOv11m-P2 (not 2.7M nano!)")
        
        # Run 1 epoch to measure VRAM
        print(f"\nStarting test training (1 epoch, batch={batch_size})...")
        results = model.train(
            data=DATASET_CONFIG,
            epochs=1,
            imgsz=IMG_SIZE,
            batch=batch_size,
            device=0,
            cache=False,
            workers=4,
            optimizer='SGD',
            lr0=0.01,
            verbose=False,
            plots=False,
            save=False,
            val=False,
            project="fod_detection",
            name=f"yolo11m_p2_test_batch{batch_size}",
            exist_ok=True
        )
        
        # Get max VRAM usage
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        print(f"\n✅ SUCCESS: Batch {batch_size}")
        print(f"   Max VRAM Allocated: {max_memory_allocated:.2f} GB")
        print(f"   Max VRAM Reserved: {max_memory_reserved:.2f} GB")
        print(f"   Available Margin: {24 - max_memory_reserved:.2f} GB\n")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'batch': batch_size,
            'success': True,
            'vram_allocated': max_memory_allocated,
            'vram_reserved': max_memory_reserved,
            'margin': 24 - max_memory_reserved
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ FAILED: Batch {batch_size} - Out of Memory")
            print(f"   Error: {e}\n")
            
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'batch': batch_size,
                'success': False,
                'error': 'OOM'
            }
        else:
            raise

if __name__ == "__main__":
    print("\n" + "="*70)
    print("YOLOv11m-P2 CORRECTED MEMORY TEST")
    print("="*70)
    print("\n🐛 BUG FIX:")
    print("   Previous tests used YOLOv11n-P2 (2.7M params, nano)")
    print("   Now testing TRUE YOLOv11m-P2 (~20M params, medium)")
    print("\n   This will use MUCH more VRAM than before!")
    print("="*70 + "\n")
    
    results = []
    max_successful_batch = None
    
    for batch in TEST_BATCHES:
        result = test_batch_size(batch)
        results.append(result)
        
        if result['success']:
            max_successful_batch = batch
            if result.get('margin', 100) < 2.0:
                print(f"⚠️  Stopping: Only {result['margin']:.2f}GB margin left\n")
                break
        else:
            print(f"🛑 Stopping: Batch {batch} failed\n")
            break
    
    # Print summary
    print("\n" + "="*70)
    print("CORRECTED YOLOv11m-P2 BATCH SIZE RESULTS")
    print("="*70)
    print(f"\n{'Batch':<8} {'Status':<12} {'VRAM (GB)':<12} {'Margin (GB)':<12}")
    print("-" * 70)
    
    for r in results:
        if r['success']:
            print(f"{r['batch']:<8} ✅ Success    {r['vram_reserved']:<12.2f} {r['margin']:<12.2f}")
        else:
            print(f"{r['batch']:<8} ❌ OOM       -            -")
    
    print("\n" + "="*70)
    print("UPDATED RECOMMENDATION")
    print("="*70)
    
    if max_successful_batch:
        print(f"\n🎯 Maximum Batch for YOLOv11m-P2: {max_successful_batch}")
        print(f"\nCompare with other models:")
        print(f"   YOLOv11m Baseline: batch 16 → 30.8GB (OOM)")
        print(f"   YOLOv8m-P2: batch 16 → OOM")
        print(f"   YOLOv11m-P2: batch {max_successful_batch} → ACTUAL MEDIUM SCALE")
    
    print("\n" + "="*70 + "\n")
