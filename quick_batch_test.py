"""
Quick Batch Size Tester - Test single model at single batch size
Usage: python quick_batch_test.py <model> <batch>

Models: v8m, v8m-p2, v11m, v11m-p2
Batch: any integer (e.g., 16, 14, 12, 10, 8, 6)

Example:
  python quick_batch_test.py v8m 16
  python quick_batch_test.py v11m-p2 8
"""

import sys
import torch
from ultralytics import YOLO
import gc
import yaml
from pathlib import Path

DATASET_CONFIG = "data/FOD-A/data.yaml"
IMG_SIZE = 640  # Paper's standard resolution (was 1280)
MAX_ITERATIONS = 1  # Only run 50 batches to measure VRAM

# Model configurations
MODEL_CONFIGS = {
    'v8m': {
        'name': 'YOLOv8m Baseline',
        'config': None,
        'weights': 'yolov8m.pt',
        'expected_params': '25.9M'
    },
    'v8m-p2': {
        'name': 'YOLOv8m-P2',
        'config': 'configs/yolov8-p2.yaml',
        'weights': 'yolov8m.pt',
        'expected_params': '42.9M'
    },
    'v11m': {
        'name': 'YOLOv11m Baseline',
        'config': None,
        'weights': 'yolo11m.pt',
        'expected_params': '20.1M'
    },
    'v11m-p2': {
        'name': 'YOLOv11m-P2',
        'config': 'configs/yolo11-p2.yaml',
        'weights': 'yolo11m.pt',
        'expected_params': '20M'
    }
}


def prepare_yolov8_p2_config():
    """Prepare YOLOv8-P2 temp config with correct nc"""
    config_path = Path('configs/yolov8-p2.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(DATASET_CONFIG, 'r') as f:
        data_config = yaml.safe_load(f)
    
    config['nc'] = data_config.get('nc', 41)
    
    temp_config = Path('configs/yolov8-p2_temp.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(temp_config)


def quick_test(model_key, batch_size):
    """Quick test of model at specific batch size"""
    
    if model_key not in MODEL_CONFIGS:
        print(f"\n❌ Invalid model: {model_key}")
        print(f"Valid models: {', '.join(MODEL_CONFIGS.keys())}\n")
        sys.exit(1)
    
    model_info = MODEL_CONFIGS[model_key]
    
    print("\n" + "="*70)
    print(f"QUICK BATCH TEST: {model_info['name']} @ Batch {batch_size}")
    print("="*70)
    print(f"Strategy: Run {MAX_ITERATIONS} iterations to measure peak VRAM")
    print(f"Expected params: {model_info['expected_params']}")
    print("="*70 + "\n")
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Load model
        if model_info['config']:
            if 'yolov8-p2' in model_info['config']:
                config_file = prepare_yolov8_p2_config()
            else:
                config_file = model_info['config']
            print(f"Loading architecture: {config_file}")
            model = YOLO(config_file)
            
            if model_info['weights']:
                print(f"Loading COCO weights: {model_info['weights']}")
                model.load(model_info['weights'])
        else:
            print(f"Loading COCO pretrained: {model_info['weights']}")
            model = YOLO(model_info['weights'])
        
        # Check model size
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"\n✓ Model loaded: {total_params:,} parameters")
        print(f"  Expected: {model_info['expected_params']}")
        
        # Run limited training to measure VRAM
        print(f"\nStarting quick test ({MAX_ITERATIONS} iterations, batch={batch_size})...\n")
        
        # Custom callback to stop after MAX_ITERATIONS
        from ultralytics.utils import callbacks
        iteration_count = [0]
        
        def on_train_batch_end(trainer):
            iteration_count[0] += 1
            if iteration_count[0] >= MAX_ITERATIONS:
                # Force stop
                trainer.stopper.stop = True
                trainer.epoch = trainer.epochs  # End training
        
        # Register callback
        callbacks.default_callbacks['on_train_batch_end'].append(on_train_batch_end)
        
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
            verbose=True,
            plots=False,
            save=False,
            val=False,
            project="fod_detection",
            name=f"quick_test_{model_key.replace('-', '_')}_batch{batch_size}",
            exist_ok=True
        )
        
        # Get max VRAM usage
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        margin = 24 - max_memory_reserved
        
        print("\n" + "="*70)
        print("VRAM USAGE RESULTS")
        print("="*70)
        print(f"Model:            {model_info['name']}")
        print(f"Batch Size:       {batch_size}")
        print(f"Parameters:       {total_params:,} ({model_info['expected_params']})")
        print(f"Iterations:       {iteration_count[0]}")
        print(f"-"*70)
        print(f"VRAM Allocated:   {max_memory_allocated:.2f} GB")
        print(f"VRAM Reserved:    {max_memory_reserved:.2f} GB")
        print(f"Margin (24GB):    {margin:.2f} GB")
        print(f"-"*70)
        
        if margin < 1.0:
            print(f"⚠️  WARNING: <1GB margin - may OOM during full training!")
        elif margin < 2.0:
            print(f"⚠️  CAUTION: <2GB margin - tight but workable")
        else:
            print(f"✅ SAFE: {margin:.2f}GB margin")
        
        print("="*70 + "\n")
        
        # Cleanup
        del model, results
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'success': True,
            'batch': batch_size,
            'params': total_params,
            'vram_allocated': max_memory_allocated,
            'vram_reserved': max_memory_reserved,
            'margin': margin
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "="*70)
            print("❌ OUT OF MEMORY")
            print("="*70)
            print(f"Model:       {model_info['name']}")
            print(f"Batch Size:  {batch_size}")
            print(f"Result:      FAILED - GPU OOM")
            print("="*70 + "\n")
            
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'success': False,
                'batch': batch_size,
                'error': 'OOM'
            }
        else:
            raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'success': False,
            'batch': batch_size,
            'error': str(e)
        }


def print_usage():
    """Print usage instructions"""
    print("\n" + "="*70)
    print("QUICK BATCH SIZE TESTER")
    print("="*70)
    print("\nUsage: python quick_batch_test.py <model> <batch>\n")
    print("Available Models:")
    print("  v8m       - YOLOv8m Baseline")
    print("  v8m-p2    - YOLOv8m with P2 head")
    print("  v11m      - YOLOv11m Baseline")
    print("  v11m-p2   - YOLOv11m with P2 head")
    print("\nExamples:")
    print("  python quick_batch_test.py v8m 16")
    print("  python quick_batch_test.py v8m-p2 8")
    print("  python quick_batch_test.py v11m 12")
    print("  python quick_batch_test.py v11m-p2 8")
    print("\nStrategy:")
    print(f"  - Runs only {MAX_ITERATIONS} iterations (~2 minutes)")
    print("  - Measures peak VRAM usage")
    print("  - Shows margin to 24GB limit")
    print("  - Folder: fod_detection/quick_test_<model>_batch<N>")
    print("="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    
    model_key = sys.argv[1].lower()
    
    try:
        batch_size = int(sys.argv[2])
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
    except ValueError as e:
        print(f"\n❌ Invalid batch size: {sys.argv[2]}")
        print(f"Batch size must be a positive integer (e.g., 16, 12, 8)\n")
        sys.exit(1)
    
    result = quick_test(model_key, batch_size)
    
    if result['success']:
        print(f"\n✅ Test complete! Margin: {result['margin']:.2f}GB")
        print(f"📁 Results: fod_detection/quick_test_{model_key.replace('-', '_')}_batch{batch_size}\n")
    else:
        print(f"\n❌ Test failed: {result.get('error', 'Unknown error')}\n")
        sys.exit(1)
