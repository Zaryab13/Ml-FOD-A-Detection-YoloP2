"""
Comprehensive Batch Size Optimization for All Models
Tests YOLOv8m Baseline, YOLOv8m-P2, YOLOv11m Baseline, YOLOv11m-P2
to find optimal batch size for fair comparison on 24GB GPU.

Strategy:
1. Test each model independently
2. Find maximum stable batch for each
3. Determine common batch size OR individual optimums
4. Recommend gradient accumulation alternatives
"""

import torch
from ultralytics import YOLO
import gc
import yaml
from pathlib import Path

DATASET_CONFIG = "data/FOD-A/data.yaml"
IMG_SIZE = 1280

# Model configurations
MODELS = {
    'YOLOv8m_Baseline': {
        'config': None,
        'weights': 'yolov8m.pt',
        'test_batches': [12, 14, 16, 18, 20],
        'expected_params': '25.9M',
        'previous_batch': 16
    },
    'YOLOv8m-P2': {
        'config': 'configs/yolov8-p2.yaml',
        'weights': 'yolov8m.pt',  # Load pretrained YOLOv8m weights
        'test_batches': [6, 8, 10, 12, 14],
        'expected_params': '42.9M',
        'previous_batch': 6
    },
    'YOLOv11m_Baseline': {
        'config': None,
        'weights': 'yolo11m.pt',
        'test_batches': [8, 10, 12, 14, 16],
        'expected_params': '20.1M',
        'previous_batch': 8
    },
    'YOLOv11m-P2': {
        'config': 'configs/yolo11-p2.yaml',
        'weights': 'yolo11m.pt',
        'test_batches': [6, 8, 10, 12],
        'expected_params': '~20M',
        'previous_batch': 6
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


def test_model_batch(model_name, model_info, batch_size):
    """Test a specific model at a specific batch size"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} at Batch {batch_size}")
    print(f"{'='*70}\n")
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load model
        if model_info['config']:
            if 'yolov8-p2' in model_info['config']:
                config_file = prepare_yolov8_p2_config()
            else:
                config_file = model_info['config']
            print(f"Loading from config: {config_file}")
            model = YOLO(config_file)
            
            # Load weights if specified
            if model_info['weights']:
                print(f"Loading weights: {model_info['weights']}")
                model.load(model_info['weights'])
        else:
            print(f"Loading pretrained: {model_info['weights']}")
            model = YOLO(model_info['weights'])
        
        # Check model size
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"✓ Model loaded: {total_params:,} parameters")
        print(f"  Expected: {model_info['expected_params']}")
        
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
            name=f"batch_test_{model_name.replace(' ', '_').replace('-', '_')}_b{batch_size}",
            exist_ok=True
        )
        
        # Get max VRAM usage
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        
        print(f"\n✅ SUCCESS: {model_name} @ Batch {batch_size}")
        print(f"   VRAM Allocated: {max_memory_allocated:.2f} GB")
        print(f"   VRAM Reserved: {max_memory_reserved:.2f} GB")
        print(f"   Margin: {24 - max_memory_reserved:.2f} GB\n")
        
        # Cleanup
        del model, results
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'batch': batch_size,
            'success': True,
            'vram_allocated': max_memory_allocated,
            'vram_reserved': max_memory_reserved,
            'margin': 24 - max_memory_reserved,
            'params': total_params
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OOM: {model_name} @ Batch {batch_size}")
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'batch': batch_size,
                'success': False,
                'error': 'OOM'
            }
        else:
            raise
    except Exception as e:
        print(f"\n❌ ERROR: {model_name} @ Batch {batch_size}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'batch': batch_size,
            'success': False,
            'error': str(e)
        }


def test_all_models():
    """Test all models to find optimal batch sizes"""
    print("\n" + "="*70)
    print("COMPREHENSIVE BATCH SIZE OPTIMIZATION")
    print("="*70)
    print("\n🎯 Goal: Find optimal batch size for each model on 24GB GPU")
    print("🔄 Strategy: Test progressively until OOM or <2GB margin")
    print("📊 Models to test: 4 (2 baselines + 2 P2 variants)")
    print("\n" + "="*70 + "\n")
    
    all_results = {}
    
    for model_name, model_info in MODELS.items():
        print(f"\n{'#'*70}")
        print(f"# {model_name}")
        print(f"# Previous batch: {model_info['previous_batch']}")
        print(f"# Expected params: {model_info['expected_params']}")
        print(f"{'#'*70}\n")
        
        model_results = []
        max_successful_batch = None
        
        for batch in model_info['test_batches']:
            result = test_model_batch(model_name, model_info, batch)
            model_results.append(result)
            
            if result['success']:
                max_successful_batch = batch
                
                # Stop if margin too tight
                if result.get('margin', 100) < 2.0:
                    print(f"⚠️  Stopping {model_name}: Only {result['margin']:.2f}GB margin")
                    break
            else:
                print(f"🛑 Stopping {model_name}: Batch {batch} failed")
                break
        
        all_results[model_name] = {
            'results': model_results,
            'max_batch': max_successful_batch,
            'previous_batch': model_info['previous_batch']
        }
    
    return all_results


def analyze_results(all_results):
    """Analyze results and recommend batch size strategy"""
    print("\n" + "="*70)
    print("BATCH SIZE TEST RESULTS - ALL MODELS")
    print("="*70 + "\n")
    
    # Print individual results
    for model_name, data in all_results.items():
        print(f"\n{model_name}:")
        print(f"{'  Batch':<10} {'Status':<12} {'VRAM (GB)':<12} {'Margin (GB)':<12}")
        print("  " + "-" * 60)
        
        for r in data['results']:
            if r['success']:
                print(f"  {r['batch']:<10} ✅ Success    {r['vram_reserved']:<12.2f} {r['margin']:<12.2f}")
            else:
                print(f"  {r['batch']:<10} ❌ {r.get('error', 'Failed'):<10} -            -")
        
        print(f"\n  Maximum batch: {data['max_batch']}")
        print(f"  Previous batch: {data['previous_batch']}")
        
        if data['max_batch'] and data['max_batch'] != data['previous_batch']:
            change = ((data['max_batch'] - data['previous_batch']) / data['previous_batch']) * 100
            print(f"  Change: {change:+.1f}%")
    
    # Find common maximum
    max_batches = [data['max_batch'] for data in all_results.values() if data['max_batch']]
    common_max = min(max_batches) if max_batches else None
    
    print("\n" + "="*70)
    print("BATCH SIZE STRATEGY RECOMMENDATION")
    print("="*70 + "\n")
    
    if common_max:
        print(f"🎯 COMMON MAXIMUM BATCH: {common_max}\n")
        print("Strategy 1: UNIFIED BATCH (Fair Comparison)")
        print(f"  Use batch={common_max} for ALL models")
        print("  ✅ Pros: True apples-to-apples comparison")
        print("  ⚠️  Cons: Some models underutilize GPU\n")
        
        print("Strategy 2: INDIVIDUAL OPTIMUMS (Maximum Performance)")
        print("  Use each model's maximum stable batch:")
        for model_name, data in all_results.items():
            if data['max_batch']:
                print(f"    {model_name}: batch={data['max_batch']}")
        print("  ✅ Pros: Maximum GPU utilization per model")
        print("  ⚠️  Cons: Batch size becomes confounding variable\n")
        
        print("Strategy 3: GRADIENT ACCUMULATION SIMULATION")
        print(f"  Base batch: {common_max}")
        print("  For models with higher max batch:")
        for model_name, data in all_results.items():
            if data['max_batch'] and data['max_batch'] > common_max:
                accumulate = data['max_batch'] // common_max
                print(f"    {model_name}: Run {accumulate}× iterations, average gradients manually")
        print("  ✅ Pros: Simulates larger batch with available VRAM")
        print("  ⚠️  Cons: Requires custom training loop\n")
    
    print("="*70)
    print("GRADIENT ACCUMULATION ALTERNATIVES")
    print("="*70 + "\n")
    
    print("❌ Native Ultralytics Accumulation: NOT SUPPORTED")
    print("   'accumulate' parameter removed from public API\n")
    
    print("✅ Alternative 1: Manual Gradient Accumulation")
    print("   Modify Ultralytics source or use custom training loop")
    print("   Complexity: High, maintainability: Low\n")
    
    print("✅ Alternative 2: Use Maximum Stable Batch")
    print("   Each model uses its own maximum batch")
    print("   Complexity: Low, performance: Optimal per model\n")
    
    print("✅ Alternative 3: Conservative Unified Batch")
    print(f"   All models use batch={common_max}")
    print("   Complexity: Low, comparison: Fair\n")
    
    print("🎯 RECOMMENDED: Strategy 2 + Document Batch Differences")
    print("   • Use maximum stable batch for each model")
    print("   • Document batch sizes in results")
    print("   • Acknowledge batch as variable in analysis")
    print("   • Focus on architecture comparison, not absolute mAP\n")
    
    return common_max


if __name__ == "__main__":
    try:
        print("\n🚀 Starting comprehensive batch size testing...")
        print("⏱️  This will take 30-60 minutes depending on failures\n")
        
        results = test_all_models()
        common_max = analyze_results(results)
        
        print("\n✅ Batch size optimization complete!")
        print(f"\n📊 Results saved in terminal output")
        print(f"💡 Next: Choose strategy and update training scripts\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user\n")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}\n")
        import traceback
        traceback.print_exc()
