"""
Project Status Checker
Run this to verify your setup and see what needs to be done next
"""

import sys
from pathlib import Path
import importlib.util

def check_mark(condition):
    return "✅" if condition else "❌"

def check_warning(condition):
    return "⚠️" if condition else "✅"

print("="*60)
print("FOD DETECTION PROJECT - STATUS CHECK")
print("="*60)
print()

# 1. Check Python version
print("1. Python Environment")
print(f"   Version: {sys.version.split()[0]}")
print(f"   {check_mark(sys.version_info >= (3, 9))} Python 3.9+ requirement met")
print()

# 2. Check critical packages
print("2. Core Dependencies")
packages = {
    'torch': 'PyTorch',
    'ultralytics': 'Ultralytics YOLO',
    'sahi': 'SAHI',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'matplotlib': 'Matplotlib',
}

for pkg, name in packages.items():
    try:
        spec = importlib.util.find_spec(pkg)
        installed = spec is not None
        if installed:
            try:
                module = __import__(pkg)
                version = getattr(module, '__version__', 'unknown')
                print(f"   {check_mark(True)} {name}: {version}")
            except:
                print(f"   {check_mark(True)} {name}: installed")
        else:
            print(f"   {check_mark(False)} {name}: NOT INSTALLED")
    except:
        print(f"   {check_mark(False)} {name}: NOT INSTALLED")
print()

# 3. Check GPU
print("3. GPU Availability")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   {check_mark(True)} CUDA Available: {gpu_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print(f"   {check_warning(False)} No CUDA detected (CPU-only training)")
except:
    print(f"   ❓ PyTorch not installed - cannot check GPU")
print()

# 4. Check dataset
print("4. Dataset Status")
# Get the script's directory and go up to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
dataset_root = project_root / "data" / "FOD-A"
if dataset_root.exists():
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"
    
    checks = {
        "FOD-A root": dataset_root.exists(),
        "Train images": train_images.exists(),
        "Train labels": train_labels.exists(),
        "Val images": val_images.exists(),
        "Val labels": val_labels.exists(),
    }
    
    for item, exists in checks.items():
        status = check_mark(exists)
        print(f"   {status} {item}")
        
        if exists and "images" in item:
            path = train_images if "Train" in item else val_images
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"      └─ {count} images found")
else:
    print(f"   {check_mark(False)} FOD-A dataset not found at: {dataset_root.absolute()}")
    print(f"   📥 ACTION REQUIRED: Download and place FOD-A dataset")
print()

# 5. Check project files
print("5. Project Files")
required_files = {
    project_root / "utils" / "dataset_loader.py": "Dataset loader",
    project_root / "notebooks" / "01_dataset_exploration.ipynb": "Exploration notebook",
    project_root / "requirements.txt": "Requirements file",
    project_root / "README.md": "Documentation",
}

for file_path, description in required_files.items():
    exists = file_path.exists()
    print(f"   {check_mark(exists)} {description}")
print()

# 6. Summary
print("="*60)
print("SUMMARY & NEXT STEPS")
print("="*60)

all_deps_installed = all([importlib.util.find_spec(pkg) is not None for pkg in ['torch', 'ultralytics', 'sahi']])
dataset_exists = dataset_root.exists()

if all_deps_installed and dataset_exists:
    print("✅ All systems ready! You can start training.")
    print()
    print("Next: Run the dataset exploration notebook:")
    print("   jupyter notebook notebooks/01_dataset_exploration.ipynb")
elif all_deps_installed:
    print("⚠️ Dependencies installed, but dataset missing.")
    print()
    print("Next: Obtain FOD-A dataset and place in data/FOD-A/")
    print("   See README.md for dataset structure")
elif dataset_exists:
    print("⚠️ Dataset found, but dependencies incomplete.")
    print()
    print("Next: Install missing packages:")
    print("   .\\setup\\install_manual.ps1")
else:
    print("❌ Setup incomplete.")
    print()
    print("Next steps:")
    print("   1. Run: .\\setup\\install_dependencies.ps1")
    print("   2. Obtain FOD-A dataset")
    print("   3. Place dataset in data/FOD-A/")

print()
