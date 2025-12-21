# FOD Detection Pipeline
## Advanced Foreign Object Debris Detection using YOLOv11 and SAHI

This repository implements a comprehensive 3-week research project to advance FOD detection beyond current state-of-the-art, building on the work by Farooq et al. (2024).

---

## Project Structure

```
Code/
├── setup/                  # Installation and setup scripts
├── data/                   # Dataset directory (FOD-A)
│   └── FOD-A/             # Place dataset here
│       ├── images/        # Raw images
│       ├── labels/        # YOLO format annotations (.txt)
│       └── metadata.csv   # Environmental conditions (Bright/Dim/Dark, Dry/Wet)
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── configs/               # YAML configurations for model architectures
├── utils/                 # Helper scripts (data loaders, visualization, metrics)
├── models/                # Trained model checkpoints
├── results/               # Evaluation results and logs
└── requirements.txt       # Python dependencies
```

---

## Quick Start

### 1. Environment Setup

**Windows PowerShell:**
```powershell
cd "d:\Zaryab\Course Work\Machine Learning\ML Project\Code"
.\setup\install_dependencies.ps1
```

**Manual Installation:**
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the **FOD-A dataset** from:
- [arXiv Paper](https://arxiv.org/abs/2110.03072)
- Contact authors if dataset not publicly hosted

**Expected Structure:**
```
data/FOD-A/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml  # YOLO dataset config
```

**YOLO Format (.txt annotation example):**
```
class_id x_center y_center width height
15 0.532 0.687 0.024 0.031
```
(Normalized coordinates: 0-1 range)

### 3. Verify Setup

Run the dataset exploration notebook:
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

---

## Dataset Information

### FOD-A Dataset Overview
- **Classes**: 31 object categories (Bolt, Nut, Screw, Washer, Wire, etc.)
- **Instances**: 30,000+ bounding box annotations
- **Challenge**: 85%+ of objects occupy <20% of image area (small object detection)
- **Environmental Conditions**:
  - **Light**: Bright, Dim, Dark
  - **Weather**: Dry, Wet

### Class Distribution (Imbalanced)
Common classes: Nut, Bolt, Screw, Metal Part  
Rare classes: Hammer, Pliers, Specific tool types

---

## 3-Week Project Timeline

### **Week 1: Foundation** (Days 1-5)
- ✅ Environment setup and dataset validation
- ✅ Implement YOLOv8-P2 architecture (shallow detection head)
- ✅ Train baseline YOLOv8-P2 model
- **Target**: Reproduce 93.8% mAP benchmark

### **Week 2: Innovation** (Days 6-10)
- Train YOLOv11m (vanilla)
- Train YOLOv11-P2 (custom architecture)
- Implement SAHI inference pipeline
- Compare inference speed vs accuracy trade-offs

### **Week 3: Validation** (Days 11-15)
- Adversarial testing (Dark/Wet subsets)
- Per-class confusion matrix analysis
- Generate comparative performance tables
- Final report and operational recommendations

---

## Key Technical Concepts

### 1. The P2 Detection Head
**Problem**: Feature vanishing in deep layers (stride 32) loses small object spatial information.

**Solution**: Add a shallow detection head at P2 (stride 4) that processes high-resolution features.

**Implementation**: Modify YOLO YAML config to add P2/4 layer (see `configs/yolov8-p2.yaml`)

### 2. SAHI (Slicing Aided Hyper Inference)
**Problem**: Small objects become even smaller in resized input images (e.g., 4K → 640px).

**Solution**: Slice high-res images into overlapping patches, detect in each patch, then merge results.

**Trade-off**: Higher mAP but lower FPS (linear with slice count)

### 3. YOLOv11 Innovations
- **C3k2 blocks**: 22% fewer parameters than YOLOv8
- **C2PSA module**: Spatial attention for small object focus
- **Improved feature extraction**: Better handling of tiny debris

---

## Critical Metrics

### Primary Metrics:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **AP_small**: Average Precision for objects <32×32 pixels (most important for FOD)
- **Recall@0.5**: Percentage of debris items detected
- **FPS**: Frames per second (real-time viability)

### Target Benchmarks:
| Model | mAP | AP_small | FPS | Params |
|-------|-----|----------|-----|--------|
| Improved YOLOv8 (Paper) | 93.8% | High | ~60 | ~25M |
| YOLOv11-P2 (Target) | >94% | Higher | >50 | <20M |

---

## Usage Examples

### Training YOLOv8-P2
```python
from ultralytics import YOLO

model = YOLO('configs/yolov8-p2.yaml')
model.train(
    data='data/FOD-A/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```

### SAHI Inference
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='models/yolov8-p2-best.pt',
    confidence_threshold=0.3,
    device='cuda:0'
)

result = get_sliced_prediction(
    'test_image.jpg',
    model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

---

## Hardware Requirements

### Minimum (Slow Training):
- CPU: 8-core processor
- RAM: 16GB
- GPU: None (CPU training possible but very slow)

### Recommended:
- CPU: 16-core processor
- RAM: 32GB
- GPU: NVIDIA RTX 3060 (12GB VRAM) or better
- Storage: 100GB SSD for dataset and models

### Optimal:
- GPU: NVIDIA RTX 4090 (24GB VRAM) or A100
- Multi-GPU setup for parallel experiments

### Cloud Alternatives:
- **Google Colab Pro**: Free GPU (T4), upgrade for A100
- **Kaggle Notebooks**: Free P100 GPU (30hrs/week)
- **AWS/GCP**: On-demand GPU instances

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in training config
```python
model.train(batch=8)  # Instead of batch=16
```

### Issue: Can't reproduce 93.8% mAP
**Check**:
1. Dataset split matches paper (70/20/10 train/val/test)
2. Image size is correct (640×640 or 1280×1280)
3. Environmental stratification in validation set
4. Augmentation settings (Mosaic, MixUp enabled)

### Issue: SAHI too slow
**Optimization**:
1. Reduce slice overlap (0.2 → 0.1)
2. Increase slice size (640 → 1024)
3. Use postprocess_match_threshold to filter duplicates

---

## References

1. Farooq, M., Muaz, M., et al. (2024). "An improved YOLOv8 for foreign object debris detection with optimized architecture for small objects." *Multimedia Tools and Applications*.

2. Akyon, F. C., et al. (2022). "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection." *arXiv:2202.06934*.

3. Jocher, G., et al. (2024). "Ultralytics YOLOv11." GitHub repository.

4. Özbilge, E., et al. (2021). "FOD-A: A Dataset for Foreign Object Debris in Airports." *arXiv:2110.03072*.

---

## License

This project is for academic research purposes. Dataset and model usage must comply with respective licenses.

---

## Contact

For questions or collaboration:
- Email: [Your Email]
- GitHub Issues: [Repository URL]

---

**Last Updated**: December 18, 2025
