"""
FOD-A Dataset Loader and Utilities
Handles loading, parsing, and validating the FOD-A dataset in YOLO format
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2


class FODDatasetConfig:
    """Configuration class for FOD-A dataset"""
    
    # 41 FOD classes (actual dataset contains more than the paper's 31)
    CLASSES = [
        'adjustableclamp', 'adjustablewrench', 'backpack', 'battery', 'bolt',
        'boltnutset', 'boltwasher', 'bottle', 'clamp', 'clamppart',
        'cloth', 'cutter', 'fuelcap', 'glove', 'hammer',
        'hose', 'label', 'luggagetag', 'metalbracket', 'metalsheet',
        'nail', 'nut', 'paintchip', 'pen', 'pliers',
        'poweradapter', 'rope', 'rubberbelt', 'screw', 'screwdriver',
        'seal', 'sodacan', 'spring', 'tape', 'tie',
        'washer', 'wire', 'woodenpiece', 'woodscrew', 'wrappingpaper',
        'wrench'
    ]
    
    # Environmental conditions
    LIGHT_CONDITIONS = ['Bright', 'Dim', 'Dark']
    WEATHER_CONDITIONS = ['Dry', 'Wet']
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / 'images'
        self.labels_dir = self.dataset_root / 'labels'
        self.config_file = self.dataset_root / 'data.yaml'
        
    def validate_structure(self) -> bool:
        """Validate dataset directory structure"""
        required_dirs = [
            self.images_dir / 'train',
            self.images_dir / 'val',
            self.labels_dir / 'train',
            self.labels_dir / 'val',
        ]
        
        missing = [d for d in required_dirs if not d.exists()]
        if missing:
            print(f"❌ Missing directories: {[str(d) for d in missing]}")
            return False
        
        print("✓ Dataset structure valid")
        return True
    
    def create_data_yaml(self, test_split: bool = True):
        """Create data.yaml file for YOLO training"""
        config = {
            'path': str(self.dataset_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if test_split else 'images/val',
            'nc': len(self.CLASSES),
            'names': self.CLASSES
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Created {self.config_file}")
        return config


class FODDatasetLoader:
    """Loader for FOD-A dataset with YOLO format annotations"""
    
    def __init__(self, dataset_root: str):
        self.config = FODDatasetConfig(dataset_root)
        self.dataset_root = Path(dataset_root)
        
    def load_annotation(self, label_path: Path) -> List[Dict]:
        """
        Load YOLO format annotation file
        Format: class_id x_center y_center width height (normalized 0-1)
        """
        if not label_path.exists():
            return []
        
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        'class_id': int(parts[0]),
                        'class_name': self.config.CLASSES[int(parts[0])],
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
        return annotations
    
    def get_bbox_absolute(self, annotation: Dict, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized YOLO bbox to absolute pixel coordinates"""
        x_center = annotation['x_center'] * img_width
        y_center = annotation['y_center'] * img_height
        width = annotation['width'] * img_width
        height = annotation['height'] * img_height
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return x1, y1, x2, y2
    
    def load_image_with_annotations(self, image_path: Path, split: str = 'train') -> Tuple[np.ndarray, List[Dict]]:
        """Load image and its corresponding annotations"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find corresponding label file
        label_filename = image_path.stem + '.txt'
        label_path = self.dataset_root / 'labels' / split / label_filename
        
        annotations = self.load_annotation(label_path)
        return img, annotations
    
    @staticmethod
    def _empty_scale_counts() -> Dict[str, int]:
        return {'small': 0, 'medium': 0, 'large': 0}

    @staticmethod
    def _add_coco_area_bin(counts: Dict[str, int], area_px: float) -> None:
        """COCO scale bins: small <32^2, medium 32^2-96^2, large >=96^2."""
        if area_px < 32 ** 2:
            counts['small'] += 1
        elif area_px < 96 ** 2:
            counts['medium'] += 1
        else:
            counts['large'] += 1

    @staticmethod
    def _add_max_dim_bin(counts: Dict[str, int], width_px: float, height_px: float) -> None:
        """Legacy figure bins based on the longest bbox side, not COCO area."""
        max_dim = max(width_px, height_px)
        if max_dim < 32:
            counts['small'] += 1
        elif max_dim < 96:
            counts['medium'] += 1
        else:
            counts['large'] += 1

    def get_dataset_statistics(self, split: str = 'train', imgsz: int = 640) -> Dict:
        """
        Calculate dataset statistics.

        Scale definitions are intentionally kept separate:
        - coco_area_objects_original: COCO area bins at source image resolution.
        - coco_area_objects_letterbox: COCO area bins after YOLO square letterbox resize.
        - max_dim_objects_original: legacy Figure-style bins using longest bbox side.
        - relative_area_bins: cumulative relative-frame-area bins; not a small-object definition.
        """
        images_dir = self.dataset_root / 'images' / split
        labels_dir = self.dataset_root / 'labels' / split
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        stats = {
            'total_images': len(image_files),
            'total_annotations': 0,
            'class_counts': {cls: 0 for cls in self.config.CLASSES},
            'coco_area_objects_original': self._empty_scale_counts(),
            'coco_area_objects_letterbox': self._empty_scale_counts(),
            'max_dim_objects_original': self._empty_scale_counts(),
            'max_dim_objects_letterbox': self._empty_scale_counts(),
            'relative_area_bins': {
                '<0.1%': 0,
                '<0.5%': 0,
                '<1%': 0,
                '<2%': 0,
                '<5%': 0,
                '<10%': 0,
                '<20%': 0,
            },
            'bbox_areas': [],
            'bbox_relative_areas': [],
        }
        
        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + '.txt')
            annotations = self.load_annotation(label_path)
            
            if annotations:
                img = Image.open(img_path)
                img_w, img_h = img.size
                
                for ann in annotations:
                    stats['total_annotations'] += 1
                    stats['class_counts'][ann['class_name']] += 1
                    
                    # Calculate absolute bbox size
                    width_px = ann['width'] * img_w
                    height_px = ann['height'] * img_h
                    area = width_px * height_px
                    relative_area = ann['width'] * ann['height']
                    stats['bbox_areas'].append(area)
                    stats['bbox_relative_areas'].append(relative_area)
                    
                    letterbox_scale = min(imgsz / img_h, imgsz / img_w)
                    letterbox_width_px = width_px * letterbox_scale
                    letterbox_height_px = height_px * letterbox_scale
                    letterbox_area = letterbox_width_px * letterbox_height_px

                    self._add_coco_area_bin(stats['coco_area_objects_original'], area)
                    self._add_coco_area_bin(stats['coco_area_objects_letterbox'], letterbox_area)
                    self._add_max_dim_bin(stats['max_dim_objects_original'], width_px, height_px)
                    self._add_max_dim_bin(stats['max_dim_objects_letterbox'], letterbox_width_px, letterbox_height_px)

                    for label, threshold in [
                        ('<0.1%', 0.001),
                        ('<0.5%', 0.005),
                        ('<1%', 0.01),
                        ('<2%', 0.02),
                        ('<5%', 0.05),
                        ('<10%', 0.10),
                        ('<20%', 0.20),
                    ]:
                        if relative_area < threshold:
                            stats['relative_area_bins'][label] += 1

        # Backward compatibility for older notebook cells. These are legacy
        # max-dimension bins, not COCO AP_small/AP_medium/AP_large bins.
        legacy_counts = stats['max_dim_objects_original']
        stats['small_objects'] = legacy_counts['small']
        stats['medium_objects'] = legacy_counts['medium']
        stats['large_objects'] = legacy_counts['large']
        
        return stats
    
    def visualize_sample(self, image_path: Path, split: str = 'train', save_path: Optional[Path] = None):
        """Visualize image with bounding boxes"""
        img, annotations = self.load_image_with_annotations(image_path, split)
        img_h, img_w = img.shape[:2]
        
        # Draw bounding boxes
        for ann in annotations:
            x1, y1, x2, y2 = self.get_bbox_absolute(ann, img_w, img_h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{ann['class_name']}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        if save_path:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), img_bgr)
        
        return img


def validate_dataset(dataset_root: str) -> bool:
    """Validate FOD-A dataset structure and create config files"""
    config = FODDatasetConfig(dataset_root)
    
    print("🔍 Validating FOD-A Dataset...")
    print(f"📁 Root: {dataset_root}")
    print()
    
    if not config.validate_structure():
        return False
    
    # Create data.yaml if it doesn't exist
    if not config.config_file.exists():
        print("📝 Creating data.yaml configuration...")
        config.create_data_yaml()
    
    # Load and display basic stats
    loader = FODDatasetLoader(dataset_root)
    
    for split in ['train', 'val']:
        split_dir = config.images_dir / split
        if split_dir.exists():
            num_images = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
            print(f"✓ {split.upper()}: {num_images} images")
    
    print()
    print("✅ Dataset validation complete!")
    return True


if __name__ == "__main__":
    # Test the loader
    dataset_root = "../data/FOD-A"
    validate_dataset(dataset_root)
