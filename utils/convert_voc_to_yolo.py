"""
Convert Pascal VOC format (XML) to YOLO format (TXT)
Converts FOD-A dataset annotations for use with Ultralytics YOLO
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml


# FOD-A classes (actual names from the dataset)
CLASSES = [
    'bolt', 'nut', 'screw', 'washer', 'nail',  # Hardware (0-4)
    'hammer', 'pliers', 'wrench', 'screwdriver',  # Tools (5-8)
    'luggagetag', 'luggagepart', 'pen', 'bottle', 'sodacan',  # Personal (9-13)
    'metalsheet', 'metalpart', 'plasticpart', 'wire', 'rock', 'wood',  # Materials (14-19)
    'rubber', 'glass', 'paper', 'cloth', 'foam',  # Additional materials (20-24)
    'cable', 'tape', 'bag', 'glove', 'cap',  # Misc (25-29)
    'adjustableclamp', 'adjustablewrench', 'battery', 'boltnutset', 'boltwasher',  # Additional (30-34)
    'clamppart', 'cutter', 'fuelcap', 'hose', 'label', 'paintchip'  # More (35-40)
]


def parse_voc_annotation(xml_file):
    """Parse Pascal VOC XML annotation file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get all objects
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower().replace(' ', '_')
        
        # Get bounding box
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'class': class_name,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    
    return width, height, objects


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert Pascal VOC bbox to YOLO format
    VOC: (xmin, ymin, xmax, ymax) in pixels
    YOLO: (x_center, y_center, width, height) normalized 0-1
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center and dimensions
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def convert_voc_to_yolo(voc_root, yolo_root):
    """
    Convert entire Pascal VOC dataset to YOLO format
    
    Args:
        voc_root: Path to VOC dataset root (contains JPEGImages, Annotations, etc.)
        yolo_root: Path to output YOLO format dataset
    """
    voc_root = Path(voc_root)
    yolo_root = Path(yolo_root)
    
    print("="*60)
    print("PASCAL VOC TO YOLO CONVERTER")
    print("="*60)
    print(f"\n📂 Input (VOC): {voc_root.absolute()}")
    print(f"📂 Output (YOLO): {yolo_root.absolute()}")
    print()
    
    # Check VOC structure
    images_dir = voc_root / 'JPEGImages'
    annotations_dir = voc_root / 'Annotations'
    imagesets_dir = voc_root / 'ImageSets' / 'Main'
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        print("\n📝 Expected VOC structure:")
        print("FOD-A/")
        print("├── JPEGImages/")
        print("├── Annotations/")
        print("└── ImageSets/Main/")
        return False
    
    print("✅ VOC structure validated")
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        (yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("✅ YOLO directories created")
    print()
    
    # Build class name to ID mapping
    class_to_id = {name: idx for idx, name in enumerate(CLASSES)}
    
    # Track unknown classes
    unknown_classes = set()
    
    # Process each split (train, val)
    stats = {'train': {'images': 0, 'annotations': 0}, 'val': {'images': 0, 'annotations': 0}}
    
    # Check what split files are available
    split_mapping = {}
    if (imagesets_dir / 'train.txt').exists() and (imagesets_dir / 'val.txt').exists():
        split_mapping = {'train': 'train.txt', 'val': 'val.txt'}
        print("✅ Found train.txt and val.txt")
    elif (imagesets_dir / 'trainval.txt').exists() and (imagesets_dir / 'test.txt').exists():
        # Use trainval for training and test for validation
        split_mapping = {'train': 'trainval.txt', 'val': 'test.txt'}
        print("✅ Found trainval.txt and test.txt (using trainval as train, test as val)")
    elif (imagesets_dir / 'trainval.txt').exists():
        # Split trainval into train and val (80/20)
        print("⚠️ Only trainval.txt found, will split 80/20 into train/val")
        split_mapping = {'trainval': 'trainval.txt'}
    else:
        print("❌ No split files found in ImageSets/Main/")
        return False
    
    for split_name, split_file_name in split_mapping.items():
        split_file = imagesets_dir / split_file_name
        
        if not split_file.exists():
            print(f"⚠️ Split file not found: {split_file}")
            continue
        
        # Determine target split (train or val)
        if split_name == 'trainval':
            # Read all IDs and split them
            with open(split_file, 'r') as f:
                all_image_ids = [line.strip() for line in f if line.strip()]
            
            # 80/20 split
            split_idx = int(len(all_image_ids) * 0.8)
            splits_data = {
                'train': all_image_ids[:split_idx],
                'val': all_image_ids[split_idx:]
            }
            print(f"📊 Split trainval.txt: {len(splits_data['train'])} train, {len(splits_data['val'])} val")
        else:
            # Direct mapping
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f if line.strip()]
            splits_data = {split_name: image_ids}
        
        # Process each split's images
        for target_split, image_ids in splits_data.items():
            print(f"📊 Processing {target_split} split ({len(image_ids)} images)...")
            
            for image_id in tqdm(image_ids, desc=f"Converting {target_split}"):
                # Find image file (could be .jpg or .png)
                image_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    candidate = images_dir / f"{image_id}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break
                
                if image_file is None:
                    print(f"⚠️ Image not found: {image_id}")
                    continue
                
                # Parse annotation
                xml_file = annotations_dir / f"{image_id}.xml"
                if not xml_file.exists():
                    print(f"⚠️ Annotation not found: {xml_file}")
                    continue
                
                try:
                    img_width, img_height, objects = parse_voc_annotation(xml_file)
                except Exception as e:
                    print(f"⚠️ Error parsing {xml_file}: {e}")
                    continue
                
                # Copy image
                dest_image = yolo_root / 'images' / target_split / image_file.name
                shutil.copy2(image_file, dest_image)
                stats[target_split]['images'] += 1
                
                # Convert annotations to YOLO format
                yolo_annotations = []
                for obj in objects:
                    class_name = obj['class']
                    
                    # Map class name to ID
                    if class_name not in class_to_id:
                        unknown_classes.add(class_name)
                        continue  # Skip unknown classes
                    
                    class_id = class_to_id[class_name]
                    
                    # Convert bbox
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        obj['bbox'], img_width, img_height
                    )
                    
                    # YOLO format: class_id x_center y_center width height
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    stats[target_split]['annotations'] += 1
                
                # Write YOLO annotation file
                if yolo_annotations:  # Only write if there are valid annotations
                    label_file = yolo_root / 'labels' / target_split / f"{image_id}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
    
    # Report unknown classes
    if unknown_classes:
        print(f"\n⚠️ Found {len(unknown_classes)} unknown classes:")
        for cls in sorted(unknown_classes):
            print(f"   - {cls}")
        print("   These were skipped. You may need to update the CLASSES list.")
    
    # Create data.yaml
    data_yaml = {
        'path': str(yolo_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_file = yolo_root / 'data.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\n📊 Statistics:")
    print(f"   Train: {stats['train']['images']} images, {stats['train']['annotations']} annotations")
    print(f"   Val:   {stats['val']['images']} images, {stats['val']['annotations']} annotations")
    print(f"\n✅ Created: {yaml_file}")
    print()
    print("📊 Next Steps:")
    print("1. Validate: python utils\\check_status.py")
    print("2. Explore: jupyter notebook notebooks\\01_dataset_exploration.ipynb")
    print()
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Pascal VOC to YOLO format')
    parser.add_argument('--voc-root', type=str, default=None,
                       help='Path to VOC dataset root (auto-detect if not specified)')
    parser.add_argument('--yolo-root', type=str, default='data/FOD-A',
                       help='Path to output YOLO dataset')
    
    args = parser.parse_args()
    
    # Auto-detect VOC root
    if args.voc_root is None:
        # Try common locations
        data_dir = Path('data')
        candidates = [
            data_dir / 'FOD-A',
            data_dir / 'FOD-A-VOC',
            data_dir / 'fod-a',
        ]
        
        # Also check for any directory with "FOD" in name
        if data_dir.exists():
            candidates.extend([d for d in data_dir.iterdir() if d.is_dir() and 'fod' in d.name.lower()])
        
        voc_root = None
        for candidate in candidates:
            if (candidate / 'JPEGImages').exists():
                voc_root = candidate
                print(f"✅ Auto-detected VOC root: {voc_root}")
                break
        
        if voc_root is None:
            print("❌ Could not auto-detect VOC dataset location")
            print("\n📝 Please specify with --voc-root argument")
            print("Or ensure Pascal VOC format dataset is in data/ directory")
            exit(1)
    else:
        voc_root = Path(args.voc_root)
    
    yolo_root = Path(args.yolo_root)
    
    success = convert_voc_to_yolo(voc_root, yolo_root)
    
    if not success:
        exit(1)
