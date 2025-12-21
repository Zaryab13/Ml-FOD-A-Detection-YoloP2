"""
FOD-A Dataset Downloader
Automatically downloads and extracts the FOD-A dataset from Google Drive
"""

import os
import sys
from pathlib import Path
import zipfile

def download_fod_dataset(version='voc'):
    """
    Download FOD-A dataset from Google Drive
    
    Args:
        version (str): 'voc' for Pascal VOC format (412MB) or 'original' for original format (8.3GB)
    """
    try:
        import gdown
    except ImportError:
        print("❌ gdown not installed. Installing now...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    # Define paths
    data_dir = Path(__file__).parent
    
    # Google Drive file IDs
    file_ids = {
        'voc': '1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x',  # Pascal VOC 412MB
        'original': '1lLBJXXaQCWaFa-1MeLAANPpSwMhCJqGh'  # Original 8.3GB
    }
    
    sizes = {
        'voc': '412 MB',
        'original': '8.3 GB'
    }
    
    if version not in file_ids:
        print(f"❌ Invalid version: {version}. Choose 'voc' or 'original'")
        return False
    
    print("="*60)
    print("FOD-A DATASET DOWNLOADER")
    print("="*60)
    print(f"\n📥 Downloading: {version.upper()} format ({sizes[version]})")
    print(f"📁 Target directory: {data_dir.absolute()}")
    print()
    
    # Download file
    output_zip = data_dir / f"FOD-A-{version}.zip"
    file_id = file_ids[version]
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        print("⏳ Downloading from Google Drive...")
        print("(This may take 5-30 minutes depending on your internet speed)")
        print()
        
        gdown.download(url, str(output_zip), quiet=False)
        
        if not output_zip.exists():
            print("\n❌ Download failed! File not found.")
            print("\n📝 Manual Download Instructions:")
            print(f"1. Open: https://drive.google.com/file/d/{file_id}/view")
            print("2. Click 'Download'")
            print(f"3. Save to: {output_zip}")
            print("4. Run this script again")
            return False
        
        print("\n✅ Download complete!")
        print(f"📦 File size: {output_zip.stat().st_size / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Download error: {e}")
        print("\n📝 Try manual download from browser:")
        print(f"Link: https://drive.google.com/file/d/{file_id}/view")
        return False
    
    # Extract
    print("\n⏳ Extracting dataset...")
    extract_dir = data_dir / "FOD-A"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} files in archive")
            
            # Extract
            zip_ref.extractall(data_dir)
        
        print("✅ Extraction complete!")
        
        # Clean up zip file
        print("\n🗑️ Cleaning up zip file...")
        output_zip.unlink()
        print("✅ Zip file deleted")
        
    except Exception as e:
        print(f"\n❌ Extraction error: {e}")
        print(f"\n📝 Manual extraction:")
        print(f"1. Locate: {output_zip}")
        print(f"2. Right-click → Extract All → {extract_dir}")
        return False
    
    # Verify structure
    print("\n🔍 Verifying dataset structure...")
    
    # Find the extracted folder (might be nested)
    extracted_folders = [f for f in data_dir.iterdir() if f.is_dir() and 'FOD' in f.name]
    
    if extracted_folders:
        print(f"✅ Found: {extracted_folders[0].name}")
        
        # List contents
        contents = list(extracted_folders[0].iterdir())
        print(f"\n📂 Dataset structure:")
        for item in contents[:10]:  # Show first 10 items
            print(f"   - {item.name}")
        if len(contents) > 10:
            print(f"   ... and {len(contents) - 10} more items")
    
    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*60)
    print("\n📊 Next Steps:")
    print("1. Convert to YOLO format:")
    print("   python utils\\convert_voc_to_yolo.py")
    print()
    print("2. Validate dataset:")
    print("   python utils\\check_status.py")
    print()
    print("3. Explore in Jupyter:")
    print("   jupyter notebook notebooks\\01_dataset_exploration.ipynb")
    print()
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download FOD-A dataset from Google Drive')
    parser.add_argument('--version', type=str, default='voc', 
                       choices=['voc', 'original'],
                       help='Dataset version: voc (412MB) or original (8.3GB)')
    
    args = parser.parse_args()
    
    success = download_fod_dataset(args.version)
    
    if not success:
        sys.exit(1)
