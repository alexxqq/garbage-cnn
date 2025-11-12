"""
Dataset Downloader for Garbage Classification
Downloads the TrashNet dataset or other garbage classification datasets
"""
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url, dest_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def download_trashnet_dataset(data_dir="data"):
    """Download TrashNet dataset from GitHub"""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # TrashNet dataset URLs
    dataset_url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
    zip_path = data_path / "dataset-resized.zip"
    
    if (data_path / "dataset-resized").exists():
        print("Dataset already exists. Skipping download.")
        return str(data_path / "dataset-resized")
    
    print("Downloading TrashNet dataset...")
    download_file(dataset_url, zip_path)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
    # Clean up zip file
    zip_path.unlink()
    
    print(f"Dataset downloaded to {data_path / 'dataset-resized'}")
    return str(data_path / "dataset-resized")

def organize_dataset(dataset_path, output_path="data/processed"):
    """Organize dataset into train/val/test splits"""
    from sklearn.model_selection import train_test_split
    import shutil
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Garbage classes from TrashNet
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Organize images
    all_images = []
    all_labels = []
    
    for cls_idx, cls in enumerate(classes):
        cls_path = dataset_path / cls
        if not cls_path.exists():
            continue
        
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
        all_images.extend(images)
        all_labels.extend([cls_idx] * len(images))
    
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Copy files to organized structure
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    for split_name, (images, labels) in splits.items():
        for img_path, label in zip(images, labels):
            cls_name = classes[label]
            dest = output_path / split_name / cls_name / img_path.name
            shutil.copy2(img_path, dest)
    
    print(f"Dataset organized into {output_path}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return str(output_path)

if __name__ == "__main__":
    dataset_path = download_trashnet_dataset()
    organize_dataset(dataset_path)

