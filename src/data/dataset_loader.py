"""Dataset loading and preparation"""
import kagglehub
import requests
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load and prepare datasets from multiple sources"""
    
    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.data['root'])
        self.processed_dir = Path(config.data['processed_dir'])
        self.raw_dir = Path(config.data['raw_dir'])
        
        # Create directories
        self.data_root.mkdir(exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def download_trashnet(self) -> Path:
        """Download TrashNet dataset from GitHub"""
        dataset_path = self.raw_dir / "trashnet"
        zip_path = self.raw_dir / "dataset-resized.zip"
        
        if (dataset_path / "dataset-resized").exists():
            logger.info("TrashNet dataset already exists")
            return dataset_path / "dataset-resized"
        
        logger.info("Downloading TrashNet dataset...")
        url = self.config.data['datasets']['trashnet']['url']
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info("Extracting TrashNet dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        zip_path.unlink()
        logger.info(f"TrashNet downloaded to {dataset_path / 'dataset-resized'}")
        return dataset_path / "dataset-resized"
    
    def download_kaggle(self) -> Path:
        """Download Kaggle dataset"""
        dataset_id = self.config.data['datasets']['kaggle']['dataset_id']
        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        
        try:
            path = kagglehub.dataset_download(dataset_id)
            logger.info(f"Kaggle dataset downloaded to {path}")
            return Path(path)
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {e}")
            raise
    
    def load_all_datasets(self) -> List[Path]:
        """Load all enabled datasets"""
        datasets = []
        
        if self.config.data['datasets']['trashnet']['enabled']:
            datasets.append(self.download_trashnet())
        
        if self.config.data['datasets']['kaggle']['enabled']:
            datasets.append(self.download_kaggle())
        
        return datasets
    
    def organize_datasets(self, dataset_paths: List[Path]) -> Path:
        """Organize multiple datasets into train/val/test splits"""
        classes = self.config.model['classes']
        splits = self.config.data['splits']
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for cls in classes:
                (self.processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Collect all images
        all_images = []
        all_labels = []
        
        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            logger.info(f"Processing dataset: {dataset_path}")
            
            for cls_idx, cls in enumerate(classes):
                # Try different possible folder structures
                possible_paths = [
                    dataset_path / cls,
                    dataset_path / "dataset-resized" / cls,
                    dataset_path / "garbage-classification" / cls,
                ]
                
                cls_path = None
                for pp in possible_paths:
                    if pp.exists():
                        cls_path = pp
                        break
                
                if cls_path is None:
                    logger.warning(f"Class folder not found for {cls} in {dataset_path}")
                    continue
                
                images = list(cls_path.glob("*.jpg")) + \
                         list(cls_path.glob("*.png")) + \
                         list(cls_path.glob("*.jpeg"))
                
                all_images.extend(images)
                all_labels.extend([cls_idx] * len(images))
                logger.info(f"Found {len(images)} images for class {cls}")
        
        logger.info(f"Total images collected: {len(all_images)}")
        
        # Split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_images,
            all_labels,
            test_size=(1 - splits['train']),
            random_state=splits['random_seed'],
            stratify=all_labels
        )
        
        val_size = splits['val'] / (splits['val'] + splits['test'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_size),
            random_state=splits['random_seed'],
            stratify=y_temp
        )
        
        # Copy files
        splits_data = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for split_name, (images, labels) in splits_data.items():
            for img_path, label in zip(images, labels):
                cls_name = classes[label]
                dest = self.processed_dir / split_name / cls_name / img_path.name
                shutil.copy2(img_path, dest)
        
        logger.info(f"Dataset organized:")
        logger.info(f"  Train: {len(X_train)}")
        logger.info(f"  Val: {len(X_val)}")
        logger.info(f"  Test: {len(X_test)}")
        
        return self.processed_dir

