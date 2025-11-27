#!/usr/bin/env python3
"""
Production Training Script
Trains garbage classification model with MLflow tracking
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.dataset_loader import DatasetLoader
from src.data.dataset import GarbageDataset, get_transforms
from src.models.model_factory import create_model
from src.models.trainer import Trainer

def main():
    # Load configuration
    config = Config("configs/config.yaml")
    
    # Setup logging
    logger = setup_logger(
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file')
    )
    
    logger.info("=" * 60)
    logger.info("GARBAGE CLASSIFICATION TRAINING")
    logger.info("=" * 60)
    
    # Device
    device_cfg = config.training['device']
    if device_cfg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    
    logger.info(f"Using device: {device}")
    
    # Load and prepare datasets
    logger.info("Loading datasets...")
    loader = DatasetLoader(config)
    dataset_paths = loader.load_all_datasets()
    processed_dir = loader.organize_datasets(dataset_paths)
    
    # Create datasets
    train_cfg = config.training
    transform_train = get_transforms(
        image_size=train_cfg['image_size'],
        is_training=True,
        augmentation_config=train_cfg['augmentation']
    )
    transform_val = get_transforms(
        image_size=train_cfg['image_size'],
        is_training=False
    )
    
    train_dataset = GarbageDataset(
        str(processed_dir / "train"),
        transform=transform_train,
        classes=config.model['classes']
    )
    val_dataset = GarbageDataset(
        str(processed_dir / "val"),
        transform=transform_val,
        classes=config.model['classes']
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for macOS compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info(f"Creating {config.model['architecture']} model...")
    model = create_model(
        architecture=config.model['architecture'],
        num_classes=len(config.model['classes']),
        pretrained=config.model['pretrained'],
        weights=config.model.get('weights')
    ).to(device)
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
