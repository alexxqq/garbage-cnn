#!/usr/bin/env python3
"""
Test Model on Kaggle Dataset
Tests trained model on the Kaggle garbage classification dataset
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
from torch.utils.data import DataLoader
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.dataset import GarbageDataset, get_transforms
from src.models.model_factory import create_model
from src.models.evaluator import Evaluator

def find_kaggle_dataset():
    """Find Kaggle dataset location"""
    # Try common locations
    base_locations = [
        Path.home() / ".cache" / "kagglehub" / "datasets" / "asdasdasasdas" / "garbage-classification" / "versions" / "2",
        Path("data/raw/kaggle"),
        Path("data/raw") / "garbage-classification",
    ]
    
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    for base_loc in base_locations:
        if not base_loc.exists():
            continue
        
        # Try different nested structures
        possible_paths = [
            base_loc,
            base_loc / "Garbage classification",
            base_loc / "Garbage classification" / "Garbage classification",
        ]
        
        for loc in possible_paths:
            if loc.exists():
                # Check for class folders
                has_classes = any((loc / cls).exists() for cls in classes)
                if has_classes:
                    return loc
    
    return None

def main():
    config = Config("configs/config.yaml")
    logger = setup_logger(level="INFO")
    
    logger.info("=" * 60)
    logger.info("TESTING ON KAGGLE DATASET")
    logger.info("=" * 60)
    
    # Find Kaggle dataset
    kaggle_path = find_kaggle_dataset()
    if not kaggle_path:
        logger.error("❌ Kaggle dataset not found!")
        logger.info("Looking in:")
        logger.info("  - ~/.cache/kagglehub/datasets/...")
        logger.info("  - data/raw/kaggle/")
        logger.info("")
        logger.info("To download Kaggle dataset, run:")
        logger.info("  python prepare_data.py")
        return
    
    logger.info(f"✓ Found Kaggle dataset at: {kaggle_path}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    transform = get_transforms(
        image_size=config.training['image_size'],
        is_training=False
    )
    
    test_dataset = GarbageDataset(
        str(kaggle_path),
        transform=transform,
        classes=config.model['classes']
    )
    
    if len(test_dataset) == 0:
        logger.error("❌ No images found in Kaggle dataset!")
        logger.info(f"Checked path: {kaggle_path}")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Load model
    model_path = Path(config.training['save_dir']) / f"best_{config.model['architecture']}.pth"
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Train a model first: python train.py")
        return
    
    logger.info(f"Loading model from {model_path}")
    model = create_model(
        architecture=config.model['architecture'],
        num_classes=len(config.model['classes']),
        pretrained=False
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Evaluate
    logger.info("Evaluating on Kaggle dataset...")
    evaluator = Evaluator(model, test_loader, device, config.model['classes'])
    metrics = evaluator.evaluate(save_path="models/kaggle_test_metrics")
    
    logger.info("=" * 60)
    logger.info("KAGGLE DATASET TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Metrics saved to: models/kaggle_test_metrics/")

if __name__ == "__main__":
    main()

