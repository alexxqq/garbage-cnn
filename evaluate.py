#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained model and generates comprehensive metrics
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Change to project root directory

import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.dataset import GarbageDataset, get_transforms
from src.models.model_factory import create_model
from src.models.evaluator import Evaluator

def main():
    config = Config("configs/config.yaml")
    logger = setup_logger(
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file')
    )
    
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    transform = get_transforms(
        image_size=config.training['image_size'],
        is_training=False
    )
    
    test_dataset = GarbageDataset(
        str(Path(config.data['processed_dir']) / "test"),
        transform=transform,
        classes=config.model['classes']
    )
    
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
        logger.error(f"Model not found: {model_path}")
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
    evaluator = Evaluator(model, test_loader, device, config.model['classes'])
    metrics = evaluator.evaluate(save_path="models/metrics")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()

