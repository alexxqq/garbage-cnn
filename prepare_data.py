#!/usr/bin/env python3
"""
Data Preparation Script
Downloads and organizes datasets
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.dataset_loader import DatasetLoader

def main():
    config = Config("configs/config.yaml")
    logger = setup_logger(
        level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.file')
    )
    
    logger.info("=" * 60)
    logger.info("DATA PREPARATION")
    logger.info("=" * 60)
    
    loader = DatasetLoader(config)
    dataset_paths = loader.load_all_datasets()
    processed_dir = loader.organize_datasets(dataset_paths)
    
    logger.info(f"Data preparation complete! Processed data in: {processed_dir}")

if __name__ == "__main__":
    main()

