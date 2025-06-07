#!/usr/bin/env python3
# main_aug.py

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing.augmentation import augment_dataset_for_language

def main():
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    
    if not os.path.exists(raw_data_dir):
        logger.error("No raw data directory found")
        return
    
    languages = [item.upper() for item in os.listdir(raw_data_dir) 
                 if os.path.isdir(os.path.join(raw_data_dir, item))]
    
    if not languages:
        logger.error("No languages found")
        return
    
    logger.info(f"Augmenting: {languages}")
    
    for language in languages:
        logger.info(f"Processing {language}")
        success = augment_dataset_for_language(language, project_root)
        logger.info(f"{language}: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()