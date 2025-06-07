#!/usr/bin/env python3
# main_train.py

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing.data_loader import load_dataset_for_training
from src.data_preprocessing.feature_extractor import extract_features
from src.models.train_model import train_model

def get_data_path(language):
    augmented_path = os.path.join(project_root, 'data', 'augmented', language.lower())
    raw_path = os.path.join(project_root, 'data', 'raw', language.lower())
    
    if os.path.exists(augmented_path):
        return augmented_path
    elif os.path.exists(raw_path):
        return raw_path
    return None

def train_language(language):
    data_path = get_data_path(language)
    if not data_path:
        logger.error(f"No data found for {language}")
        return False
    
    logger.info(f"Loading {language} from {data_path}")
    landmarks_df = load_dataset_for_training(data_path, language)
    
    if landmarks_df.empty:
        logger.error(f"No landmarks for {language}")
        return False
    
    logger.info(f"Extracting features for {language}")
    features_df = extract_features(landmarks_df, max_features=80, perform_selection=True)
    
    if features_df.empty:
        logger.error(f"No features for {language}")
        return False
    
    # Save processed data
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    landmarks_file = os.path.join(processed_dir, f'{language.lower()}_landmarks.csv')
    features_file = os.path.join(processed_dir, f'{language.lower()}_features.csv')
    
    try:
        landmarks_df.to_csv(landmarks_file, index=False)
        features_df.to_csv(features_file, index=False)
        logger.info(f"Saved processed data to {processed_dir}")
    except Exception as e:
        logger.warning(f"Failed to save processed data: {e}")
    
    logger.info(f"Training model for {language}")
    X_test, y_test = train_model(features_df, 'sign_language_model', language)
    
    return X_test is not None

def main():
    languages = []
    
    for data_dir in ['augmented', 'raw']:
        path = os.path.join(project_root, 'data', data_dir)
        if os.path.exists(path):
            for item in os.listdir(path):
                if os.path.isdir(os.path.join(path, item)):
                    lang = item.upper()
                    if lang not in languages:
                        languages.append(lang)
    
    if not languages:
        logger.error("No training data found")
        return
    
    logger.info(f"Training: {languages}")
    
    for language in languages:
        logger.info(f"Processing {language}")
        success = train_language(language)
        logger.info(f"{language}: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()