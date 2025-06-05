#!/usr/bin/env python3
# main.py - OPTIMIZED TRAINING PIPELINE WITH AUGMENTATION
# REPLACE the existing main.py with this file

import os
import sys
import pandas as pd
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Project configuration
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules
try:
    from src.data_preprocessing.data_loader import load_and_process_dataset
    from src.data_preprocessing.feature_extractor import extract_features, analyze_feature_importance
    from src.data_preprocessing.augmentation import augment_dataset_for_language  # Added augmentation import
    from src.models.train_model import train_model
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all source files are in correct directories")
    sys.exit(1)

class OptimizedTrainingPipeline:
    def __init__(self):
        self.project_root = project_root
        self.data_stats = {}
        self.use_augmentation = True  # Flag to enable/disable augmentation
        
    def check_data_availability(self):
        """Check if training data is available"""
        logger.info("Checking data availability...")
        
        datasets = {
            'BISINDO': os.path.join(self.project_root, 'data', 'raw', 'bisindo'),
            'SIBI': os.path.join(self.project_root, 'data', 'raw', 'sibi')
        }
        
        available_datasets = {}
        
        for lang_type, data_path in datasets.items():
            if os.path.exists(data_path):
                subfolders = [f for f in os.listdir(data_path) 
                             if os.path.isdir(os.path.join(data_path, f))]
                
                if subfolders:
                    # Count images
                    total_images = 0
                    class_distribution = {}
                    
                    for subfolder in subfolders:
                        subfolder_path = os.path.join(data_path, subfolder)
                        images = [f for f in os.listdir(subfolder_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        total_images += len(images)
                        class_distribution[subfolder] = len(images)
                    
                    if total_images > 0:
                        available_datasets[lang_type] = {
                            'path': data_path,
                            'classes': len(subfolders),
                            'total_images': total_images,
                            'class_distribution': class_distribution,
                            'min_images_per_class': min(class_distribution.values()),
                            'max_images_per_class': max(class_distribution.values())
                        }
                        
                        logger.info(f"Found {lang_type}: {len(subfolders)} classes, {total_images} images")
                        logger.info(f"  Images per class: {min(class_distribution.values())} - {max(class_distribution.values())}")
                    else:
                        logger.warning(f"No images found in {lang_type}")
                else:
                    logger.warning(f"No class folders found in {lang_type}")
            else:
                logger.warning(f"Directory not found: {data_path}")
        
        if not available_datasets:
            logger.error("No training data found!")
            logger.error("Please add training images to:")
            for lang_type, data_path in datasets.items():
                logger.error(f"  {data_path}")
            return False, {}
        
        self.data_stats = available_datasets
        return True, available_datasets
    
    def perform_augmentation(self, language_type):
        """Perform data augmentation for a language"""
        logger.info(f"Step 0: Performing data augmentation for {language_type}")
        
        try:
            # Check if we need augmentation
            data_info = self.data_stats[language_type]
            min_images = data_info['min_images_per_class']
            
            # Skip augmentation if we have enough data
            if min_images >= 15 and not self.use_augmentation:
                logger.info(f"Sufficient data for {language_type}, skipping augmentation")
                return data_info['path']  # Return original path
            
            # Perform augmentation
            success = augment_dataset_for_language(language_type, self.project_root)
            
            if success:
                # Update path to use augmented data
                augmented_path = os.path.join(self.project_root, 'data', 'augmented', language_type.lower())
                
                # Verify augmented data
                if os.path.exists(augmented_path):
                    subfolders = [f for f in os.listdir(augmented_path) 
                                 if os.path.isdir(os.path.join(augmented_path, f))]
                    
                    if subfolders:
                        # Count augmented images
                        total_augmented = 0
                        for subfolder in subfolders:
                            subfolder_path = os.path.join(augmented_path, subfolder)
                            images = [f for f in os.listdir(subfolder_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            total_augmented += len(images)
                        
                        logger.info(f"Augmentation successful: {total_augmented} total images after augmentation")
                        return augmented_path
                    else:
                        logger.warning(f"No class folders in augmented data, using original")
                        return data_info['path']
                else:
                    logger.warning(f"Augmented folder not created, using original data")
                    return data_info['path']
            else:
                logger.warning(f"Augmentation failed for {language_type}, using original data")
                return data_info['path']
                
        except Exception as e:
            logger.error(f"Augmentation error for {language_type}: {e}")
            logger.warning(f"Using original data for {language_type}")
            return self.data_stats[language_type]['path']
    
    def process_single_dataset(self, language_type, data_info):
        """Process a single dataset with optimized pipeline including augmentation"""
        logger.info(f"Processing {language_type} dataset")
        logger.info(f"Classes: {data_info['classes']}, Images: {data_info['total_images']}")
        
        start_time = time.time()
        
        # Step 0: Data Augmentation (NEW)
        if self.use_augmentation:
            try:
                augmented_data_path = self.perform_augmentation(language_type)
                logger.info(f"Using data from: {augmented_data_path}")
            except Exception as e:
                logger.error(f"Augmentation failed: {e}")
                augmented_data_path = data_info['path']  # Fallback to original
        else:
            augmented_data_path = data_info['path']
        
        # Step 1: Load landmarks
        logger.info("Step 1: Extracting hand landmarks from images...")
        try:
            landmarks_df = load_and_process_dataset(augmented_data_path, language_type)
            
            if landmarks_df.empty:
                logger.error(f"Failed to load landmarks for {language_type}")
                return False
            
            logger.info(f"Extracted landmarks from {len(landmarks_df)} images")
            logger.info(f"Classes found: {landmarks_df['label'].nunique()}")
            
            # Log class distribution
            class_counts = landmarks_df['label'].value_counts()
            logger.info(f"Sample distribution: {dict(class_counts)}")
            
        except Exception as e:
            logger.error(f"Landmark extraction failed for {language_type}: {e}")
            return False
        
        # Step 2: Feature extraction
        logger.info("Step 2: Extracting optimized features...")
        try:
            # Adaptive feature count based on dataset size
            max_features = min(80, len(landmarks_df) // 3)
            max_features = max(20, max_features)  # Minimum 20 features
            
            features_df = extract_features(
                landmarks_df, 
                max_features=max_features, 
                perform_selection=True
            )
            
            if features_df.empty:
                logger.error(f"Feature extraction failed for {language_type}")
                return False
            
            # Analyze features
            feature_importance = analyze_feature_importance(features_df)
            
            feature_count = len([col for col in features_df.columns 
                               if col not in ['label', 'sign_language_type']])
            
            logger.info(f"Extracted {feature_count} optimized features")
            logger.info(f"Max features limit: {max_features}")
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {language_type}: {e}")
            return False
        
        # Step 3: Model training
        logger.info("Step 3: Training optimized models...")
        try:
            X_test, y_test = train_model(
                features_df, 
                'sign_language_model', 
                language_type
            )
            
            if X_test is None:
                logger.error(f"Model training failed for {language_type}")
                return False
            
            logger.info(f"Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed for {language_type}: {e}")
            return False
        
        # Step 4: Save processed data
        try:
            processed_dir = os.path.join(self.project_root, 'data', 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            
            # Save landmarks
            landmarks_file = os.path.join(processed_dir, f'{language_type.lower()}_landmarks.csv')
            landmarks_df.to_csv(landmarks_file, index=False)
            
            # Save features
            features_file = os.path.join(processed_dir, f'{language_type.lower()}_features.csv')
            features_df.to_csv(features_file, index=False)
            
            logger.info(f"Processed data saved to {processed_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save processed data: {e}")
        
        processing_time = time.time() - start_time
        logger.info(f"{language_type} processing completed in {processing_time:.2f} seconds")
        
        return True
    
    def validate_trained_models(self):
        """Validate that models were created correctly"""
        logger.info("Validating trained models...")
        
        model_dir = os.path.join(self.project_root, 'data', 'models')
        
        if not os.path.exists(model_dir):
            logger.error("Models directory not found")
            return False
        
        expected_models = []
        for lang_type in self.data_stats.keys():
            lang_lower = lang_type.lower()
            expected_models.extend([
                f'sign_language_model_{lang_lower}_tensorflow.h5',
                f'sign_language_model_{lang_lower}_tensorflow_meta.pkl',
                f'sign_language_model_{lang_lower}_sklearn.pkl',
                f'sign_language_model_{lang_lower}.pkl'
            ])
        
        created_models = []
        missing_models = []
        
        for model_file in expected_models:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                created_models.append(f"{model_file} ({file_size/1024:.1f} KB)")
            else:
                missing_models.append(model_file)
        
        logger.info(f"Model validation results:")
        logger.info(f"  Created: {len(created_models)} models")
        for model in created_models:
            logger.info(f"    {model}")
        
        if missing_models:
            logger.warning(f"  Missing: {len(missing_models)} models")
            for model in missing_models:
                logger.warning(f"    {model}")
        
        return len(created_models) > len(missing_models)
    
    def check_augmented_folders(self):
        """Check and report status of augmented folders"""
        logger.info("Checking augmented data status...")
        
        augmented_base = os.path.join(self.project_root, 'data', 'augmented')
        
        if not os.path.exists(augmented_base):
            logger.info("Augmented folder does not exist yet")
            return
        
        for lang_type in self.data_stats.keys():
            augmented_path = os.path.join(augmented_base, lang_type.lower())
            
            if os.path.exists(augmented_path):
                subfolders = [f for f in os.listdir(augmented_path) 
                             if os.path.isdir(os.path.join(augmented_path, f))]
                
                if subfolders:
                    total_images = 0
                    for subfolder in subfolders:
                        subfolder_path = os.path.join(augmented_path, subfolder)
                        images = [f for f in os.listdir(subfolder_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        total_images += len(images)
                    
                    logger.info(f"  {lang_type} augmented: {len(subfolders)} classes, {total_images} images")
                else:
                    logger.info(f"  {lang_type} augmented: folder exists but empty")
            else:
                logger.info(f"  {lang_type} augmented: not created")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        logger.info("TRAINING PIPELINE REPORT")
        logger.info("=" * 50)
        
        # Dataset summary
        logger.info("Dataset Summary:")
        total_images = 0
        total_classes = 0
        
        for lang_type, info in self.data_stats.items():
            total_images += info['total_images']
            total_classes += info['classes']
            
            logger.info(f"  {lang_type}:")
            logger.info(f"    Classes: {info['classes']}")
            logger.info(f"    Total images: {info['total_images']}")
            logger.info(f"    Images per class: {info['min_images_per_class']} - {info['max_images_per_class']}")
        
        logger.info(f"Overall Statistics:")
        logger.info(f"  Languages processed: {len(self.data_stats)}")
        logger.info(f"  Total classes: {total_classes}")
        logger.info(f"  Total images: {total_images}")
        logger.info(f"  Augmentation used: {'Yes' if self.use_augmentation else 'No'}")
        
        # Check augmented data status
        self.check_augmented_folders()
        
        # Model summary
        model_dir = os.path.join(self.project_root, 'data', 'models')
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pkl', '.h5'))]
            total_model_size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in model_files)
            
            logger.info(f"Model Summary:")
            logger.info(f"  Model files created: {len(model_files)}")
            logger.info(f"  Total model size: {total_model_size/1024/1024:.2f} MB")
        
        # Recommendations
        logger.info(f"Recommendations:")
        
        for lang_type, info in self.data_stats.items():
            if info['min_images_per_class'] < 10:
                logger.warning(f"  {lang_type}: Consider adding more images per class (minimum 10-15 recommended)")
            
            if info['max_images_per_class'] / info['min_images_per_class'] > 3:
                logger.warning(f"  {lang_type}: Class imbalance detected - consider balancing dataset")
        
        logger.info(f"Next Steps:")
        logger.info(f"  1. Test models with: python camera_test.py")
        logger.info(f"  2. Or run: python run_silent.py test")
        logger.info(f"  3. Check model performance in real-time")
        
        logger.info("=" * 50)
    
    def run_optimized_pipeline(self, enable_augmentation=True):
        """Run the complete optimized training pipeline"""
        logger.info("Starting OPTIMIZED Sign Language Training Pipeline")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Augmentation enabled: {enable_augmentation}")
        
        self.use_augmentation = enable_augmentation
        total_start_time = time.time()
        
        # Check data availability
        data_available, datasets = self.check_data_availability()
        if not data_available:
            return False
        
        # Process each dataset
        successful_datasets = []
        failed_datasets = []
        
        for language_type, data_info in datasets.items():
            logger.info(f"Processing {language_type} dataset...")
            
            success = self.process_single_dataset(language_type, data_info)
            
            if success:
                successful_datasets.append(language_type)
            else:
                failed_datasets.append(language_type)
        
        # Validate models
        models_valid = self.validate_trained_models()
        
        # Generate report
        self.generate_training_report()
        
        # Final summary
        total_time = time.time() - total_start_time
        
        logger.info(f"PIPELINE SUMMARY:")
        logger.info(f"  Total processing time: {total_time:.2f} seconds")
        logger.info(f"  Successful datasets: {len(successful_datasets)} {successful_datasets}")
        logger.info(f"  Failed datasets: {len(failed_datasets)} {failed_datasets}")
        logger.info(f"  Models validation: {'PASSED' if models_valid else 'FAILED'}")
        logger.info(f"  Augmentation used: {'Yes' if self.use_augmentation else 'No'}")
        
        if successful_datasets and models_valid:
            logger.info(f"  Status: TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"  Ready for testing: python camera_test.py")
            return True
        else:
            logger.error(f"  Status: TRAINING FAILED")
            return False

def main():
    """Main execution function"""
    try:
        # Parse command line arguments for augmentation control
        enable_augmentation = True
        if len(sys.argv) > 1:
            if sys.argv[1].lower() in ['no-aug', 'no-augmentation', '--no-aug']:
                enable_augmentation = False
                print("Augmentation disabled by command line argument")
        
        # Initialize and run pipeline
        pipeline = OptimizedTrainingPipeline()
        success = pipeline.run_optimized_pipeline(enable_augmentation=enable_augmentation)
        
        if success:
            print("\nTraining completed successfully!")
            print("Next: Run 'python camera_test.py' to test the models")
            print("Augmented data should now be in data/augmented/ folders")
        else:
            print("\nTraining failed. Check the logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()