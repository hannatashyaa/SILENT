# camera_test.py - OPTIMIZED VERSION
# REPLACE the existing camera_test.py with this file

import cv2
import mediapipe as mp
import time
import os
import sys
import joblib
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project Configuration
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.data_preprocessing.feature_extractor import extract_features
except ImportError:
    extract_features = None
    logger.warning("Feature extractor not available - using fallback")

class OptimizedSignRecognizer:
    def __init__(self):
        self.project_root = project_root
        self.models = {}
        self.current_language = None
        self.current_model_data = None
        self.current_model_type = None
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Recognition settings
        self.translated_text = ""
        self.last_predicted_sign = ""
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 1.0
        self.min_hold_time = 0.8
        self.current_sign_hold_start_time = None
        self.current_sign_being_held = None
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0
        
        # Debug mode
        self.debug_mode = False
        self.frame_count = 0
        
        # Load models
        self.load_available_models()
        
    def load_available_models(self):
        """Load and validate available models"""
        model_configs = [
            {
                'name': 'SIBI',
                'tensorflow_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi_tensorflow.h5'),
                'tensorflow_meta_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi_tensorflow_meta.pkl'),
                'sklearn_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi_sklearn.pkl'),
                'combined_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_sibi.pkl'),
            },
            {
                'name': 'BISINDO',
                'tensorflow_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo_tensorflow.h5'),
                'tensorflow_meta_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo_tensorflow_meta.pkl'),
                'sklearn_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo_sklearn.pkl'),
                'combined_path': os.path.join(self.project_root, 'data', 'models', 'sign_language_model_bisindo.pkl'),
            }
        ]
        
        logger.info("Loading optimized models...")
        
        for config in model_configs:
            model_info = {
                'tensorflow_model': None,
                'tensorflow_meta': None,
                'sklearn_model': None,
                'available_models': []
            }
            
            # Load TensorFlow model
            if os.path.exists(config['tensorflow_path']) and os.path.exists(config['tensorflow_meta_path']):
                try:
                    tf_model = load_model(config['tensorflow_path'])
                    tf_meta = joblib.load(config['tensorflow_meta_path'])
                    
                    if self.validate_tensorflow_model(tf_model, tf_meta, config['name']):
                        model_info['tensorflow_model'] = tf_model
                        model_info['tensorflow_meta'] = tf_meta
                        model_info['available_models'].append('tensorflow')
                        logger.info(f"  {config['name']}: TensorFlow model loaded successfully")
                    else:
                        logger.warning(f"  {config['name']}: TensorFlow model validation failed")
                        
                except Exception as e:
                    logger.warning(f"  {config['name']}: TensorFlow load failed - {e}")
            
            # Load Scikit-learn model
            if os.path.exists(config['sklearn_path']):
                try:
                    sklearn_data = joblib.load(config['sklearn_path'])
                    
                    if self.validate_sklearn_model(sklearn_data, config['name']):
                        model_info['sklearn_model'] = sklearn_data
                        model_info['available_models'].append('sklearn')
                        logger.info(f"  {config['name']}: Scikit-learn model loaded successfully")
                    else:
                        logger.warning(f"  {config['name']}: Scikit-learn model validation failed")
                        
                except Exception as e:
                    logger.warning(f"  {config['name']}: Scikit-learn load failed - {e}")
            
            if model_info['available_models']:
                self.models[config['name']] = model_info
                logger.info(f"  {config['name']}: Available models - {', '.join(model_info['available_models'])}")
        
        if not self.models:
            logger.error("No valid models found! Please run training first.")
            return False
        
        logger.info(f"Successfully loaded {len(self.models)} language models")
        return True
    
    def validate_tensorflow_model(self, model, meta, language):
        """Validate TensorFlow model"""
        try:
            input_shape = model.input_shape[1]
            test_data = np.random.rand(1, input_shape) * 0.1
            
            pred_prob = model.predict(test_data, verbose=0)
            pred_class = np.argmax(pred_prob, axis=1)[0]
            
            if 'label_encoder' in meta:
                label_encoder = meta['label_encoder']
                pred_label = label_encoder.inverse_transform([pred_class])[0]
            
            logger.debug(f"    {language} TensorFlow model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"    {language} TensorFlow validation failed: {e}")
            return False
    
    def validate_sklearn_model(self, model_data, language):
        """Validate Scikit-learn model"""
        try:
            model = model_data['model']
            
            if hasattr(model, 'n_features_in_'):
                input_shape = model.n_features_in_
            else:
                input_shape = 50  # Default fallback
            
            test_data = np.random.rand(1, input_shape) * 0.1
            pred = model.predict(test_data)[0]
            
            logger.debug(f"    {language} Scikit-learn model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"    {language} Scikit-learn validation failed: {e}")
            return False
    
    def switch_language(self, language):
        """Switch to specified language model"""
        language = language.upper()
        
        if language not in self.models:
            logger.warning(f"{language} model not available")
            return False
        
        self.current_language = language
        model_info = self.models[language]
        
        # Prefer TensorFlow first, then sklearn
        if 'tensorflow' in model_info['available_models']:
            self.current_model_type = 'tensorflow'
            self.current_model_data = {
                'model': model_info['tensorflow_model'],
                'meta': model_info['tensorflow_meta'],
                'type': 'tensorflow'
            }
            logger.info(f"Switched to {language} (TensorFlow model)")
        elif 'sklearn' in model_info['available_models']:
            self.current_model_type = 'sklearn'
            self.current_model_data = {
                'model': model_info['sklearn_model']['model'],
                'scaler': model_info['sklearn_model']['scaler'],
                'feature_names': model_info['sklearn_model']['feature_names'],
                'type': 'sklearn'
            }
            logger.info(f"Switched to {language} (Scikit-learn model)")
        else:
            logger.error(f"No valid models for {language}")
            return False
        
        self.clear_translation_state()
        return True
    
    def switch_model_type(self, model_type):
        """Switch between model types for current language"""
        if not self.current_language:
            logger.warning("Select a language first")
            return False
        
        model_info = self.models[self.current_language]
        
        if model_type not in model_info['available_models']:
            logger.warning(f"{model_type.upper()} model not available for {self.current_language}")
            return False
        
        self.current_model_type = model_type
        
        if model_type == 'tensorflow':
            self.current_model_data = {
                'model': model_info['tensorflow_model'],
                'meta': model_info['tensorflow_meta'],
                'type': 'tensorflow'
            }
        else:
            self.current_model_data = {
                'model': model_info['sklearn_model']['model'],
                'scaler': model_info['sklearn_model']['scaler'],
                'feature_names': model_info['sklearn_model']['feature_names'],
                'type': 'sklearn'
            }
        
        logger.info(f"Switched to {model_type.upper()} model for {self.current_language}")
        return True
    
    def clear_translation_state(self):
        """Clear translation state"""
        self.translated_text = ""
        self.last_predicted_sign = ""
        self.current_sign_being_held = None
        self.current_sign_hold_start_time = None
    
    def extract_landmarks_from_frame(self, frame):
        """Extract hand landmarks from camera frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Process landmarks for up to 2 hands
        landmarks_flat = []
        
        # Sort hands by position (left to right)
        hands_data = []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist_x = hand_landmarks.landmark[0].x
            hands_data.append((wrist_x, hand_landmarks))
        
        hands_data.sort(key=lambda x: x[0])
        
        # Extract landmarks for up to 2 hands
        for hand_idx in range(2):
            if hand_idx < len(hands_data):
                hand_landmarks = hands_data[hand_idx][1]
                
                for landmark in hand_landmarks.landmark:
                    landmarks_flat.extend([
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.z)
                    ])
            else:
                # Pad with zeros for missing hand
                landmarks_flat.extend([0.0] * 63)
        
        # Ensure exactly 126 values
        if len(landmarks_flat) != 126:
            landmarks_flat = landmarks_flat[:126]
            while len(landmarks_flat) < 126:
                landmarks_flat.append(0.0)
        
        return landmarks_flat, results.multi_hand_landmarks
    
    def extract_features_from_landmarks(self, landmarks_data):
        """Extract features from landmarks using same pipeline as training"""
        try:
            if extract_features is not None:
                # Use same pipeline as training
                landmark_cols = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']]
                df_landmarks = pd.DataFrame([landmarks_data], columns=landmark_cols)
                
                # Extract features without selection (use all features)
                features_df = extract_features(df_landmarks, perform_selection=False)
                
                if not features_df.empty:
                    # Remove label columns
                    features_for_prediction = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
                    return features_for_prediction
            
            # Fallback: create basic features
            return self.create_basic_features(landmarks_data)
            
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Feature extraction error: {e}")
            return self.create_basic_features(landmarks_data)
    
    def create_basic_features(self, landmarks_data):
        """Create basic features as fallback"""
        try:
            features = {}
            
            # Split into 2 hands
            hand1_data = landmarks_data[:63]
            hand2_data = landmarks_data[63:]
            
            for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
                if len(hand_data) >= 63:
                    x_coords = hand_data[::3]
                    y_coords = hand_data[1::3]
                    z_coords = hand_data[2::3]
                    
                    # Basic statistics
                    features[f'h{hand_idx}_x_mean'] = np.mean(x_coords)
                    features[f'h{hand_idx}_y_mean'] = np.mean(y_coords)
                    features[f'h{hand_idx}_z_mean'] = np.mean(z_coords)
                    features[f'h{hand_idx}_x_std'] = np.std(x_coords)
                    features[f'h{hand_idx}_y_std'] = np.std(y_coords)
                    features[f'h{hand_idx}_z_std'] = np.std(z_coords)
                    
                    # Distances
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    for tip_idx in [4, 8, 12, 16, 20]:  # fingertips
                        if tip_idx < len(x_coords):
                            dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                         (y_coords[tip_idx] - wrist_y)**2 + 
                                         (z_coords[tip_idx] - wrist_z)**2)
                            features[f'h{hand_idx}_tip_{tip_idx}_dist'] = dist
                else:
                    # Fill with zeros for missing hand
                    for stat in ['x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std']:
                        features[f'h{hand_idx}_{stat}'] = 0.0
                    for tip_idx in [4, 8, 12, 16, 20]:
                        features[f'h{hand_idx}_tip_{tip_idx}_dist'] = 0.0
            
            return pd.DataFrame([features])
            
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Basic feature creation error: {e}")
            # Ultimate fallback
            basic_features = {f'feature_{i}': 0.0 for i in range(50)}
            return pd.DataFrame([basic_features])
    
    def predict_sign(self, landmarks_data):
        """Make prediction using current model"""
        if not self.current_language or self.current_model_data is None:
            return "Select Language (1=SIBI, 2=BISINDO)"
        
        self.frame_count += 1
        
        try:
            # Extract features
            features_df = self.extract_features_from_landmarks(landmarks_data)
            
            if features_df.empty:
                return "Feature Extraction Failed"
            
            if self.debug_mode and self.frame_count % 30 == 0:
                logger.info(f"Frame {self.frame_count}: Features shape: {features_df.shape}")
            
            # Make prediction based on model type
            if self.current_model_type == 'tensorflow':
                prediction, confidence = self.predict_tensorflow(features_df)
            else:
                prediction, confidence = self.predict_sklearn(features_df)
            
            if prediction is None:
                return "Prediction Failed"
            
            # Format output with confidence levels
            if confidence >= 0.8:
                return f"HIGH {prediction}"
            elif confidence >= 0.6:
                return f"MED {prediction}"
            elif confidence >= 0.4:
                return f"LOW {prediction}"
            else:
                return f"UNCERTAIN {prediction}"
                
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Prediction error: {e}")
            return f"Error: {str(e)[:20]}"
    
    def predict_tensorflow(self, features_df):
        """Make prediction using TensorFlow model"""
        try:
            model = self.current_model_data['model']
            meta = self.current_model_data['meta']
            
            # Handle feature compatibility
            scaler = meta.get('scaler')
            label_encoder = meta.get('label_encoder')
            feature_names = meta.get('feature_names')
            
            # Match features to training
            if feature_names:
                missing_features = set(feature_names) - set(features_df.columns)
                for feature in missing_features:
                    features_df[feature] = 0.0
                features_df = features_df[feature_names]
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            predictions_prob = model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(predictions_prob, axis=1)[0]
            confidence = np.max(predictions_prob)
            
            # Convert to label
            if label_encoder:
                prediction = label_encoder.inverse_transform([predicted_class_idx])[0]
            else:
                prediction = f"Class_{predicted_class_idx}"
            
            return prediction, confidence
            
        except Exception as e:
            if self.debug_mode:
                logger.error(f"TensorFlow prediction error: {e}")
            return None, 0.0
    
    def predict_sklearn(self, features_df):
        """Make prediction using Scikit-learn model"""
        try:
            model = self.current_model_data['model']
            scaler = self.current_model_data.get('scaler')
            feature_names = self.current_model_data.get('feature_names')
            
            # Handle feature compatibility
            if feature_names:
                missing_features = set(feature_names) - set(features_df.columns)
                for feature in missing_features:
                    features_df[feature] = 0.0
                features_df = features_df[feature_names]
            
            # Apply scaling
            if scaler:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Get confidence
            try:
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            except:
                confidence = 0.7
            
            return prediction, confidence
            
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Sklearn prediction error: {e}")
            return None, 0.0
    
    def update_translation(self, predicted_label):
        """Update translation with validation"""
        if any(indicator in predicted_label for indicator in ["HIGH", "MED"]):
            parts = predicted_label.split()
            if len(parts) >= 2:
                clean_prediction = parts[1]
                
                current_time = time.time()
                
                if self.current_sign_being_held == clean_prediction:
                    if (current_time - self.current_sign_hold_start_time >= self.min_hold_time and
                        (current_time - self.last_prediction_time >= self.prediction_cooldown or
                         self.last_predicted_sign != clean_prediction)):
                        
                        self.translated_text += clean_prediction
                        self.last_predicted_sign = clean_prediction
                        self.last_prediction_time = current_time
                        logger.info(f"Added '{clean_prediction}' using {self.current_language} ({self.current_model_type.upper()})")
                else:
                    self.current_sign_being_held = clean_prediction
                    self.current_sign_hold_start_time = current_time
    
    def calculate_fps(self):
        """Calculate FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.avg_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        return self.avg_fps
    
    def draw_interface(self, frame, predicted_label):
        """Draw interface"""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps = self.calculate_fps()
        
        # Language and model info
        if self.current_language:
            lang_text = f"Language: {self.current_language} ({self.current_model_type.upper()})"
        else:
            lang_text = "Language: Press 1 or 2"
        cv2.putText(frame, lang_text, (20, 40), font, 0.7, (255, 0, 255), 2)
        
        # Prediction
        cv2.putText(frame, f"Sign: {predicted_label}", (20, 80), font, 1.0, (0, 255, 0), 2)
        
        # Translation
        trans_text = self.translated_text if self.translated_text else "[empty]"
        cv2.putText(frame, f"Text: {trans_text}", (20, 120), font, 0.8, (255, 255, 0), 2)
        
        # Model availability
        if self.current_language and self.current_language in self.models:
            model_info = self.models[self.current_language]
            available_models = ', '.join(model_info['available_models'])
            cv2.putText(frame, f"Available: {available_models}", (20, 160), font, 0.5, (255, 255, 255), 1)
        
        # Performance
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {self.frame_count}", (20, 190), font, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"OPTIMIZED - Dual Model System", (20, 210), font, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "1=SIBI, 2=BISINDO, T=TensorFlow, K=Sklearn", (20, 240), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "D=Debug, S=Space, C=Clear, Q=Quit", (20, 260), font, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main camera loop"""
        if not self.models:
            logger.error("No valid models available. Please run training first.")
            return
            
        print("\nSILENT - OPTIMIZED Sign Language Recognition")
        print("=" * 60)
        print("OPTIMIZED FEATURES:")
        print("- Dual model support (TensorFlow + Scikit-learn)")
        print("- Advanced feature extraction")
        print("- Model validation and selection")
        print("- Adaptive architecture based on dataset size")
        print("- Robust 2-hand detection for BISINDO")
        print("=" * 60)
        print("Controls:")
        print("  1 - Switch to SIBI")
        print("  2 - Switch to BISINDO")
        print("  T - Switch to TensorFlow model")
        print("  K - Switch to Scikit-learn model")
        print("  D - Toggle debug mode")
        print("  S - Add space")
        print("  C - Clear text")
        print("  Q - Quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open camera")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                predicted_label = "No Hand Detected"
                
                try:
                    landmark_result = self.extract_landmarks_from_frame(frame)
                    
                    if landmark_result[0] is not None:
                        landmarks_data, multi_hand_landmarks = landmark_result
                        
                        # Draw hand landmarks
                        for hand_landmarks in multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                            )
                        
                        if self.current_language:
                            predicted_label = self.predict_sign(landmarks_data)
                            self.update_translation(predicted_label)
                        else:
                            predicted_label = "Select Language: 1=SIBI, 2=BISINDO"
                        
                except Exception as e:
                    predicted_label = f"Error: {str(e)[:15]}"
                    if self.debug_mode:
                        logger.error(f"Frame error: {e}")
                
                self.draw_interface(frame, predicted_label)
                cv2.imshow('SILENT - OPTIMIZED Recognition', frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    if self.switch_language('SIBI'):
                        print("Switched to SIBI")
                elif key == ord('2'):
                    if self.switch_language('BISINDO'):
                        print("Switched to BISINDO")
                elif key == ord('t'):
                    if self.switch_model_type('tensorflow'):
                        print("Switched to TensorFlow model")
                elif key == ord('k'):
                    if self.switch_model_type('sklearn'):
                        print("Switched to Scikit-learn model")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                elif key == ord('s'):
                    if self.translated_text and not self.translated_text.endswith(' '):
                        self.translated_text += " "
                        print(f"Added space. Text: '{self.translated_text}'")
                elif key == ord('c'):
                    self.clear_translation_state()
                    print("Text cleared")
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
            print(f"\nFinal text: '{self.translated_text}'")

def main():
    try:
        recognizer = OptimizedSignRecognizer()
        recognizer.run()
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()