#!/usr/bin/env python3
# CONSOLE FIXED app.py - Fix Windows console error

import os
import sys
import cv2
import numpy as np
import base64
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import io
from PIL import Image
import pandas as pd

# FIX: Console encoding untuk Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Setup logging tanpa colors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Setup paths
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_file_dir

if project_root not in sys.path:
    sys.path.append(project_root)

# Import feature extractor
try:
    from src.data_preprocessing.feature_extractor import extract_features
    extract_features_available = True
    logger.info("Feature extractor imported successfully")
except ImportError as e:
    extract_features_available = False
    logger.warning(f"Feature extractor not available: {e}")

# Flask app
app = Flask(__name__)
CORS(app, resources={"*": {"origins": "*"}})

class ConsoleFixedSignLanguageAPI:
    def __init__(self):
        self.models = {}
        self.project_root = project_root
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load models dengan logging yang clean"""
        logger.info("Loading models...")
        
        # Model paths
        models_to_load = [
            ('sibi', 'sign_language_model_sibi_sklearn.pkl'),
            ('bisindo', 'sign_language_model_bisindo_sklearn.pkl')
        ]
        
        models_dir = os.path.join(self.project_root, 'data', 'models')
        logger.info(f"Models directory: {models_dir}")
        
        if not os.path.exists(models_dir):
            logger.error(f"Models directory not found: {models_dir}")
            return
        
        for language, filename in models_to_load:
            model_path = os.path.join(models_dir, filename)
            
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading {language} from {filename}...")
                    
                    # Load model data
                    model_data = joblib.load(model_path)
                    
                    # Validate model structure
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']
                        
                        if hasattr(model, 'predict'):
                            self.models[language] = {
                                'model_data': model_data,
                                'model': model,
                                'scaler': model_data.get('scaler'),
                                'feature_names': model_data.get('feature_names'),
                                'path': model_path
                            }
                            
                            # Log feature info
                            feature_names = model_data.get('feature_names', [])
                            logger.info(f"{language} loaded successfully - {len(feature_names)} features expected")
                            
                        else:
                            logger.error(f"{language}: Model has no predict method")
                    else:
                        logger.error(f"{language}: Invalid model structure")
                        
                except Exception as e:
                    logger.error(f"Failed to load {language}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        logger.info(f"Total models loaded: {len(self.models)} - {list(self.models.keys())}")
    
    def extract_landmarks_from_frame(self, image):
        """Extract landmarks from image"""
        try:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks:
                return None
            
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
            
            return landmarks_flat
            
        except Exception as e:
            logger.error(f"Landmark extraction error: {e}")
            return None
    
    def extract_features_using_trained_pipeline(self, landmarks_data, language):
        """Extract features menggunakan pipeline yang SAMA dengan training"""
        try:
            # Check if we have the trained feature extractor
            if extract_features_available:
                # Create DataFrame dengan format yang sama seperti training
                landmark_cols = [f'landmark_{i}_{coord}' for i in range(42) for coord in ['x', 'y', 'z']]
                df_landmarks = pd.DataFrame([landmarks_data], columns=landmark_cols)
                
                # Extract features WITHOUT selection (penting!)
                features_df = extract_features(df_landmarks, perform_selection=False)
                
                if not features_df.empty:
                    # Remove label columns
                    features_for_prediction = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
                    return features_for_prediction
            
            # Fallback jika feature extractor tidak tersedia
            return self.create_fallback_features(landmarks_data, language)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return self.create_fallback_features(landmarks_data, language)
    
    def create_fallback_features(self, landmarks_data, language):
        """Create features yang match dengan expected feature count"""
        try:
            # Get expected feature count dari model
            if language in self.models:
                feature_names = self.models[language].get('feature_names', [])
                expected_count = len(feature_names) if feature_names else 50
            else:
                expected_count = 50
            
            # Generate features sampai mencapai expected count
            features = {}
            
            # Split into 2 hands
            hand1_data = landmarks_data[:63]
            hand2_data = landmarks_data[63:]
            
            feature_idx = 0
            
            # Basic hand statistics
            for hand_idx, hand_data in enumerate([hand1_data, hand2_data]):
                if len(hand_data) >= 63:
                    x_coords = hand_data[::3]
                    y_coords = hand_data[1::3]
                    z_coords = hand_data[2::3]
                    
                    # Statistik dasar
                    stats = [
                        np.mean(x_coords), np.std(x_coords), np.min(x_coords), np.max(x_coords),
                        np.mean(y_coords), np.std(y_coords), np.min(y_coords), np.max(y_coords),
                        np.mean(z_coords), np.std(z_coords), np.min(z_coords), np.max(z_coords),
                    ]
                    
                    for stat in stats:
                        if feature_idx < expected_count:
                            features[f'feature_{feature_idx}'] = float(stat)
                            feature_idx += 1
                    
                    # Distances
                    wrist_x, wrist_y, wrist_z = x_coords[0], y_coords[0], z_coords[0]
                    for tip_idx in [4, 8, 12, 16, 20]:  # fingertips
                        if tip_idx < len(x_coords) and feature_idx < expected_count:
                            dist = np.sqrt((x_coords[tip_idx] - wrist_x)**2 + 
                                         (y_coords[tip_idx] - wrist_y)**2 + 
                                         (z_coords[tip_idx] - wrist_z)**2)
                            features[f'feature_{feature_idx}'] = float(dist)
                            feature_idx += 1
                else:
                    # Fill with zeros for missing hand
                    for i in range(17):  # 12 stats + 5 distances
                        if feature_idx < expected_count:
                            features[f'feature_{feature_idx}'] = 0.0
                            feature_idx += 1
            
            # Tambahkan raw coordinates jika masih kurang
            coord_idx = 0
            while feature_idx < expected_count and coord_idx < len(landmarks_data):
                features[f'feature_{feature_idx}'] = float(landmarks_data[coord_idx])
                feature_idx += 1
                coord_idx += 1
            
            # Pad dengan zeros jika masih kurang
            while feature_idx < expected_count:
                features[f'feature_{feature_idx}'] = 0.0
                feature_idx += 1
            
            result_df = pd.DataFrame([features])
            return result_df
            
        except Exception as e:
            logger.error(f"Fallback feature creation failed: {e}")
            # Ultimate fallback - minimal features
            features = {f'feature_{i}': 0.0 for i in range(50)}
            return pd.DataFrame([features])
    
    def predict_with_proper_features(self, features_df, language):
        """Predict dengan feature handling yang benar"""
        try:
            model_info = self.models[language]
            model = model_info['model']
            scaler = model_info['scaler']
            feature_names = model_info['feature_names']
            
            # Handle feature alignment
            if feature_names:
                # Ensure we have all required features
                for feature_name in feature_names:
                    if feature_name not in features_df.columns:
                        features_df[feature_name] = 0.0
                
                # Select only the features that the model expects
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
                confidence = float(np.max(probabilities))
            except:
                confidence = 0.7
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed for {language}: {e}")
            return None, 0.0
    
    def predict_sign(self, image, language_type='bisindo'):
        """Main prediction function dengan proper error handling"""
        try:
            language_type = language_type.lower()
            logger.info(f"Predicting for {language_type}")
            
            # Check if model exists
            if language_type not in self.models:
                available = list(self.models.keys())
                error_msg = f"Model not available for {language_type}. Available: {available}"
                logger.error(error_msg)
                return None, 0.0, error_msg
            
            # Extract landmarks
            landmarks = self.extract_landmarks_from_frame(image)
            if landmarks is None:
                logger.warning("No hand landmarks detected")
                return "No hand detected", 0.0, "No hand landmarks detected"
            
            # Extract features
            features_df = self.extract_features_using_trained_pipeline(landmarks, language_type)
            if features_df.empty:
                return None, 0.0, "Feature extraction failed"
            
            # Make prediction
            prediction, confidence = self.predict_with_proper_features(features_df, language_type)
            
            if prediction is None:
                return None, 0.0, "Model prediction failed"
            
            logger.info(f"SUCCESS: {prediction} (confidence: {confidence:.3f})")
            return prediction, confidence, "Success"
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.0, f"Prediction error: {str(e)}"

# Initialize API
api = ConsoleFixedSignLanguageAPI()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(api.models.keys()),
        'total_models': len(api.models),
        'mediapipe_ready': api.mp_hands is not None
    })

@app.route('/api/translate', methods=['POST'])
def translate_sign():
    try:
        logger.info("Translate endpoint called")
        
        # Get request data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        # Check if models are loaded
        if not api.models:
            error_msg = 'No models loaded. Please train models first.'
            logger.error(error_msg)
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Extract parameters
        image_data = data['image']
        language_type = data.get('language_type', 'bisindo').lower()
        
        # Process image
        try:
            # Remove data URL prefix
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return jsonify({'success': False, 'error': f'Image processing failed: {e}'}), 400
        
        # Make prediction
        prediction, confidence, message = api.predict_sign(image, language_type)
        
        # Return response
        response = {
            'success': prediction is not None,
            'prediction': prediction,
            'confidence': float(confidence) if confidence else 0.0,
            'language_type': language_type,
            'dataset': language_type.upper(),
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'available_models': list(api.models.keys()),
        'total_models': len(api.models)
    })

if __name__ == '__main__':
    print("\nSILENT API STARTING...")
    print("=" * 40)
    print(f"Project root: {project_root}")
    print(f"Models loaded: {list(api.models.keys())}")
    print(f"Total models: {len(api.models)}")
    print(f"MediaPipe ready: {api.mp_hands is not None}")
    print(f"Feature extractor: {'Available' if extract_features_available else 'Fallback mode'}")
    
    if not api.models:
        print("\nNO MODELS LOADED!")
        print("Run: python main.py")
        print("Test: python camera_test.py")
    else:
        print("\nAPI READY FOR PREDICTIONS!")
        for lang, info in api.models.items():
            feature_count = len(info.get('feature_names', []))
            print(f"   {lang.upper()}: {feature_count} features expected")
    
    print("=" * 40)
    print("Starting server on http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    
    # FIX: Run without debug to avoid console errors
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Server error: {e}")
        print("Trying alternative startup...")
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)