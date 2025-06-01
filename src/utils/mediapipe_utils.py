# src/utils/mediapipe_utils.py

import cv2
import mediapipe as mp
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_hand_landmarks(image_path):
    """
    Ekstrak landmark tangan dari gambar menggunakan MediaPipe.
    
    Args:
        image_path (str): Path ke file gambar
        
    Returns:
        numpy.ndarray atau None: Array landmark (126 elemen: 42 landmark x 3 koordinat)
                                atau None jika gagal
    """
    try:
        # Initialize MediaPipe hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,  # True untuk gambar statis
            max_num_hands=2,  # Maksimal 2 tangan (untuk BISINDO)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Baca gambar
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None
        
        # Convert BGR to RGB (MediaPipe menggunakan RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process gambar
        results = hands.process(rgb_image)
        
        # Initialize landmark array (42 landmarks x 3 coordinates = 126 values)
        # 21 landmarks per hand, maksimal 2 tangan
        landmarks_array = np.zeros(126)  # 42 landmarks * 3 koordinat (x, y, z)
        
        if results.multi_hand_landmarks:
            # Process setiap tangan yang terdeteksi
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:  # Maksimal 2 tangan
                    break
                    
                # Extract landmark coordinates untuk tangan ini
                start_idx = hand_idx * 63  # 21 landmarks * 3 koordinat per tangan
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    if i >= 21:  # Maksimal 21 landmarks per tangan
                        break
                    
                    # Setiap landmark memiliki koordinat x, y, z
                    idx_base = start_idx + (i * 3)
                    landmarks_array[idx_base] = landmark.x      # x coordinate
                    landmarks_array[idx_base + 1] = landmark.y  # y coordinate  
                    landmarks_array[idx_base + 2] = landmark.z  # z coordinate
            
            logger.debug(f"Successfully extracted landmarks from {image_path}")
            hands.close()
            return landmarks_array
        else:
            logger.debug(f"No hands detected in {image_path}")
            hands.close()
            return None
            
    except Exception as e:
        logger.error(f"Error extracting landmarks from {image_path}: {e}")
        return None

def visualize_landmarks(image_path, output_path=None):
    """
    Visualisasi landmark pada gambar untuk debugging.
    
    Args:
        image_path (str): Path ke file gambar input
        output_path (str): Path untuk menyimpan gambar hasil (opsional)
        
    Returns:
        numpy.ndarray atau None: Gambar dengan landmark yang digambar
    """
    try:
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        # Convert untuk processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        # Gambar landmark
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        # Simpan jika diminta
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"Visualization saved to {output_path}")
        
        hands.close()
        return image
        
    except Exception as e:
        logger.error(f"Error visualizing landmarks: {e}")
        return None

def validate_landmarks(landmarks_array):
    """
    Validasi array landmarks.
    
    Args:
        landmarks_array (numpy.ndarray): Array landmarks
        
    Returns:
        bool: True jika valid, False jika tidak
    """
    if landmarks_array is None:
        return False
    
    if not isinstance(landmarks_array, np.ndarray):
        return False
    
    if landmarks_array.shape[0] != 126:  # 42 landmarks * 3 koordinat
        logger.warning(f"Invalid landmarks shape: {landmarks_array.shape}, expected (126,)")
        return False
    
    # Check for reasonable coordinate ranges
    # X dan Y harus dalam range [0, 1], Z bisa negatif
    x_coords = landmarks_array[0::3]  # Setiap koordinat ke-3 dimulai dari 0
    y_coords = landmarks_array[1::3]  # Setiap koordinat ke-3 dimulai dari 1
    
    if np.any(x_coords < 0) or np.any(x_coords > 1):
        logger.warning("X coordinates out of expected range [0, 1]")
        return False
    
    if np.any(y_coords < 0) or np.any(y_coords > 1):
        logger.warning("Y coordinates out of expected range [0, 1]")
        return False
    
    return True

def batch_extract_landmarks(image_folder, output_csv=None):
    """
    Ekstrak landmarks dari seluruh folder gambar.
    
    Args:
        image_folder (str): Path ke folder berisi gambar
        output_csv (str): Path file CSV output (opsional)
        
    Returns:
        list: List of tuples (image_name, landmarks_array)
    """
    results = []
    
    if not os.path.exists(image_folder):
        logger.error(f"Folder not found: {image_folder}")
        return results
    
    # Get semua file gambar
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(image_extensions)]
    
    logger.info(f"Processing {len(image_files)} images from {image_folder}")
    
    successful = 0
    failed = 0
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        landmarks = extract_hand_landmarks(image_path)
        
        if landmarks is not None and validate_landmarks(landmarks):
            results.append((image_file, landmarks))
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
    
    # Save ke CSV jika diminta
    if output_csv and results:
        try:
            import pandas as pd
            
            # Prepare data for DataFrame
            data_rows = []
            for image_name, landmarks in results:
                row = {'image_name': image_name}
                for i in range(126):
                    row[f'landmark_{i}'] = landmarks[i]
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
    
    return results

class MediaPipeHandler:
    """
    Handler class untuk MediaPipe (kompatibilitas dengan kode lama)
    """
    def __init__(self):
        """Initialize MediaPipe hands solution"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support 2 hands untuk BISINDO
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("MediaPipe hands initialized")
    
    def process_frame(self, image):
        """Process frame and extract hand landmarks"""
        try:
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return None
    
    def extract_landmarks(self, results):
        """Extract normalized landmarks from MediaPipe results"""
        if not results.multi_hand_landmarks:
            return None
        
        # Initialize landmark array untuk 2 tangan
        landmarks_array = np.zeros(126)  # 42 landmarks * 3 koordinat
        
        # Process setiap tangan
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:  # Maksimal 2 tangan
                break
                
            start_idx = hand_idx * 63  # 21 landmarks * 3 koordinat per tangan
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i >= 21:
                    break
                
                idx_base = start_idx + (i * 3)
                landmarks_array[idx_base] = landmark.x
                landmarks_array[idx_base + 1] = landmark.y
                landmarks_array[idx_base + 2] = landmark.z
        
        return landmarks_array
    
    def draw_landmarks(self, image, results):
        """Draw landmarks on image for visualization"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        return image
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()

# Fungsi utilitas tambahan
def check_mediapipe_installation():
    """Check apakah MediaPipe terinstall dengan benar"""
    try:
        mp_version = mp.__version__
        logger.info(f"MediaPipe version: {mp_version}")
        return True
    except Exception as e:
        logger.error(f"MediaPipe not properly installed: {e}")
        return False

def test_landmark_extraction(test_image_path):
    """Test landmark extraction pada gambar tertentu"""
    logger.info(f"Testing landmark extraction on: {test_image_path}")
    
    landmarks = extract_hand_landmarks(test_image_path)
    
    if landmarks is not None:
        logger.info(f"✓ Landmarks extracted successfully")
        logger.info(f"  Shape: {landmarks.shape}")
        logger.info(f"  Range: {landmarks.min():.3f} to {landmarks.max():.3f}")
        logger.info(f"  Non-zero values: {np.count_nonzero(landmarks)}/126")
        return True
    else:
        logger.error(f"✗ Failed to extract landmarks")
        return False