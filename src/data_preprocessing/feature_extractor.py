# src/data_preprocessing/feature_extractor.py - OPTIMIZED VERSION
# REPLACE the existing feature_extractor.py with this file

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_angle_between_points(p1, p2, p3):
    """Calculate angle between three points with p2 as vertex"""
    try:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle if np.isfinite(angle) else 0.0
    except:
        return 0.0

def extract_hand_normalized_features(features_df, hand_idx):
    """Extract normalized features for one hand"""
    base_idx = hand_idx * 21
    
    # Key landmark indices
    key_landmarks = {
        'wrist': 0, 'thumb_tip': 4, 'index_tip': 8, 'middle_tip': 12, 
        'ring_tip': 16, 'pinky_tip': 20, 'index_pip': 6, 'middle_pip': 10
    }
    
    # Check if hand landmarks exist
    wrist_col = f'landmark_{base_idx}_x'
    if wrist_col not in features_df.columns:
        # Create zero features for missing hand
        for name in key_landmarks.keys():
            if name != 'wrist':
                features_df[f'h{hand_idx}_norm_{name}_x'] = 0.0
                features_df[f'h{hand_idx}_norm_{name}_y'] = 0.0
                features_df[f'h{hand_idx}_dist_{name}'] = 0.0
        return features_df
    
    # Get wrist position for normalization
    wrist_x = features_df[f'landmark_{base_idx}_x'].fillna(0)
    wrist_y = features_df[f'landmark_{base_idx}_y'].fillna(0)
    
    # Normalize positions relative to wrist
    for name, rel_idx in key_landmarks.items():
        if name == 'wrist':
            continue
            
        abs_idx = base_idx + rel_idx
        x_col = f'landmark_{abs_idx}_x'
        y_col = f'landmark_{abs_idx}_y'
        
        if x_col in features_df.columns and y_col in features_df.columns:
            # Normalized position
            norm_x = features_df[x_col].fillna(0) - wrist_x
            norm_y = features_df[y_col].fillna(0) - wrist_y
            
            features_df[f'h{hand_idx}_norm_{name}_x'] = norm_x
            features_df[f'h{hand_idx}_norm_{name}_y'] = norm_y
            
            # Distance from wrist
            features_df[f'h{hand_idx}_dist_{name}'] = np.sqrt(norm_x**2 + norm_y**2)
        else:
            features_df[f'h{hand_idx}_norm_{name}_x'] = 0.0
            features_df[f'h{hand_idx}_norm_{name}_y'] = 0.0
            features_df[f'h{hand_idx}_dist_{name}'] = 0.0
    
    return features_df

def extract_finger_angles(features_df, hand_idx):
    """Extract finger angles for one hand"""
    base_idx = hand_idx * 21
    
    # Finger joint chains
    finger_chains = [
        ('thumb', [0, 1, 2, 4]),
        ('index', [0, 5, 6, 8]),
        ('middle', [0, 9, 10, 12]),
        ('ring', [0, 13, 14, 16]),
        ('pinky', [0, 17, 18, 20])
    ]
    
    for finger_name, chain in finger_chains:
        if len(chain) >= 3:
            p1_idx = base_idx + chain[0]  # wrist
            p2_idx = base_idx + chain[1]  # mcp
            p3_idx = base_idx + chain[2]  # pip
            
            cols_needed = [f'landmark_{idx}_{coord}' for idx in [p1_idx, p2_idx, p3_idx] 
                          for coord in ['x', 'y']]
            
            if all(col in features_df.columns for col in cols_needed):
                angles = []
                for row_idx in range(len(features_df)):
                    p1 = (features_df[f'landmark_{p1_idx}_x'].iloc[row_idx], 
                          features_df[f'landmark_{p1_idx}_y'].iloc[row_idx])
                    p2 = (features_df[f'landmark_{p2_idx}_x'].iloc[row_idx], 
                          features_df[f'landmark_{p2_idx}_y'].iloc[row_idx])
                    p3 = (features_df[f'landmark_{p3_idx}_x'].iloc[row_idx], 
                          features_df[f'landmark_{p3_idx}_y'].iloc[row_idx])
                    
                    angle = calculate_angle_between_points(p1, p2, p3)
                    angles.append(angle)
                
                features_df[f'h{hand_idx}_angle_{finger_name}'] = angles
            else:
                features_df[f'h{hand_idx}_angle_{finger_name}'] = 0.0
    
    return features_df

def extract_inter_finger_distances(features_df, hand_idx):
    """Extract distances between fingertips"""
    base_idx = hand_idx * 21
    fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    fingertip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    for i, (tip1_idx, tip1_name) in enumerate(zip(fingertip_indices, fingertip_names)):
        for tip2_idx, tip2_name in zip(fingertip_indices[i+1:], fingertip_names[i+1:]):
            abs_idx1 = base_idx + tip1_idx
            abs_idx2 = base_idx + tip2_idx
            
            cols_needed = [f'landmark_{abs_idx1}_x', f'landmark_{abs_idx1}_y',
                          f'landmark_{abs_idx2}_x', f'landmark_{abs_idx2}_y']
            
            if all(col in features_df.columns for col in cols_needed):
                dx = features_df[f'landmark_{abs_idx1}_x'].fillna(0) - features_df[f'landmark_{abs_idx2}_x'].fillna(0)
                dy = features_df[f'landmark_{abs_idx1}_y'].fillna(0) - features_df[f'landmark_{abs_idx2}_y'].fillna(0)
                dist = np.sqrt(dx**2 + dy**2)
                features_df[f'h{hand_idx}_interdist_{tip1_name}_{tip2_name}'] = dist
            else:
                features_df[f'h{hand_idx}_interdist_{tip1_name}_{tip2_name}'] = 0.0
    
    return features_df

def extract_hand_geometry(features_df, hand_idx):
    """Extract geometric features for one hand"""
    base_idx = hand_idx * 21
    
    # Get all landmark columns for this hand
    x_cols = [f'landmark_{base_idx + i}_x' for i in range(21) 
              if f'landmark_{base_idx + i}_x' in features_df.columns]
    y_cols = [f'landmark_{base_idx + i}_y' for i in range(21) 
              if f'landmark_{base_idx + i}_y' in features_df.columns]
    
    if x_cols and y_cols:
        # Bounding box
        x_min = features_df[x_cols].fillna(0).min(axis=1)
        x_max = features_df[x_cols].fillna(0).max(axis=1)
        y_min = features_df[y_cols].fillna(0).min(axis=1)
        y_max = features_df[y_cols].fillna(0).max(axis=1)
        
        width = x_max - x_min
        height = y_max - y_min
        
        features_df[f'h{hand_idx}_bbox_width'] = width
        features_df[f'h{hand_idx}_bbox_height'] = height
        features_df[f'h{hand_idx}_bbox_aspect'] = width / (height + 1e-8)
        features_df[f'h{hand_idx}_bbox_area'] = width * height
        
        # Centroid
        features_df[f'h{hand_idx}_centroid_x'] = features_df[x_cols].fillna(0).mean(axis=1)
        features_df[f'h{hand_idx}_centroid_y'] = features_df[y_cols].fillna(0).mean(axis=1)
        
        # Hand span (thumb to pinky)
        thumb_tip = base_idx + 4
        pinky_tip = base_idx + 20
        
        if (f'landmark_{thumb_tip}_x' in features_df.columns and 
            f'landmark_{pinky_tip}_x' in features_df.columns):
            dx = features_df[f'landmark_{thumb_tip}_x'].fillna(0) - features_df[f'landmark_{pinky_tip}_x'].fillna(0)
            dy = features_df[f'landmark_{thumb_tip}_y'].fillna(0) - features_df[f'landmark_{pinky_tip}_y'].fillna(0)
            features_df[f'h{hand_idx}_span'] = np.sqrt(dx**2 + dy**2)
        else:
            features_df[f'h{hand_idx}_span'] = 0.0
    else:
        # No landmarks available - create zero features
        for feature in ['bbox_width', 'bbox_height', 'bbox_aspect', 'bbox_area', 
                       'centroid_x', 'centroid_y', 'span']:
            features_df[f'h{hand_idx}_{feature}'] = 0.0
    
    return features_df

def extract_two_hand_features(features_df):
    """Extract features for two-hand interactions (BISINDO)"""
    # Check if both hands exist
    if not (f'landmark_0_x' in features_df.columns and f'landmark_21_x' in features_df.columns):
        # Single hand mode
        features_df['inter_hand_distance'] = 0.0
        features_df['hand_relative_x'] = 0.0
        features_df['hand_relative_y'] = 0.0
        return features_df
    
    # Distance between wrists
    dx_wrist = features_df['landmark_0_x'].fillna(0) - features_df['landmark_21_x'].fillna(0)
    dy_wrist = features_df['landmark_0_y'].fillna(0) - features_df['landmark_21_y'].fillna(0)
    features_df['inter_hand_distance'] = np.sqrt(dx_wrist**2 + dy_wrist**2)
    features_df['hand_relative_x'] = dx_wrist
    features_df['hand_relative_y'] = dy_wrist
    
    # Cross-hand fingertip distances (most important for BISINDO)
    key_tips = [4, 8, 12]  # thumb, index, middle
    for tip1 in key_tips:
        for tip2 in key_tips:
            idx1 = tip1  # hand 0
            idx2 = 21 + tip2  # hand 1
            
            cols_needed = [f'landmark_{idx1}_x', f'landmark_{idx1}_y',
                          f'landmark_{idx2}_x', f'landmark_{idx2}_y']
            
            if all(col in features_df.columns for col in cols_needed):
                dx = features_df[f'landmark_{idx1}_x'].fillna(0) - features_df[f'landmark_{idx2}_x'].fillna(0)
                dy = features_df[f'landmark_{idx1}_y'].fillna(0) - features_df[f'landmark_{idx2}_y'].fillna(0)
                dist = np.sqrt(dx**2 + dy**2)
                features_df[f'cross_dist_{tip1}_{tip2}'] = dist
            else:
                features_df[f'cross_dist_{tip1}_{tip2}'] = 0.0
    
    return features_df

def extract_core_features(df_landmarks):
    """Extract core discriminative features for sign language recognition"""
    features_df = df_landmarks.copy()
    
    # Replace infinite values
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_columns] = features_df[numeric_columns].replace([np.inf, -np.inf], 0)
    
    # Extract features for each hand
    for hand_idx in range(2):
        features_df = extract_hand_normalized_features(features_df, hand_idx)
        features_df = extract_finger_angles(features_df, hand_idx)
        features_df = extract_inter_finger_distances(features_df, hand_idx)
        features_df = extract_hand_geometry(features_df, hand_idx)
    
    # Extract two-hand features
    features_df = extract_two_hand_features(features_df)
    
    return features_df

def select_best_features(features_df, max_features=80):
    """Select most important features using statistical methods"""
    if 'label' not in features_df.columns:
        logger.warning("No label column found, skipping feature selection")
        return features_df
    
    # Separate features and labels
    X = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
    y = features_df['label']
    
    # Remove constant features
    constant_features = X.columns[X.std() == 0]
    if len(constant_features) > 0:
        X = X.drop(columns=constant_features)
        logger.info(f"Removed {len(constant_features)} constant features")
    
    # Remove highly correlated features
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_features = [column for column in upper_triangle.columns 
                         if any(upper_triangle[column] > 0.95)]
    
    if high_corr_features:
        X = X.drop(columns=high_corr_features)
        logger.info(f"Removed {len(high_corr_features)} highly correlated features")
    
    # Feature selection if we have too many features
    if len(X.columns) > max_features:
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            
            logger.info(f"Selected {len(selected_features)} best features from {len(X.columns)}")
            
            # Create result dataframe
            result_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            result_df['label'] = y
            if 'sign_language_type' in features_df.columns:
                result_df['sign_language_type'] = features_df['sign_language_type']
            
            return result_df
        
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using all features")
    
    # Return original if selection fails or not needed
    result_df = X.copy()
    result_df['label'] = y
    if 'sign_language_type' in features_df.columns:
        result_df['sign_language_type'] = features_df['sign_language_type']
    
    return result_df

def extract_features(df_landmarks, max_features=80, perform_selection=True):
    """
    Main function for optimized feature extraction
    
    Args:
        df_landmarks (pd.DataFrame): DataFrame with landmark columns + label
        max_features (int): Maximum number of features to select
        perform_selection (bool): Whether to perform feature selection
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    logger.info("Starting optimized feature extraction...")
    
    if df_landmarks.empty:
        logger.warning("Input DataFrame is empty")
        return pd.DataFrame()
    
    try:
        # Extract core features
        features_df = extract_core_features(df_landmarks)
        
        # Get feature columns (exclude original landmarks and labels)
        feature_columns = [col for col in features_df.columns 
                          if not col.startswith('landmark_') and col not in ['label', 'sign_language_type']]
        
        if not feature_columns:
            logger.error("No feature columns generated")
            return df_landmarks
        
        # Clean features
        for col in feature_columns:
            features_df[col] = features_df[col].fillna(0)
            features_df[col] = features_df[col].replace([np.inf, -np.inf], 0)
            # Ensure finite values
            features_df[col] = np.where(np.isfinite(features_df[col]), features_df[col], 0)
        
        # Keep only feature columns + labels
        keep_columns = feature_columns.copy()
        if 'label' in features_df.columns:
            keep_columns.append('label')
        if 'sign_language_type' in features_df.columns:
            keep_columns.append('sign_language_type')
        
        features_df = features_df[keep_columns]
        
        logger.info(f"Generated {len(feature_columns)} core features")
        
        # Feature selection
        if perform_selection and len(feature_columns) > max_features:
            logger.info("Performing feature selection...")
            features_df = select_best_features(features_df, max_features)
        
        # Final validation
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = features_df[numeric_cols].describe()
            logger.info(f"Final feature stats - Shape: {features_df.shape}")
            logger.info(f"Feature range: {stats.loc['min'].min():.3f} to {stats.loc['max'].max():.3f}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        
        # Fallback: create basic features
        basic_features = [f'basic_feature_{i}' for i in range(20)]
        basic_df = pd.DataFrame(0, index=range(len(df_landmarks)), columns=basic_features)
        
        if 'label' in df_landmarks.columns:
            basic_df['label'] = df_landmarks['label']
        if 'sign_language_type' in df_landmarks.columns:
            basic_df['sign_language_type'] = df_landmarks['sign_language_type']
            
        logger.info("Using fallback basic features")
        return basic_df

def analyze_feature_importance(features_df, top_k=15):
    """Analyze and report most important features"""
    if 'label' not in features_df.columns:
        return None
    
    X = features_df.drop(columns=['label', 'sign_language_type'], errors='ignore')
    y = features_df['label']
    
    try:
        scores = mutual_info_classif(X, y, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top {top_k} most important features:")
        for idx, row in feature_importance.head(top_k).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        return None