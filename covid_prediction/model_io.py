"""
Model persistence for COVID-19 prediction model.

This module provides functionality to save and load trained models with
metadata and feature engineering transformations using joblib.
"""

import os
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List


def save_model(
    model: Any,
    metadata: Dict[str, Any],
    output_dir: str = 'models',
    feature_engineer: Optional[Any] = None
) -> str:
    """
    Save trained model with metadata and feature engineer.
    
    Args:
        model: Trained model object
        metadata: Dictionary containing:
            - algorithm: str
            - training_date: str
            - dataset_size: int
            - class_balance_method: str (optional)
            - feature_names: List[str]
            - feature_engineering_applied: bool
            - feature_engineer: FeatureEngineer object (if feature engineering was applied)
        output_dir: Directory to save model
        feature_engineer: Optional FeatureEngineer object to save with model
        
    Returns:
        Path to saved model file (includes timestamp)
        
    Raises:
        ValueError: If model or metadata is invalid
    """
    if model is None:
        raise ValueError(
            "[ModelIO] ValueError: Cannot save None model. "
            "Must provide a valid trained model."
        )
    
    if not metadata:
        raise ValueError(
            "[ModelIO] ValueError: Metadata cannot be empty. "
            "Must provide training metadata."
        )
    
    # Validate required metadata fields
    required_fields = ['algorithm', 'training_date', 'dataset_size', 'feature_names']
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        raise ValueError(
            f"[ModelIO] ValueError: Missing required metadata fields: {', '.join(missing_fields)}"
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    algorithm = metadata.get('algorithm', 'model')
    filename = f"{algorithm}_{timestamp}.joblib"
    filepath = os.path.join(output_dir, filename)
    
    # Package model with metadata and feature engineer
    model_package = {
        'model': model,
        'metadata': metadata,
        'feature_engineer': feature_engineer
    }
    
    # Save using joblib
    try:
        joblib.dump(model_package, filepath)
    except Exception as e:
        raise ValueError(
            f"[ModelIO] ValueError: Failed to save model - {str(e)}"
        )
    
    return filepath


def load_model(model_path: str) -> Tuple[Any, Dict[str, Any], Optional[Any]]:
    """
    Load trained model, metadata, and feature engineer from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (model, metadata, feature_engineer)
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is corrupted or invalid
    """
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[ModelIO] FileNotFoundError: Model file not found at '{model_path}'. "
            f"Please check the file path and try again."
        )
    
    # Load model package
    try:
        model_package = joblib.load(model_path)
    except Exception as e:
        raise ValueError(
            f"[ModelIO] ValueError: Failed to load model file - file may be corrupted. "
            f"Error: {str(e)}"
        )
    
    # Validate model package structure
    if not isinstance(model_package, dict):
        raise ValueError(
            "[ModelIO] ValueError: Invalid model file format - expected dictionary structure. "
            "File may be corrupted or from an incompatible version."
        )
    
    if 'model' not in model_package:
        raise ValueError(
            "[ModelIO] ValueError: Invalid model file - missing 'model' key. "
            "File may be corrupted."
        )
    
    if 'metadata' not in model_package:
        raise ValueError(
            "[ModelIO] ValueError: Invalid model file - missing 'metadata' key. "
            "File may be corrupted."
        )
    
    model = model_package['model']
    metadata = model_package['metadata']
    feature_engineer = model_package.get('feature_engineer', None)
    
    # Validate model is not None
    if model is None:
        raise ValueError(
            "[ModelIO] ValueError: Loaded model is None. "
            "File may be corrupted."
        )
    
    # Validate metadata is a dictionary
    if not isinstance(metadata, dict):
        raise ValueError(
            "[ModelIO] ValueError: Invalid metadata format - expected dictionary. "
            "File may be corrupted."
        )
    
    return model, metadata, feature_engineer


def verify_model_compatibility(
    metadata: Dict[str, Any],
    expected_features: List[str]
) -> bool:
    """
    Verify loaded model is compatible with current feature schema.
    
    Args:
        metadata: Model metadata
        expected_features: List of expected feature names
        
    Returns:
        True if compatible
        
    Raises:
        ValueError: If incompatible (with details)
    """
    if not metadata:
        raise ValueError(
            "[ModelIO] ValueError: Metadata is empty. "
            "Cannot verify compatibility."
        )
    
    if 'feature_names' not in metadata:
        raise ValueError(
            "[ModelIO] ValueError: Metadata missing 'feature_names' field. "
            "Cannot verify feature schema compatibility."
        )
    
    model_features = metadata['feature_names']
    
    # Check if feature lists match
    if len(model_features) != len(expected_features):
        raise ValueError(
            f"[ModelIO] ValueError: Feature schema mismatch - "
            f"model expects {len(model_features)} features but got {len(expected_features)} features. "
            f"Model features: {model_features[:5]}... "
            f"Expected features: {expected_features[:5]}..."
        )
    
    # Check if feature names match (order matters for model predictions)
    mismatched_features = []
    for i, (model_feat, expected_feat) in enumerate(zip(model_features, expected_features)):
        if model_feat != expected_feat:
            mismatched_features.append((i, model_feat, expected_feat))
    
    if mismatched_features:
        mismatch_details = [
            f"Position {i}: model has '{model_feat}', expected '{expected_feat}'"
            for i, model_feat, expected_feat in mismatched_features[:3]
        ]
        raise ValueError(
            f"[ModelIO] ValueError: Feature schema mismatch - "
            f"feature names do not match. "
            f"Mismatches: {'; '.join(mismatch_details)}"
            + (f" (and {len(mismatched_features) - 3} more)" if len(mismatched_features) > 3 else "")
        )
    
    return True
