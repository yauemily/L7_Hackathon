"""
Prediction service for COVID-19 prediction model.

This module provides functionality to load trained models and make predictions
on new data with comprehensive input validation.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from covid_prediction.model_io import load_model, verify_model_compatibility
from covid_prediction.preprocessor import CovidDataPreprocessor
from covid_prediction.models import PredictionResult


class PredictionService:
    """
    Service for making COVID-19 predictions with input validation.
    
    This service loads a trained model and feature engineer, validates input
    features, applies necessary transformations, and generates predictions.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize prediction service with trained model.
        
        Args:
            model_path: Path to saved model file
            
        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If model file is corrupted
        """
        # Load model, metadata, and feature engineer
        self.model, self.metadata, self.feature_engineer = load_model(model_path)
        
        # Initialize preprocessor for encoding categorical variables
        self.preprocessor = CovidDataPreprocessor()
        
        # Load test_indication encoder from metadata if available
        if 'test_indication_encoder' in self.metadata:
            self.preprocessor.test_indication_encoder = self.metadata['test_indication_encoder']
        
        # Extract expected feature names from metadata
        self.expected_features = self._get_original_feature_names()
        
        # Required input features
        self.required_features = [
            'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
            'age_60_and_above', 'gender', 'test_indication'
        ]
    
    def _get_original_feature_names(self) -> list:
        """
        Extract original feature names before feature engineering.
        
        Returns:
            List of original feature names
        """
        # Standard feature order for COVID prediction
        return [
            'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
            'age_60_and_above', 'gender', 'test_indication'
        ]
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate input features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails (with specific error message)
        """
        # Check for missing features
        missing_features = [f for f in self.required_features if f not in features]
        if missing_features:
            raise ValueError(
                f"[PredictionService] ValueError: Missing required features. "
                f"Expected: {', '.join(missing_features)}"
            )
        
        # Validate symptom values (must be 0 or 1)
        symptom_features = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']
        for symptom in symptom_features:
            value = features[symptom]
            if value not in [0, 1]:
                raise ValueError(
                    f"[PredictionService] ValueError: Invalid symptom value. "
                    f"Expected: 0 or 1, Got: {value} for feature '{symptom}'"
                )
        
        # Validate gender (must be 'male' or 'female')
        gender = features['gender']
        if gender not in ['male', 'female']:
            raise ValueError(
                f"[PredictionService] ValueError: Invalid gender. "
                f"Expected: 'male' or 'female', Got: '{gender}'"
            )
        
        # Validate age_60_and_above (must be 'Yes' or 'No')
        age = features['age_60_and_above']
        if age not in ['Yes', 'No']:
            raise ValueError(
                f"[PredictionService] ValueError: Invalid age value. "
                f"Expected: 'Yes' or 'No', Got: '{age}'"
            )
        
        # Validate test_indication (must not be empty)
        test_indication = features['test_indication']
        if not test_indication or (isinstance(test_indication, str) and test_indication.strip() == ''):
            raise ValueError(
                "[PredictionService] ValueError: Missing test indication. "
                "Expected: non-empty string"
            )
        
        return True
    
    def predict(self, features: Dict[str, Any]) -> PredictionResult:
        """
        Generate prediction for input features.
        
        Args:
            features: Dictionary containing:
                - cough: int (0 or 1)
                - fever: int (0 or 1)
                - sore_throat: int (0 or 1)
                - shortness_of_breath: int (0 or 1)
                - head_ache: int (0 or 1)
                - age_60_and_above: str ('Yes' or 'No')
                - gender: str ('male' or 'female')
                - test_indication: str
                
        Returns:
            PredictionResult with predicted_class and confidence
            
        Raises:
            ValueError: If features are invalid or missing
        """
        # Validate input features
        self.validate_features(features)
        
        # Encode categorical variables
        encoded_features = self._encode_features(features)
        
        # Create feature vector in correct order
        feature_vector = np.array([[
            encoded_features['cough'],
            encoded_features['fever'],
            encoded_features['sore_throat'],
            encoded_features['shortness_of_breath'],
            encoded_features['head_ache'],
            encoded_features['age_60_and_above'],
            encoded_features['gender'],
            encoded_features['test_indication']
        ]])
        
        # Apply feature engineering transformations if needed
        if self.feature_engineer is not None:
            feature_vector = self.feature_engineer.transform(feature_vector)
        
        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        
        # Get prediction probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_vector)[0]
            # Confidence is the probability of the predicted class
            confidence = float(probabilities[prediction])
        else:
            # If model doesn't support predict_proba, use binary prediction
            confidence = 1.0
        
        # Convert prediction to class label
        predicted_class = 'positive' if prediction == 1 else 'negative'
        
        # Create result with timestamp
        timestamp = datetime.now().isoformat()
        
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            timestamp=timestamp
        )
    
    def _encode_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode categorical features to numeric values.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Dictionary with encoded features
        """
        encoded = features.copy()
        
        # Encode gender: male=0, female=1
        encoded['gender'] = 0 if features['gender'] == 'male' else 1
        
        # Encode age_60_and_above: No=0, Yes=1
        encoded['age_60_and_above'] = 0 if features['age_60_and_above'] == 'No' else 1
        
        # Encode test_indication using the preprocessor's encoder
        # For prediction, we need to handle unseen test_indication values
        test_indication = str(features['test_indication'])
        
        # Check if the encoder has been fitted (has classes_)
        if hasattr(self.preprocessor.test_indication_encoder, 'classes_'):
            # Check if the test_indication is in the known classes
            if test_indication in self.preprocessor.test_indication_encoder.classes_:
                encoded['test_indication'] = self.preprocessor.test_indication_encoder.transform([test_indication])[0]
            else:
                # For unseen test_indication, use a default value (e.g., 0 or most common)
                encoded['test_indication'] = 0
        else:
            # If encoder not fitted, use a default value
            encoded['test_indication'] = 0
        
        return encoded
