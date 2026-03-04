"""
Data models and schemas for COVID-19 prediction system.

This module consolidates all data models and schemas used throughout the system,
including input features, prediction results, training metadata, and evaluation reports.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class FeatureVector:
    """
    Input features for COVID-19 prediction.
    
    Attributes:
        cough: Binary indicator (0 or 1) for cough symptom
        fever: Binary indicator (0 or 1) for fever symptom
        sore_throat: Binary indicator (0 or 1) for sore throat symptom
        shortness_of_breath: Binary indicator (0 or 1) for shortness of breath symptom
        head_ache: Binary indicator (0 or 1) for headache symptom
        age_60_and_above: Age category ('Yes' or 'No')
        gender: Gender ('male' or 'female')
        test_indication: Reason for COVID test (free text)
    """
    cough: int  # 0 or 1
    fever: int  # 0 or 1
    sore_throat: int  # 0 or 1
    shortness_of_breath: int  # 0 or 1
    head_ache: int  # 0 or 1
    age_60_and_above: str  # 'Yes' or 'No'
    gender: str  # 'male' or 'female'
    test_indication: str  # Free text


@dataclass
class PredictionResult:
    """
    Output from prediction service.
    
    Attributes:
        predicted_class: Predicted COVID test result ('positive' or 'negative')
        confidence: Prediction confidence probability between 0.0 and 1.0
        timestamp: ISO format timestamp of when prediction was made
    """
    predicted_class: str  # 'positive' or 'negative'
    confidence: float  # Probability between 0.0 and 1.0
    timestamp: str  # ISO format timestamp


@dataclass
class TrainingMetadata:
    """
    Metadata for trained model.
    
    This metadata is saved with the model to track training configuration,
    feature engineering details, and class distribution information.
    
    Attributes:
        algorithm: ML algorithm used ('logistic_regression', 'random_forest', 'gradient_boosting')
        training_date: ISO format timestamp of when model was trained
        dataset_size: Number of training samples used
        class_balance_method: Method used for class balancing (optional)
        feature_names: Ordered list of feature names after feature engineering
        hyperparameters: Algorithm-specific hyperparameters used
        class_distribution: Count of each class in training data
        feature_engineering_applied: Whether feature engineering was applied
        original_feature_count: Number of features before engineering
        engineered_feature_count: Number of features after engineering
    """
    algorithm: str  # 'logistic_regression', 'random_forest', 'gradient_boosting'
    training_date: str  # ISO format
    dataset_size: int  # Number of training samples
    class_balance_method: Optional[str]  # 'class_weights', 'smote', 'undersample', or None
    feature_names: List[str]  # Ordered list of feature names (after feature engineering)
    hyperparameters: Dict[str, Any]  # Algorithm-specific parameters
    class_distribution: Dict[str, int]  # Count of each class in training data
    feature_engineering_applied: bool  # Whether feature engineering was applied
    original_feature_count: int  # Number of features before engineering
    engineered_feature_count: int  # Number of features after engineering


@dataclass
class EvaluationReport:
    """
    Model evaluation results.
    
    Contains comprehensive performance metrics for model evaluation,
    including per-class metrics and confusion matrix.
    
    Attributes:
        accuracy: Overall accuracy score (0.0 to 1.0)
        precision_positive: Precision for positive class (0.0 to 1.0)
        precision_negative: Precision for negative class (0.0 to 1.0)
        recall_positive: Recall for positive class (0.0 to 1.0)
        recall_negative: Recall for negative class (0.0 to 1.0)
        f1_positive: F1-score for positive class (0.0 to 1.0)
        f1_negative: F1-score for negative class (0.0 to 1.0)
        auc_roc: Area under ROC curve (0.0 to 1.0)
        confusion_matrix: 2x2 numpy array [[TN, FP], [FN, TP]]
        class_distribution: Test set class counts
    """
    accuracy: float
    precision_positive: float
    precision_negative: float
    recall_positive: float
    recall_negative: float
    f1_positive: float
    f1_negative: float
    auc_roc: float
    confusion_matrix: np.ndarray  # 2x2 array
    class_distribution: Dict[str, int]  # Test set class counts
