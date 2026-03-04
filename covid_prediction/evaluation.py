"""
Model evaluation module for COVID-19 prediction model.

This module provides the ModelEvaluator class for computing comprehensive
performance metrics including accuracy, precision, recall, F1-score, AUC-ROC,
and confusion matrix.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class ModelEvaluator:
    """
    Evaluates trained models on test data.
    
    Computes comprehensive performance metrics including:
    - Accuracy
    - Precision (per class and weighted average)
    - Recall (per class and weighted average)
    - F1-score (per class and weighted average)
    - AUC-ROC
    - Confusion matrix
    """
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model with predict() and predict_proba() methods
            X_test: Test features (n_samples, n_features)
            y_test: Test labels (n_samples,)
            
        Returns:
            Dictionary containing all evaluation metrics:
                - accuracy: float
                - precision_positive: float
                - precision_negative: float
                - recall_positive: float
                - recall_negative: float
                - f1_positive: float
                - f1_negative: float
                - auc_roc: float
                - confusion_matrix: np.ndarray (2x2)
                - class_distribution: Dict[str, int]
        """
        # Generate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Compute confusion matrix
        cm = self.generate_confusion_matrix(y_test, y_pred)
        
        # Compute all metrics
        metrics = self.compute_metrics(y_test, y_pred, y_prob)
        
        # Add confusion matrix to metrics
        metrics['confusion_matrix'] = cm
        
        # Add class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        class_dist = {f'class_{int(cls)}': int(count) for cls, count in zip(unique, counts)}
        metrics['class_distribution'] = class_dist
        
        return metrics
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Metrics computed:
        - Accuracy
        - Precision (per class: negative=0, positive=1)
        - Recall (per class: negative=0, positive=1)
        - F1-score (per class: negative=0, positive=1)
        - AUC-ROC
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            y_prob: Prediction probabilities for positive class (n_samples,)
            
        Returns:
            Dictionary of metric names to values
        """
        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Compute precision for both classes
        # labels=[0, 1] ensures we get metrics for both negative and positive
        precision_per_class = precision_score(y_true, y_pred, labels=[0, 1], 
                                             average=None, zero_division=0)
        precision_negative = precision_per_class[0]
        precision_positive = precision_per_class[1]
        
        # Compute recall for both classes
        recall_per_class = recall_score(y_true, y_pred, labels=[0, 1], 
                                       average=None, zero_division=0)
        recall_negative = recall_per_class[0]
        recall_positive = recall_per_class[1]
        
        # Compute F1-score for both classes
        f1_per_class = f1_score(y_true, y_pred, labels=[0, 1], 
                               average=None, zero_division=0)
        f1_negative = f1_per_class[0]
        f1_positive = f1_per_class[1]
        
        # Compute AUC-ROC
        auc_roc = roc_auc_score(y_true, y_prob)
        
        return {
            'accuracy': float(accuracy),
            'precision_negative': float(precision_negative),
            'precision_positive': float(precision_positive),
            'recall_negative': float(recall_negative),
            'recall_positive': float(recall_positive),
            'f1_negative': float(f1_negative),
            'f1_positive': float(f1_positive),
            'auc_roc': float(auc_roc)
        }
    
    def generate_confusion_matrix(self, y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            
        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
            where rows are true labels and columns are predicted labels
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return cm
