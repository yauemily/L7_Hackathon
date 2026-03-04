"""
Unit tests for model evaluation module.
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from covid_prediction.evaluation import ModelEvaluator


def test_evaluate_with_known_predictions():
    """Test evaluation with known predictions."""
    # Create simple test data
    X_test = np.array([[1, 0, 1, 0, 1],
                       [0, 1, 0, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0]])
    y_test = np.array([1, 0, 1, 0])
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_test, y_test)
    
    # Evaluate
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, X_test, y_test)
    
    # Check that all required metrics are present
    assert 'accuracy' in report
    assert 'precision_positive' in report
    assert 'precision_negative' in report
    assert 'recall_positive' in report
    assert 'recall_negative' in report
    assert 'f1_positive' in report
    assert 'f1_negative' in report
    assert 'auc_roc' in report
    assert 'confusion_matrix' in report
    assert 'class_distribution' in report
    
    # Check that metrics are in valid ranges
    assert 0 <= report['accuracy'] <= 1
    assert 0 <= report['precision_positive'] <= 1
    assert 0 <= report['precision_negative'] <= 1
    assert 0 <= report['recall_positive'] <= 1
    assert 0 <= report['recall_negative'] <= 1
    assert 0 <= report['f1_positive'] <= 1
    assert 0 <= report['f1_negative'] <= 1
    assert 0 <= report['auc_roc'] <= 1
    
    # Check confusion matrix shape
    assert report['confusion_matrix'].shape == (2, 2)
    assert np.all(report['confusion_matrix'] >= 0)


def test_compute_metrics():
    """Test compute_metrics with known values."""
    evaluator = ModelEvaluator()
    
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    # Perfect predictions should have accuracy = 1.0
    assert metrics['accuracy'] == 1.0
    assert metrics['precision_positive'] == 1.0
    assert metrics['precision_negative'] == 1.0
    assert metrics['recall_positive'] == 1.0
    assert metrics['recall_negative'] == 1.0
    assert metrics['f1_positive'] == 1.0
    assert metrics['f1_negative'] == 1.0
    assert metrics['auc_roc'] == 1.0


def test_generate_confusion_matrix():
    """Test confusion matrix generation."""
    evaluator = ModelEvaluator()
    
    # Known predictions
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    
    cm = evaluator.generate_confusion_matrix(y_true, y_pred)
    
    # Check shape
    assert cm.shape == (2, 2)
    
    # Manually compute expected confusion matrix
    # TN (true 0, pred 0): 2
    # FP (true 0, pred 1): 1
    # FN (true 1, pred 0): 1
    # TP (true 1, pred 1): 2
    expected_cm = np.array([[2, 1],
                           [1, 2]])
    
    np.testing.assert_array_equal(cm, expected_cm)


def test_metrics_for_both_classes():
    """Test that metrics are computed for both positive and negative classes."""
    evaluator = ModelEvaluator()
    
    # Create imbalanced predictions
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.9])
    
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    # Check that we have separate metrics for both classes
    assert 'precision_negative' in metrics
    assert 'precision_positive' in metrics
    assert 'recall_negative' in metrics
    assert 'recall_positive' in metrics
    assert 'f1_negative' in metrics
    assert 'f1_positive' in metrics
    
    # All metrics should be valid (between 0 and 1)
    for key, value in metrics.items():
        assert 0 <= value <= 1, f"{key} = {value} is not in [0, 1]"


def test_auc_roc_computation():
    """Test AUC-ROC computation."""
    evaluator = ModelEvaluator()
    
    # Perfect separation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    # Perfect separation should give AUC-ROC = 1.0
    assert metrics['auc_roc'] == 1.0
    
    # Random predictions (probabilities don't match labels well)
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    
    # Random probabilities should give AUC-ROC around 0.5
    assert 0.4 <= metrics['auc_roc'] <= 0.6
