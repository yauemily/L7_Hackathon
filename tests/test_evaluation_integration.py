"""
Integration tests for model evaluation with actual trained models.
"""

import pytest
import numpy as np
from covid_prediction.data_loader import CovidDatasetLoader
from covid_prediction.preprocessor import CovidDataPreprocessor
from covid_prediction.feature_engineering import FeatureEngineer
from covid_prediction.model_io import load_model
from covid_prediction.evaluation import ModelEvaluator


def test_evaluate_with_actual_model():
    """Test evaluation with an actual trained model."""
    # Load and preprocess data
    loader = CovidDatasetLoader()
    df = loader.load_dataset("Data/corona_tested_individuals_ver_006.english.csv")
    
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2, random_state=42)
    
    # Load a trained model
    try:
        model, metadata, feature_engineer = load_model("models/random_forest_20260304_201857.joblib")
    except FileNotFoundError:
        pytest.skip("No trained model found")
    
    # Apply feature engineering if needed
    if feature_engineer is not None:
        X_test = feature_engineer.transform(X_test)
    
    # Evaluate
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, X_test, y_test)
    
    # Verify report structure
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
    
    # Verify metrics are in valid ranges
    assert 0 <= report['accuracy'] <= 1
    assert 0 <= report['auc_roc'] <= 1
    
    # Verify confusion matrix
    cm = report['confusion_matrix']
    assert cm.shape == (2, 2)
    assert np.all(cm >= 0)
    assert cm.sum() == len(y_test)
    
    # Print report for manual inspection
    print("\n=== Evaluation Report ===")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"AUC-ROC: {report['auc_roc']:.4f}")
    print(f"\nNegative Class (0):")
    print(f"  Precision: {report['precision_negative']:.4f}")
    print(f"  Recall: {report['recall_negative']:.4f}")
    print(f"  F1-Score: {report['f1_negative']:.4f}")
    print(f"\nPositive Class (1):")
    print(f"  Precision: {report['precision_positive']:.4f}")
    print(f"  Recall: {report['recall_positive']:.4f}")
    print(f"  F1-Score: {report['f1_positive']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print(f"\nClass Distribution: {report['class_distribution']}")


def test_evaluate_all_algorithms():
    """Test evaluation with all three algorithm types."""
    # Load and preprocess data
    loader = CovidDatasetLoader()
    df = loader.load_dataset("Data/corona_tested_individuals_ver_006.english.csv")
    
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2, random_state=42)
    
    evaluator = ModelEvaluator()
    
    # Try to evaluate each algorithm
    algorithms = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    for algorithm in algorithms:
        # Find a model file for this algorithm
        import os
        model_files = [f for f in os.listdir('models') if f.startswith(algorithm) and f.endswith('.joblib')]
        
        if not model_files:
            print(f"\nSkipping {algorithm} - no model found")
            continue
        
        model_path = os.path.join('models', model_files[0])
        model, metadata, feature_engineer = load_model(model_path)
        
        # Apply feature engineering if needed
        # Check both metadata flag and presence of feature_engineer object
        if feature_engineer is not None:
            X_test_transformed = feature_engineer.transform(X_test)
        else:
            X_test_transformed = X_test
        
        # Evaluate
        report = evaluator.evaluate(model, X_test_transformed, y_test)
        
        print(f"\n=== {algorithm.upper()} ===")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"AUC-ROC: {report['auc_roc']:.4f}")
        print(f"F1 (Positive): {report['f1_positive']:.4f}")
        print(f"F1 (Negative): {report['f1_negative']:.4f}")
        
        # Verify all metrics are valid
        assert 0 <= report['accuracy'] <= 1
        assert 0 <= report['auc_roc'] <= 1
