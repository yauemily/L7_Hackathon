"""
Basic tests for prediction service to verify implementation.
"""

import pytest
import numpy as np
from covid_prediction.prediction import PredictionService, PredictionResult


def test_prediction_result_dataclass():
    """Test PredictionResult dataclass instantiation."""
    result = PredictionResult(
        predicted_class='positive',
        confidence=0.85,
        timestamp='2024-01-01T12:00:00'
    )
    
    assert result.predicted_class == 'positive'
    assert result.confidence == 0.85
    assert result.timestamp == '2024-01-01T12:00:00'


def test_validate_features_missing_features():
    """Test validation detects missing features."""
    # Create a mock model file first
    from covid_prediction.training import TrainingPipeline
    from covid_prediction.model_io import save_model
    
    # Create minimal training data with both classes
    X_train = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1]
    ])
    y_train = np.array([1, 0])
    
    # Train a simple model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_train, y_train, algorithm='logistic_regression', balance_classes=False)
    
    # Save model
    metadata = {
        'algorithm': 'logistic_regression',
        'training_date': '2024-01-01',
        'dataset_size': 1,
        'feature_names': ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                         'age_60_and_above', 'gender', 'test_indication'],
        'feature_engineering_applied': False,
        'original_feature_count': 8,
        'engineered_feature_count': 8
    }
    model_path = save_model(model, metadata, output_dir='models')
    
    # Create prediction service
    service = PredictionService(model_path)
    
    # Test with missing features
    incomplete_features = {
        'cough': 1,
        'fever': 0
        # Missing other required features
    }
    
    with pytest.raises(ValueError) as exc_info:
        service.validate_features(incomplete_features)
    
    assert "Missing required features" in str(exc_info.value)


def test_validate_features_invalid_symptom():
    """Test validation detects invalid symptom values."""
    from covid_prediction.training import TrainingPipeline
    from covid_prediction.model_io import save_model
    
    # Create minimal training data with both classes
    X_train = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1]
    ])
    y_train = np.array([1, 0])
    
    # Train a simple model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_train, y_train, algorithm='logistic_regression', balance_classes=False)
    
    # Save model
    metadata = {
        'algorithm': 'logistic_regression',
        'training_date': '2024-01-01',
        'dataset_size': 1,
        'feature_names': ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                         'age_60_and_above', 'gender', 'test_indication'],
        'feature_engineering_applied': False,
        'original_feature_count': 8,
        'engineered_feature_count': 8
    }
    model_path = save_model(model, metadata, output_dir='models')
    
    # Create prediction service
    service = PredictionService(model_path)
    
    # Test with invalid symptom value
    invalid_features = {
        'cough': 2,  # Invalid: should be 0 or 1
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'male',
        'test_indication': 'Contact with confirmed'
    }
    
    with pytest.raises(ValueError) as exc_info:
        service.validate_features(invalid_features)
    
    assert "Invalid symptom value" in str(exc_info.value)
    assert "cough" in str(exc_info.value)


def test_validate_features_invalid_gender():
    """Test validation detects invalid gender values."""
    from covid_prediction.training import TrainingPipeline
    from covid_prediction.model_io import save_model
    
    # Create minimal training data with both classes
    X_train = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1]
    ])
    y_train = np.array([1, 0])
    
    # Train a simple model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_train, y_train, algorithm='logistic_regression', balance_classes=False)
    
    # Save model
    metadata = {
        'algorithm': 'logistic_regression',
        'training_date': '2024-01-01',
        'dataset_size': 1,
        'feature_names': ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                         'age_60_and_above', 'gender', 'test_indication'],
        'feature_engineering_applied': False,
        'original_feature_count': 8,
        'engineered_feature_count': 8
    }
    model_path = save_model(model, metadata, output_dir='models')
    
    # Create prediction service
    service = PredictionService(model_path)
    
    # Test with invalid gender
    invalid_features = {
        'cough': 1,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'other',  # Invalid: should be 'male' or 'female'
        'test_indication': 'Contact with confirmed'
    }
    
    with pytest.raises(ValueError) as exc_info:
        service.validate_features(invalid_features)
    
    assert "Invalid gender" in str(exc_info.value)


def test_validate_features_invalid_age():
    """Test validation detects invalid age values."""
    from covid_prediction.training import TrainingPipeline
    from covid_prediction.model_io import save_model
    
    # Create minimal training data with both classes
    X_train = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1]
    ])
    y_train = np.array([1, 0])
    
    # Train a simple model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_train, y_train, algorithm='logistic_regression', balance_classes=False)
    
    # Save model
    metadata = {
        'algorithm': 'logistic_regression',
        'training_date': '2024-01-01',
        'dataset_size': 1,
        'feature_names': ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                         'age_60_and_above', 'gender', 'test_indication'],
        'feature_engineering_applied': False,
        'original_feature_count': 8,
        'engineered_feature_count': 8
    }
    model_path = save_model(model, metadata, output_dir='models')
    
    # Create prediction service
    service = PredictionService(model_path)
    
    # Test with invalid age
    invalid_features = {
        'cough': 1,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Maybe',  # Invalid: should be 'Yes' or 'No'
        'gender': 'male',
        'test_indication': 'Contact with confirmed'
    }
    
    with pytest.raises(ValueError) as exc_info:
        service.validate_features(invalid_features)
    
    assert "Invalid age value" in str(exc_info.value)


def test_validate_features_empty_test_indication():
    """Test validation detects empty test indication."""
    from covid_prediction.training import TrainingPipeline
    from covid_prediction.model_io import save_model
    
    # Create minimal training data with both classes
    X_train = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1]
    ])
    y_train = np.array([1, 0])
    
    # Train a simple model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_train, y_train, algorithm='logistic_regression', balance_classes=False)
    
    # Save model
    metadata = {
        'algorithm': 'logistic_regression',
        'training_date': '2024-01-01',
        'dataset_size': 1,
        'feature_names': ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                         'age_60_and_above', 'gender', 'test_indication'],
        'feature_engineering_applied': False,
        'original_feature_count': 8,
        'engineered_feature_count': 8
    }
    model_path = save_model(model, metadata, output_dir='models')
    
    # Create prediction service
    service = PredictionService(model_path)
    
    # Test with empty test indication
    invalid_features = {
        'cough': 1,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'male',
        'test_indication': ''  # Invalid: should not be empty
    }
    
    with pytest.raises(ValueError) as exc_info:
        service.validate_features(invalid_features)
    
    assert "Missing test indication" in str(exc_info.value)


def test_predict_valid_features():
    """Test prediction with valid features."""
    from covid_prediction.training import TrainingPipeline
    from covid_prediction.model_io import save_model
    
    # Create training data with multiple samples
    X_train = np.array([
        [1, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1]
    ])
    y_train = np.array([1, 0, 1, 0])
    
    # Train a simple model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_train, y_train, algorithm='logistic_regression', balance_classes=False)
    
    # Save model
    metadata = {
        'algorithm': 'logistic_regression',
        'training_date': '2024-01-01',
        'dataset_size': 4,
        'feature_names': ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                         'age_60_and_above', 'gender', 'test_indication'],
        'feature_engineering_applied': False,
        'original_feature_count': 8,
        'engineered_feature_count': 8
    }
    model_path = save_model(model, metadata, output_dir='models')
    
    # Create prediction service
    service = PredictionService(model_path)
    
    # Test with valid features
    valid_features = {
        'cough': 1,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'male',
        'test_indication': 'Contact with confirmed'
    }
    
    result = service.predict(valid_features)
    
    # Verify result structure
    assert isinstance(result, PredictionResult)
    assert result.predicted_class in ['positive', 'negative']
    assert 0.0 <= result.confidence <= 1.0
    assert result.timestamp is not None
