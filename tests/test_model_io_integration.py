"""
Integration tests for model_io with training pipeline and feature engineering.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from covid_prediction.training import TrainingPipeline
from covid_prediction.feature_engineering import FeatureEngineer
from covid_prediction.model_io import save_model, load_model, verify_model_compatibility


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(50, 8))  # 50 samples, 8 features
    y = np.random.randint(0, 2, size=50)
    return X, y


def test_save_load_with_training_pipeline(sample_training_data, temp_dir):
    """Test saving and loading model trained with TrainingPipeline."""
    X, y = sample_training_data
    
    # Train model
    pipeline = TrainingPipeline()
    model = pipeline.train(X, y, algorithm='random_forest', balance_classes=False)
    
    # Save model using model_io
    filepath = save_model(model, pipeline.metadata, output_dir=temp_dir)
    
    # Load model
    loaded_model, loaded_metadata, loaded_fe = load_model(filepath)
    
    # Verify predictions match
    X_test = X[:5]
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    
    np.testing.assert_array_equal(original_pred, loaded_pred)


def test_save_load_with_feature_engineer(sample_training_data, temp_dir):
    """Test saving and loading model with feature engineering."""
    X, y = sample_training_data
    
    # Apply feature engineering
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    fe = FeatureEngineer(algorithm='logistic_regression')
    X_engineered, engineered_names = fe.fit_transform(X, feature_names)
    
    # Train model
    pipeline = TrainingPipeline()
    model = pipeline.train(X_engineered, y, algorithm='logistic_regression', 
                          balance_classes=False, feature_names=engineered_names)
    
    # Save model with feature engineer
    filepath = save_model(model, pipeline.metadata, output_dir=temp_dir, 
                         feature_engineer=fe)
    
    # Load model
    loaded_model, loaded_metadata, loaded_fe = load_model(filepath)
    
    # Verify feature engineer was saved
    assert loaded_fe is not None
    assert loaded_fe.algorithm == 'logistic_regression'
    
    # Verify predictions work with feature engineering
    X_test = X[:5]
    X_test_engineered = loaded_fe.transform(X_test)
    predictions = loaded_model.predict(X_test_engineered)
    
    assert len(predictions) == 5
    assert all(pred in [0, 1] for pred in predictions)


def test_verify_compatibility_with_training_metadata(sample_training_data):
    """Test verify_model_compatibility with real training metadata."""
    X, y = sample_training_data
    
    # Train model
    pipeline = TrainingPipeline()
    feature_names = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                    'head_ache', 'age_60_and_above', 'gender', 'test_indication']
    model = pipeline.train(X, y, algorithm='random_forest', 
                          balance_classes=False, feature_names=feature_names)
    
    # Verify compatibility with same features
    result = verify_model_compatibility(pipeline.metadata, feature_names)
    assert result is True


def test_verify_incompatibility_with_different_features(sample_training_data):
    """Test that incompatible features are detected."""
    X, y = sample_training_data
    
    # Train model with specific features
    pipeline = TrainingPipeline()
    feature_names = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                    'head_ache', 'age_60_and_above', 'gender', 'test_indication']
    model = pipeline.train(X, y, algorithm='random_forest', 
                          balance_classes=False, feature_names=feature_names)
    
    # Try to verify with different features
    different_features = ['cough', 'fever', 'headache', 'shortness_of_breath', 
                         'head_ache', 'age_60_and_above', 'gender', 'test_indication']
    
    with pytest.raises(ValueError) as exc_info:
        verify_model_compatibility(pipeline.metadata, different_features)
    
    assert 'mismatch' in str(exc_info.value).lower()


def test_model_filename_format(sample_training_data, temp_dir):
    """Test that saved model filename follows expected format."""
    X, y = sample_training_data
    
    # Train models with different algorithms
    algorithms = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    for algorithm in algorithms:
        pipeline = TrainingPipeline()
        model = pipeline.train(X, y, algorithm=algorithm, balance_classes=False)
        filepath = save_model(model, pipeline.metadata, output_dir=temp_dir)
        
        filename = os.path.basename(filepath)
        assert filename.startswith(algorithm)
        assert filename.endswith('.joblib')
        
        # Verify timestamp format (YYYYMMDD_HHMMSS)
        parts = filename.replace('.joblib', '').split('_')
        assert len(parts) >= 3
