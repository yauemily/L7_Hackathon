"""
Unit tests for model persistence functions.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from covid_prediction.model_io import save_model, load_model, verify_model_compatibility


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'algorithm': 'random_forest',
        'training_date': '2024-03-04T10:00:00',
        'dataset_size': 100,
        'feature_names': ['cough', 'fever'],
        'feature_engineering_applied': False,
        'class_balance_method': 'class_weights'
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path)


def test_save_model_creates_file(sample_model, sample_metadata, temp_dir):
    """Test that save_model creates a file with timestamp."""
    filepath = save_model(sample_model, sample_metadata, output_dir=temp_dir)
    
    assert os.path.exists(filepath)
    assert filepath.startswith(os.path.join(temp_dir, 'random_forest_'))
    assert filepath.endswith('.joblib')


def test_save_model_includes_timestamp(sample_model, sample_metadata, temp_dir):
    """Test that filename includes timestamp."""
    filepath = save_model(sample_model, sample_metadata, output_dir=temp_dir)
    filename = os.path.basename(filepath)
    
    # Filename should be: algorithm_YYYYMMDD_HHMMSS.joblib
    parts = filename.replace('.joblib', '').split('_')
    assert len(parts) >= 3  # algorithm, date, time
    
    # Check date part is 8 digits
    date_part = parts[-2]
    assert len(date_part) == 8
    assert date_part.isdigit()
    
    # Check time part is 6 digits
    time_part = parts[-1]
    assert len(time_part) == 6
    assert time_part.isdigit()


def test_save_and_load_model_roundtrip(sample_model, sample_metadata, temp_dir):
    """Test that saving and loading preserves model functionality."""
    # Save model
    filepath = save_model(sample_model, sample_metadata, output_dir=temp_dir)
    
    # Load model
    loaded_model, loaded_metadata, loaded_fe = load_model(filepath)
    
    # Verify model makes same predictions
    X_test = np.array([[0, 1], [1, 0]])
    original_pred = sample_model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    
    np.testing.assert_array_equal(original_pred, loaded_pred)


def test_load_model_returns_metadata(sample_model, sample_metadata, temp_dir):
    """Test that load_model returns correct metadata."""
    filepath = save_model(sample_model, sample_metadata, output_dir=temp_dir)
    loaded_model, loaded_metadata, loaded_fe = load_model(filepath)
    
    assert loaded_metadata['algorithm'] == 'random_forest'
    assert loaded_metadata['dataset_size'] == 100
    assert loaded_metadata['feature_names'] == ['cough', 'fever']


def test_load_nonexistent_file_raises_error():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_model('nonexistent_model.joblib')
    
    assert 'FileNotFoundError' in str(exc_info.value)
    assert 'not found' in str(exc_info.value)


def test_load_corrupted_file_raises_error(temp_dir):
    """Test that loading corrupted file raises ValueError."""
    # Create a corrupted file
    corrupted_path = os.path.join(temp_dir, 'corrupted.joblib')
    with open(corrupted_path, 'w') as f:
        f.write('This is not a valid joblib file')
    
    with pytest.raises(ValueError) as exc_info:
        load_model(corrupted_path)
    
    assert 'ValueError' in str(exc_info.value)
    assert 'corrupted' in str(exc_info.value).lower()


def test_verify_model_compatibility_success(sample_metadata):
    """Test that compatible feature schemas pass verification."""
    expected_features = ['cough', 'fever']
    result = verify_model_compatibility(sample_metadata, expected_features)
    assert result is True


def test_verify_model_compatibility_different_count(sample_metadata):
    """Test that different feature counts raise ValueError."""
    expected_features = ['cough', 'fever', 'sore_throat']
    
    with pytest.raises(ValueError) as exc_info:
        verify_model_compatibility(sample_metadata, expected_features)
    
    assert 'ValueError' in str(exc_info.value)
    assert 'mismatch' in str(exc_info.value).lower()
    assert '2 features' in str(exc_info.value)
    assert '3 features' in str(exc_info.value)


def test_verify_model_compatibility_different_names(sample_metadata):
    """Test that different feature names raise ValueError."""
    expected_features = ['cough', 'headache']  # Different from 'fever'
    
    with pytest.raises(ValueError) as exc_info:
        verify_model_compatibility(sample_metadata, expected_features)
    
    assert 'ValueError' in str(exc_info.value)
    assert 'mismatch' in str(exc_info.value).lower()


def test_save_model_with_none_raises_error(sample_metadata, temp_dir):
    """Test that saving None model raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        save_model(None, sample_metadata, output_dir=temp_dir)
    
    assert 'ValueError' in str(exc_info.value)
    assert 'None model' in str(exc_info.value)


def test_save_model_with_empty_metadata_raises_error(sample_model, temp_dir):
    """Test that saving with empty metadata raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        save_model(sample_model, {}, output_dir=temp_dir)
    
    assert 'ValueError' in str(exc_info.value)
    assert 'empty' in str(exc_info.value).lower()


def test_save_model_with_missing_required_fields(sample_model, temp_dir):
    """Test that saving with missing required metadata fields raises ValueError."""
    incomplete_metadata = {
        'algorithm': 'random_forest',
        # Missing: training_date, dataset_size, feature_names
    }
    
    with pytest.raises(ValueError) as exc_info:
        save_model(sample_model, incomplete_metadata, output_dir=temp_dir)
    
    assert 'ValueError' in str(exc_info.value)
    assert 'Missing required metadata fields' in str(exc_info.value)
