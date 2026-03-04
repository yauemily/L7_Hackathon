"""
Unit tests for CovidDataPreprocessor.
"""

import pytest
import pandas as pd
import numpy as np
from covid_prediction.preprocessor import CovidDataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample COVID test data for testing."""
    return pd.DataFrame({
        'test_date': ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', 
                      '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', '2020-03-10'],
        'cough': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'fever': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        'sore_throat': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        'shortness_of_breath': [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'head_ache': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        'age_60_and_above': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'test_indication': ['Contact with confirmed', 'Abroad', 'Contact with confirmed', 'Other', 'Abroad',
                           'Contact with confirmed', 'Abroad', 'Other', 'Contact with confirmed', 'Abroad'],
        'corona_result': ['positive', 'negative', 'positive', 'negative', 'positive',
                         'negative', 'positive', 'negative', 'positive', 'negative']
    })


@pytest.fixture
def data_with_missing():
    """Create sample data with missing values."""
    return pd.DataFrame({
        'test_date': ['2020-03-01', '2020-03-02', '2020-03-03'],
        'cough': [1, 0, 1],
        'fever': [1, 1, 0],
        'sore_throat': [0, 1, 1],
        'shortness_of_breath': [0, 0, 1],
        'head_ache': [1, 0, 0],
        'age_60_and_above': ['Yes', None, 'No'],
        'gender': ['male', 'female', 'male'],
        'test_indication': ['Contact with confirmed', 'Abroad', 'Other'],
        'corona_result': ['positive', 'negative', 'positive']
    })


def test_handle_missing_values_drop(data_with_missing):
    """Test handling missing values with drop strategy."""
    preprocessor = CovidDataPreprocessor()
    df_clean = preprocessor.handle_missing_values(data_with_missing, strategy='drop')
    
    # Should have 2 rows (dropped 1 with missing value)
    assert len(df_clean) == 2
    assert df_clean['age_60_and_above'].isna().sum() == 0


def test_handle_missing_values_impute(data_with_missing):
    """Test handling missing values with impute strategy."""
    preprocessor = CovidDataPreprocessor()
    df_clean = preprocessor.handle_missing_values(data_with_missing, strategy='impute')
    
    # Should still have 3 rows
    assert len(df_clean) == 3
    assert df_clean['age_60_and_above'].isna().sum() == 0


def test_encode_categorical(sample_data):
    """Test categorical encoding produces numeric types."""
    preprocessor = CovidDataPreprocessor()
    df_encoded = preprocessor.encode_categorical(sample_data, fit=True)
    
    # Check gender encoding
    assert df_encoded['gender'].dtype in [np.int64, np.int32, int]
    assert set(df_encoded['gender'].unique()).issubset({0, 1})
    
    # Check age_60_and_above encoding
    assert df_encoded['age_60_and_above'].dtype in [np.int64, np.int32, int]
    assert set(df_encoded['age_60_and_above'].unique()).issubset({0, 1})
    
    # Check test_indication encoding
    assert df_encoded['test_indication'].dtype in [np.int64, np.int32, int]
    
    # Check corona_result encoding
    assert df_encoded['corona_result'].dtype in [np.int64, np.int32, int]
    assert set(df_encoded['corona_result'].unique()).issubset({0, 1})


def test_preprocess_returns_correct_shapes(sample_data):
    """Test preprocess returns feature matrix and target vector."""
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(sample_data)
    
    # Check shapes
    assert X.shape[0] == len(sample_data)  # Same number of rows
    assert X.shape[1] == 8  # 8 features
    assert y.shape[0] == len(sample_data)  # Same number of rows
    assert len(y.shape) == 1  # 1D array


def test_symptom_features_are_binary(sample_data):
    """Test that symptom features are binary (0/1)."""
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(sample_data)
    
    # First 5 columns are symptoms
    symptom_features = X[:, :5]
    
    # Check all values are 0 or 1
    assert np.all(np.isin(symptom_features, [0, 1]))


def test_split_data_preserves_size(sample_data):
    """Test train-test split preserves data size."""
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(sample_data)
    
    test_size = 0.2
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=test_size)
    
    # Check total size is preserved
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    
    # Check test size is approximately correct (within 1 sample for small datasets)
    expected_test_size = int(len(X) * test_size)
    assert abs(len(X_test) - expected_test_size) <= 1


def test_split_data_with_custom_ratio(sample_data):
    """Test train-test split with custom ratio."""
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(sample_data)
    
    test_size = 0.4
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=test_size)
    
    # Check test size is approximately correct
    expected_test_size = int(len(X) * test_size)
    assert abs(len(X_test) - expected_test_size) <= 1
