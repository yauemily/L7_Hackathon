"""
End-to-end integration test for the data pipeline.

This test validates that the complete data pipeline (loading, preprocessing, 
feature engineering) works correctly with the actual COVID dataset.
"""

import pytest
import numpy as np
from covid_prediction.data_loader import CovidDatasetLoader
from covid_prediction.preprocessor import CovidDataPreprocessor
from covid_prediction.feature_engineering import FeatureEngineer


def test_data_pipeline_end_to_end():
    """Test complete data pipeline from loading to feature engineering."""
    # Step 1: Load the actual COVID dataset
    loader = CovidDatasetLoader()
    df = loader.load_dataset('Data/corona_tested_individuals_ver_006.english.csv')
    
    # Verify dataset loaded successfully
    assert df is not None
    assert len(df) > 0
    print(f"✓ Loaded {len(df)} rows from dataset")
    
    # Step 2: Preprocess the data
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(df)
    
    # Verify preprocessing worked
    assert X is not None
    assert y is not None
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 8  # 8 features
    print(f"✓ Preprocessed data: X shape {X.shape}, y shape {y.shape}")
    
    # Verify symptom features are binary
    symptom_features = X[:, :5]
    assert np.all(np.isin(symptom_features, [0, 1]))
    print(f"✓ Symptom features are binary")
    
    # Verify target is binary
    assert np.all(np.isin(y, [0, 1]))
    print(f"✓ Target variable is binary")
    
    # Step 3: Split the data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    
    # Verify split worked
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    print(f"✓ Split data: train={len(X_train)}, test={len(X_test)}")
    
    # Step 4: Test feature engineering for logistic regression
    feature_names = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                     'head_ache', 'age_60_and_above', 'gender', 'test_indication']
    
    engineer_lr = FeatureEngineer(algorithm='logistic_regression')
    X_train_eng, eng_feature_names = engineer_lr.fit_transform(X_train, feature_names)
    
    # Verify feature engineering worked
    assert X_train_eng.shape[0] == X_train.shape[0]  # Same number of samples
    assert X_train_eng.shape[1] > X_train.shape[1]  # More features after engineering
    print(f"✓ Feature engineering (LR): {X_train.shape[1]} → {X_train_eng.shape[1]} features")
    
    # Test transform on test data
    X_test_eng = engineer_lr.transform(X_test)
    assert X_test_eng.shape[0] == X_test.shape[0]
    assert X_test_eng.shape[1] == X_train_eng.shape[1]  # Same number of features
    print(f"✓ Transform test data: shape {X_test_eng.shape}")
    
    # Step 5: Test feature engineering for random forest (should not apply engineering)
    engineer_rf = FeatureEngineer(algorithm='random_forest')
    X_train_rf, rf_feature_names = engineer_rf.fit_transform(X_train, feature_names)
    
    # Verify no engineering applied
    assert X_train_rf.shape == X_train.shape  # Same shape
    assert np.array_equal(X_train_rf, X_train)  # Same values
    print(f"✓ Feature engineering (RF): no engineering applied, shape {X_train_rf.shape}")
    
    # Step 6: Verify feature scaling for logistic regression
    # Check that scaled features have approximately zero mean and unit variance
    feature_means = np.mean(X_train_eng, axis=0)
    feature_stds = np.std(X_train_eng, axis=0)
    
    # All features should have mean close to 0 and std close to 1
    assert np.allclose(feature_means, 0, atol=1e-7)
    assert np.allclose(feature_stds, 1, atol=0.1)
    print(f"✓ Feature scaling verified: mean≈0, std≈1")
    
    print("\n✅ Data pipeline end-to-end test PASSED!")


def test_data_pipeline_with_gradient_boosting():
    """Test data pipeline with gradient boosting (no feature engineering)."""
    # Load and preprocess
    loader = CovidDatasetLoader()
    df = loader.load_dataset('Data/corona_tested_individuals_ver_006.english.csv')
    
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    
    # Test feature engineering for gradient boosting
    feature_names = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                     'head_ache', 'age_60_and_above', 'gender', 'test_indication']
    
    engineer_gb = FeatureEngineer(algorithm='gradient_boosting')
    X_train_gb, gb_feature_names = engineer_gb.fit_transform(X_train, feature_names)
    
    # Verify no engineering applied
    assert X_train_gb.shape == X_train.shape
    assert np.array_equal(X_train_gb, X_train)
    assert gb_feature_names == feature_names
    
    # Test transform
    X_test_gb = engineer_gb.transform(X_test)
    assert X_test_gb.shape == X_test.shape
    assert np.array_equal(X_test_gb, X_test)
    
    print("✅ Gradient boosting pipeline test PASSED!")


def test_feature_engineer_transform_before_fit_raises_error():
    """Test that calling transform before fit_transform raises an error."""
    engineer = FeatureEngineer(algorithm='logistic_regression')
    X_dummy = np.array([[1, 0, 1, 0, 1, 0, 0, 1]])
    
    with pytest.raises(RuntimeError, match="Transform called before fit_transform"):
        engineer.transform(X_dummy)
    
    print("✅ Transform before fit error handling test PASSED!")


if __name__ == '__main__':
    # Run tests
    test_data_pipeline_end_to_end()
    test_data_pipeline_with_gradient_boosting()
    test_feature_engineer_transform_before_fit_raises_error()
