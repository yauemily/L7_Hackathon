"""
Unit tests for feature importance analyzer.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from covid_prediction.feature_importance import FeatureImportanceAnalyzer
import os
import tempfile


class TestFeatureImportanceAnalyzer:
    """Test suite for FeatureImportanceAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        return X, y, feature_names
    
    @pytest.fixture
    def analyzer(self):
        """Create FeatureImportanceAnalyzer instance."""
        return FeatureImportanceAnalyzer()
    
    def test_extract_importance_from_random_forest(self, analyzer, sample_data):
        """Test extraction from random forest model (feature_importances_)."""
        X, y, feature_names = sample_data
        
        # Train a random forest model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Verify structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ['feature', 'importance']
        assert len(importance_df) == len(feature_names)
        
        # Verify all features are present
        assert set(importance_df['feature']) == set(feature_names)
        
        # Verify sorted in descending order
        assert importance_df['importance'].is_monotonic_decreasing
        
        # Verify importance scores are non-negative
        assert (importance_df['importance'] >= 0).all()
    
    def test_extract_importance_from_logistic_regression(self, analyzer, sample_data):
        """Test extraction from logistic regression model (coef_)."""
        X, y, feature_names = sample_data
        
        # Train a logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Verify structure
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ['feature', 'importance']
        assert len(importance_df) == len(feature_names)
        
        # Verify all features are present
        assert set(importance_df['feature']) == set(feature_names)
        
        # Verify sorted in descending order
        assert importance_df['importance'].is_monotonic_decreasing
        
        # Verify importance scores are non-negative (absolute values of coefficients)
        assert (importance_df['importance'] >= 0).all()
    
    def test_extract_importance_unsupported_model(self, analyzer, sample_data):
        """Test graceful handling of models that don't support feature importance."""
        _, _, feature_names = sample_data
        
        # Create a mock model without feature_importances_ or coef_
        class UnsupportedModel:
            pass
        
        model = UnsupportedModel()
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            analyzer.extract_importance(model, feature_names)
        
        assert "does not support feature importance" in str(exc_info.value)
    
    def test_extract_importance_feature_count_mismatch(self, analyzer, sample_data):
        """Test error handling when feature count doesn't match."""
        X, y, feature_names = sample_data
        
        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Provide wrong number of feature names
        wrong_feature_names = ['feature_1', 'feature_2']
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            analyzer.extract_importance(model, wrong_feature_names)
        
        assert "Feature count mismatch" in str(exc_info.value)
    
    def test_visualize_importance(self, analyzer, sample_data):
        """Test visualization creation."""
        X, y, feature_names = sample_data
        
        # Train a model and extract importance
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Create visualization without saving
        analyzer.visualize_importance(importance_df)
        
        # Should complete without error
        assert True
    
    def test_visualize_importance_with_save(self, analyzer, sample_data):
        """Test visualization saving to file."""
        X, y, feature_names = sample_data
        
        # Train a model and extract importance
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create and save visualization
            analyzer.visualize_importance(importance_df, save_path=tmp_path)
            
            # Verify file was created
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_visualize_importance_top_n(self, analyzer, sample_data):
        """Test visualization with top_n parameter."""
        X, y, feature_names = sample_data
        
        # Train a model and extract importance
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Create visualization with top 3 features
        analyzer.visualize_importance(importance_df, top_n=3)
        
        # Should complete without error
        assert True
    
    def test_save_report(self, analyzer, sample_data):
        """Test saving feature importance report to CSV."""
        X, y, feature_names = sample_data
        
        # Train a model and extract importance
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save report
            analyzer.save_report(importance_df, tmp_path)
            
            # Verify file was created
            assert os.path.exists(tmp_path)
            
            # Load and verify content
            loaded_df = pd.read_csv(tmp_path)
            assert list(loaded_df.columns) == ['feature', 'importance']
            assert len(loaded_df) == len(feature_names)
            
            # Verify data matches
            pd.testing.assert_frame_equal(loaded_df, importance_df)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_sorting_by_importance(self, analyzer, sample_data):
        """Test that features are correctly sorted by importance."""
        X, y, feature_names = sample_data
        
        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Verify first feature has highest importance
        max_importance = importance_df['importance'].max()
        assert importance_df.iloc[0]['importance'] == max_importance
        
        # Verify last feature has lowest importance
        min_importance = importance_df['importance'].min()
        assert importance_df.iloc[-1]['importance'] == min_importance
