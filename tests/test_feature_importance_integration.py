"""
Integration tests for feature importance analyzer with real models.
"""

import pytest
import numpy as np
import os
import tempfile
from covid_prediction.feature_importance import FeatureImportanceAnalyzer
from covid_prediction.model_io import load_model


class TestFeatureImportanceIntegration:
    """Integration tests for feature importance with saved models."""
    
    @pytest.fixture
    def analyzer(self):
        """Create FeatureImportanceAnalyzer instance."""
        return FeatureImportanceAnalyzer()
    
    def test_feature_importance_with_saved_random_forest(self, analyzer):
        """Test feature importance extraction from saved random forest model."""
        model_path = 'models/random_forest_20260304_201857.joblib'
        
        # Skip if model doesn't exist
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")
        
        # Load the model
        model, metadata, _ = load_model(model_path)
        
        # Get feature names from metadata
        feature_names = metadata['feature_names']
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Verify structure
        assert len(importance_df) == len(feature_names)
        assert importance_df['importance'].is_monotonic_decreasing
        assert (importance_df['importance'] >= 0).all()
        
        # Print top 5 features for inspection
        print("\nTop 5 most important features (Random Forest):")
        print(importance_df.head())
    
    def test_feature_importance_with_saved_logistic_regression(self, analyzer):
        """Test feature importance extraction from saved logistic regression model."""
        model_path = 'models/logistic_regression_20260304_201857.joblib'
        
        # Skip if model doesn't exist
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")
        
        # Load the model
        model, metadata, _ = load_model(model_path)
        
        # Get feature names from metadata
        feature_names = metadata['feature_names']
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Verify structure
        assert len(importance_df) == len(feature_names)
        assert importance_df['importance'].is_monotonic_decreasing
        assert (importance_df['importance'] >= 0).all()
        
        # Print top 5 features for inspection
        print("\nTop 5 most important features (Logistic Regression):")
        print(importance_df.head())
    
    def test_feature_importance_with_saved_gradient_boosting(self, analyzer):
        """Test feature importance extraction from saved gradient boosting model."""
        model_path = 'models/gradient_boosting_20260304_201857.joblib'
        
        # Skip if model doesn't exist
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")
        
        # Load the model
        model, metadata, _ = load_model(model_path)
        
        # Get feature names from metadata
        feature_names = metadata['feature_names']
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Verify structure
        assert len(importance_df) == len(feature_names)
        assert importance_df['importance'].is_monotonic_decreasing
        assert (importance_df['importance'] >= 0).all()
        
        # Print top 5 features for inspection
        print("\nTop 5 most important features (Gradient Boosting):")
        print(importance_df.head())
    
    def test_full_workflow_with_visualization_and_report(self, analyzer):
        """Test complete workflow: extract, visualize, and save report."""
        model_path = 'models/random_forest_20260304_201857.joblib'
        
        # Skip if model doesn't exist
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")
        
        # Load the model
        model, metadata, _ = load_model(model_path)
        feature_names = metadata['feature_names']
        
        # Extract importance
        importance_df = analyzer.extract_importance(model, feature_names)
        
        # Create temporary files for outputs
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            img_path = tmp_img.name
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv:
            csv_path = tmp_csv.name
        
        try:
            # Create visualization
            analyzer.visualize_importance(importance_df, save_path=img_path, top_n=10)
            assert os.path.exists(img_path)
            assert os.path.getsize(img_path) > 0
            
            # Save report
            analyzer.save_report(importance_df, csv_path)
            assert os.path.exists(csv_path)
            assert os.path.getsize(csv_path) > 0
            
            print(f"\nVisualization saved to: {img_path}")
            print(f"Report saved to: {csv_path}")
        finally:
            # Clean up
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)
