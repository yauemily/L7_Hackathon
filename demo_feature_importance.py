"""
Demonstration script for feature importance analysis.

This script shows how to use the FeatureImportanceAnalyzer to:
1. Extract feature importance from trained models
2. Visualize the most important features
3. Save feature importance reports
"""

import os
from covid_prediction.feature_importance import FeatureImportanceAnalyzer
from covid_prediction.model_io import load_model


def analyze_model_importance(model_path: str, output_prefix: str):
    """
    Analyze and visualize feature importance for a trained model.
    
    Args:
        model_path: Path to saved model file
        output_prefix: Prefix for output files (e.g., 'random_forest')
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_path}")
    print(f"{'='*70}")
    
    # Load the model
    model, metadata, _ = load_model(model_path)
    
    # Get feature names from metadata
    feature_names = metadata['feature_names']
    algorithm = metadata['algorithm']
    
    print(f"Algorithm: {algorithm}")
    print(f"Number of features: {len(feature_names)}")
    
    # Create analyzer
    analyzer = FeatureImportanceAnalyzer()
    
    # Extract importance
    importance_df = analyzer.extract_importance(model, feature_names)
    
    # Display top 10 features
    print(f"\nTop 10 Most Important Features:")
    print("-" * 70)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    # Create visualization
    viz_path = f"models/{output_prefix}_feature_importance.png"
    analyzer.visualize_importance(importance_df, save_path=viz_path, top_n=15)
    print(f"\nVisualization saved to: {viz_path}")
    
    # Save report
    report_path = f"models/{output_prefix}_feature_importance.csv"
    analyzer.save_report(importance_df, report_path)
    print(f"Report saved to: {report_path}")


def main():
    """Main function to analyze all trained models."""
    print("\n" + "="*70)
    print("Feature Importance Analysis for COVID-19 Prediction Models")
    print("="*70)
    
    # List of models to analyze
    models = [
        ('models/random_forest_20260304_201857.joblib', 'random_forest'),
        ('models/logistic_regression_20260304_201857.joblib', 'logistic_regression'),
        ('models/gradient_boosting_20260304_201857.joblib', 'gradient_boosting')
    ]
    
    # Analyze each model
    for model_path, output_prefix in models:
        if os.path.exists(model_path):
            try:
                analyze_model_importance(model_path, output_prefix)
            except Exception as e:
                print(f"\nError analyzing {model_path}: {e}")
        else:
            print(f"\nSkipping {model_path} - file not found")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
