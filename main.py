#!/usr/bin/env python3
"""
COVID-19 Prediction Model - End-to-End Integration Script

This script demonstrates the complete workflow from data loading to prediction,
testing all three algorithms (logistic regression, random forest, gradient boosting).

Workflow:
1. Load dataset using CovidDatasetLoader
2. Preprocess data using CovidDataPreprocessor
3. Apply feature engineering using FeatureEngineer
4. Train models using TrainingPipeline (all three algorithms)
5. Evaluate models using ModelEvaluator
6. Generate feature importance using FeatureImportanceAnalyzer
7. Save models using model_io functions
8. Load model and make sample predictions using PredictionService
9. Print evaluation metrics and feature importance
"""

import os
import sys
from datetime import datetime

# Import all required modules
from covid_prediction.data_loader import CovidDatasetLoader
from covid_prediction.preprocessor import CovidDataPreprocessor
from covid_prediction.feature_engineering import FeatureEngineer
from covid_prediction.training import TrainingPipeline
from covid_prediction.evaluation import ModelEvaluator
from covid_prediction.feature_importance import FeatureImportanceAnalyzer
from covid_prediction.model_io import save_model
from covid_prediction.prediction import PredictionService


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_metrics(metrics: dict, algorithm: str):
    """Print evaluation metrics in a formatted way."""
    print(f"\n{algorithm.upper()} - Evaluation Metrics:")
    print("-" * 60)
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:               {metrics['auc_roc']:.4f}")
    print(f"\n  Positive Class (COVID+):")
    print(f"    Precision:           {metrics['precision_positive']:.4f}")
    print(f"    Recall:              {metrics['recall_positive']:.4f}")
    print(f"    F1-Score:            {metrics['f1_positive']:.4f}")
    print(f"\n  Negative Class (COVID-):")
    print(f"    Precision:           {metrics['precision_negative']:.4f}")
    print(f"    Recall:              {metrics['recall_negative']:.4f}")
    print(f"    F1-Score:            {metrics['f1_negative']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    [[TN={cm[0][0]:5d}, FP={cm[0][1]:5d}]")
    print(f"     [FN={cm[1][0]:5d}, TP={cm[1][1]:5d}]]")
    print(f"\n  Class Distribution (Test Set):")
    for cls, count in metrics['class_distribution'].items():
        print(f"    {cls}: {count}")


def main():
    """Main execution function."""
    print_section_header("COVID-19 PREDICTION MODEL - END-TO-END WORKFLOW")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    dataset_path = "Data/corona_tested_individuals_ver_006.english.csv"
    models_dir = "models"
    reports_dir = "reports"
    
    # Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print_section_header("STEP 1: Loading Dataset")
    
    loader = CovidDatasetLoader()
    print(f"Loading dataset from: {dataset_path}")
    df = loader.load_dataset(dataset_path)
    print(f"✓ Dataset loaded successfully")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # =========================================================================
    # STEP 2: Preprocess Data
    # =========================================================================
    print_section_header("STEP 2: Preprocessing Data")
    
    preprocessor = CovidDataPreprocessor()
    print("Preprocessing data (handling missing values, encoding categorical variables)...")
    X, y = preprocessor.preprocess(df, missing_value_strategy='drop')
    print(f"✓ Preprocessing complete")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape: {y.shape}")
    print(f"  Class distribution: Negative={sum(y==0)}, Positive={sum(y==1)}")
    
    # Split data into train and test sets
    print("\nSplitting data into train/test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2, random_state=42)
    print(f"✓ Data split complete")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Define feature names
    feature_names = [
        'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
        'age_60_and_above', 'gender', 'test_indication'
    ]
    
    # =========================================================================
    # STEP 3-7: Train and Evaluate All Three Algorithms
    # =========================================================================
    algorithms = ['logistic_regression', 'random_forest', 'gradient_boosting']
    trained_models = {}
    
    for algorithm in algorithms:
        print_section_header(f"TRAINING AND EVALUATING: {algorithm.upper()}")
        
        # Step 3: Apply Feature Engineering
        print(f"\nApplying feature engineering for {algorithm}...")
        feature_engineer = FeatureEngineer(algorithm=algorithm)
        X_train_engineered, engineered_feature_names = feature_engineer.fit_transform(
            X_train, feature_names
        )
        X_test_engineered = feature_engineer.transform(X_test)
        
        if algorithm == 'logistic_regression':
            print(f"✓ Feature engineering applied")
            print(f"  Original features: {len(feature_names)}")
            print(f"  Engineered features: {len(engineered_feature_names)}")
        else:
            print(f"✓ No feature engineering (tree-based model uses raw features)")
            print(f"  Features: {len(feature_names)}")
        
        # Step 4: Train Model
        print(f"\nTraining {algorithm} model...")
        pipeline = TrainingPipeline()
        model = pipeline.train(
            X_train_engineered,
            y_train,
            algorithm=algorithm,
            balance_classes=True,
            feature_names=engineered_feature_names
        )
        print(f"✓ Model trained successfully")
        print(f"  Algorithm: {pipeline.metadata['algorithm']}")
        print(f"  Dataset size: {pipeline.metadata['dataset_size']}")
        print(f"  Class balance method: {pipeline.metadata.get('class_balance_method', 'None')}")
        
        # Step 5: Evaluate Model
        print(f"\nEvaluating {algorithm} model on test set...")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test_engineered, y_test)
        print_metrics(metrics, algorithm)
        
        # Step 6: Generate Feature Importance
        print(f"\nGenerating feature importance for {algorithm}...")
        analyzer = FeatureImportanceAnalyzer()
        try:
            importance_df = analyzer.extract_importance(model, engineered_feature_names)
            print(f"✓ Feature importance extracted")
            print(f"\n  Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"    {idx+1}. {row['feature']:<30} {row['importance']:.6f}")
            
            # Save feature importance report
            importance_report_path = os.path.join(
                reports_dir, 
                f"feature_importance_{algorithm}.csv"
            )
            analyzer.save_report(importance_df, importance_report_path)
            print(f"  ✓ Feature importance report saved to: {importance_report_path}")
            
            # Save feature importance visualization
            importance_viz_path = os.path.join(
                reports_dir,
                f"feature_importance_{algorithm}.png"
            )
            analyzer.visualize_importance(importance_df, save_path=importance_viz_path, top_n=15)
            print(f"  ✓ Feature importance visualization saved to: {importance_viz_path}")
            
        except ValueError as e:
            print(f"  ⚠ Feature importance not available: {e}")
        
        # Step 7: Save Model
        print(f"\nSaving {algorithm} model...")
        # Add feature engineering metadata
        pipeline.metadata['feature_engineering_applied'] = (algorithm == 'logistic_regression')
        pipeline.metadata['original_feature_count'] = len(feature_names)
        pipeline.metadata['engineered_feature_count'] = len(engineered_feature_names)
        pipeline.metadata['test_indication_encoder'] = preprocessor.test_indication_encoder
        
        model_path = save_model(
            model=model,
            metadata=pipeline.metadata,
            output_dir=models_dir,
            feature_engineer=feature_engineer
        )
        print(f"✓ Model saved to: {model_path}")
        
        # Store for later use
        trained_models[algorithm] = {
            'model_path': model_path,
            'metrics': metrics,
            'feature_engineer': feature_engineer
        }
    
    # =========================================================================
    # STEP 8: Load Model and Make Sample Predictions
    # =========================================================================
    print_section_header("STEP 8: Making Sample Predictions")
    
    # Use the best performing model (let's use random forest as default)
    best_algorithm = 'random_forest'
    best_model_path = trained_models[best_algorithm]['model_path']
    
    print(f"\nLoading {best_algorithm} model for predictions...")
    prediction_service = PredictionService(best_model_path)
    print(f"✓ Model loaded successfully")
    
    # Define sample test cases
    sample_cases = [
        {
            'name': 'High Risk Patient',
            'features': {
                'cough': 1,
                'fever': 1,
                'sore_throat': 1,
                'shortness_of_breath': 1,
                'head_ache': 1,
                'age_60_and_above': 'Yes',
                'gender': 'male',
                'test_indication': 'Contact with confirmed'
            }
        },
        {
            'name': 'Low Risk Patient',
            'features': {
                'cough': 0,
                'fever': 0,
                'sore_throat': 0,
                'shortness_of_breath': 0,
                'head_ache': 0,
                'age_60_and_above': 'No',
                'gender': 'female',
                'test_indication': 'Abroad'
            }
        },
        {
            'name': 'Moderate Risk Patient',
            'features': {
                'cough': 1,
                'fever': 1,
                'sore_throat': 0,
                'shortness_of_breath': 0,
                'head_ache': 0,
                'age_60_and_above': 'No',
                'gender': 'male',
                'test_indication': 'Other'
            }
        }
    ]
    
    print("\nMaking predictions on sample cases:")
    print("-" * 60)
    
    for case in sample_cases:
        print(f"\n{case['name']}:")
        features = case['features']
        
        # Display input features
        print("  Input Features:")
        for key, value in features.items():
            print(f"    {key}: {value}")
        
        # Make prediction
        result = prediction_service.predict(features)
        
        # Display prediction result
        print(f"\n  Prediction Result:")
        print(f"    Predicted Class: {result.predicted_class.upper()}")
        print(f"    Confidence:      {result.confidence:.4f} ({result.confidence*100:.2f}%)")
        print(f"    Timestamp:       {result.timestamp}")
    
    # =========================================================================
    # STEP 9: Summary and Comparison
    # =========================================================================
    print_section_header("STEP 9: Model Comparison Summary")
    
    print("\nPerformance Comparison Across All Algorithms:")
    print("-" * 80)
    print(f"{'Algorithm':<25} {'Accuracy':<12} {'AUC-ROC':<12} {'F1 (Pos)':<12} {'F1 (Neg)':<12}")
    print("-" * 80)
    
    for algorithm in algorithms:
        metrics = trained_models[algorithm]['metrics']
        print(f"{algorithm:<25} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['auc_roc']:<12.4f} "
              f"{metrics['f1_positive']:<12.4f} "
              f"{metrics['f1_negative']:<12.4f}")
    
    # Find best model by AUC-ROC
    best_auc_algorithm = max(algorithms, key=lambda a: trained_models[a]['metrics']['auc_roc'])
    best_auc = trained_models[best_auc_algorithm]['metrics']['auc_roc']
    
    print("\n" + "=" * 80)
    print(f"Best Model: {best_auc_algorithm.upper()} (AUC-ROC: {best_auc:.4f})")
    print("=" * 80)
    
    print(f"\nAll models saved in: {models_dir}/")
    print(f"All reports saved in: {reports_dir}/")
    
    print_section_header("WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("  1. Review evaluation metrics and feature importance reports")
    print("  2. Use the best performing model for production predictions")
    print("  3. Monitor model performance on new data")
    print("  4. Retrain periodically with updated data")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
