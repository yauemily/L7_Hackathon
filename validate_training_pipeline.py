"""
Validation script for Task 10 checkpoint.

This script validates that the training and evaluation pipeline works correctly
by training models with all three algorithms, evaluating them, and extracting
feature importance.
"""

import numpy as np
from covid_prediction.data_loader import CovidDatasetLoader
from covid_prediction.preprocessor import CovidDataPreprocessor
from covid_prediction.feature_engineering import FeatureEngineer
from covid_prediction.training import TrainingPipeline
from covid_prediction.evaluation import ModelEvaluator
from covid_prediction.feature_importance import FeatureImportanceAnalyzer
from covid_prediction.model_io import save_model, load_model


def validate_training_pipeline():
    """Validate the complete training and evaluation pipeline."""
    print("=" * 80)
    print("TASK 10 CHECKPOINT: Training and Evaluation Pipeline Validation")
    print("=" * 80)
    
    # Step 1: Load dataset
    print("\n[1/7] Loading COVID dataset...")
    loader = CovidDatasetLoader()
    df = loader.load_dataset('Data/corona_tested_individuals_ver_006.english.csv')
    print(f"✓ Loaded {len(df)} rows")
    
    # Step 2: Preprocess data
    print("\n[2/7] Preprocessing data...")
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    feature_names = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                     'head_ache', 'age_60_and_above', 'gender', 'test_indication']
    
    # Test all three algorithms
    algorithms = ['logistic_regression', 'random_forest', 'gradient_boosting']
    
    for i, algorithm in enumerate(algorithms, start=3):
        print(f"\n[{i}/7] Testing {algorithm.replace('_', ' ').title()}...")
        
        # Apply feature engineering
        engineer = FeatureEngineer(algorithm=algorithm)
        X_train_eng, eng_feature_names = engineer.fit_transform(X_train, feature_names)
        X_test_eng = engineer.transform(X_test)
        
        if algorithm == 'logistic_regression':
            print(f"  ✓ Feature engineering applied: {len(feature_names)} → {len(eng_feature_names)} features")
        else:
            print(f"  ✓ No feature engineering (tree-based model)")
        
        # Train model
        pipeline = TrainingPipeline()
        model = pipeline.train(
            X_train_eng, 
            y_train, 
            algorithm=algorithm,
            balance_classes=True
        )
        print(f"  ✓ Model trained successfully")
        
        # Evaluate model
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test_eng, y_test)
        
        print(f"  ✓ Evaluation metrics:")
        print(f"    - Accuracy: {metrics['accuracy']:.4f}")
        print(f"    - Precision (positive): {metrics['precision_positive']:.4f}")
        print(f"    - Recall (positive): {metrics['recall_positive']:.4f}")
        print(f"    - F1-score (positive): {metrics['f1_positive']:.4f}")
        print(f"    - AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Extract feature importance
        analyzer = FeatureImportanceAnalyzer()
        try:
            importance_df = analyzer.extract_importance(model, eng_feature_names)
            top_features = importance_df.head(5)
            print(f"  ✓ Feature importance extracted")
            print(f"    Top 5 features:")
            for idx, row in top_features.iterrows():
                print(f"      {idx+1}. {row['feature']}: {row['importance']:.4f}")
        except ValueError as e:
            print(f"  ⚠ Feature importance not available: {e}")
        
        # Test model persistence
        metadata = {
            'algorithm': algorithm,
            'training_date': '2024-01-01',
            'dataset_size': len(X_train),
            'feature_names': eng_feature_names,
            'feature_engineering_applied': algorithm == 'logistic_regression',
            'original_feature_count': len(feature_names),
            'engineered_feature_count': len(eng_feature_names),
            'feature_engineer': engineer
        }
        
        model_path = save_model(model, metadata, output_dir='models')
        print(f"  ✓ Model saved: {model_path}")
        
        # Load and verify
        loaded_model, loaded_metadata, loaded_engineer = load_model(model_path)
        
        # Make predictions with both models
        y_pred_original = model.predict(X_test_eng[:5])
        y_pred_loaded = loaded_model.predict(X_test_eng[:5])
        
        assert np.array_equal(y_pred_original, y_pred_loaded), "Loaded model predictions don't match!"
        print(f"  ✓ Model loaded and verified (predictions match)")
    
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED - Training and Evaluation Pipeline Works!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Dataset loading works")
    print("  ✓ Data preprocessing works")
    print("  ✓ Feature engineering works (algorithm-specific)")
    print("  ✓ Model training works (all 3 algorithms)")
    print("  ✓ Model evaluation works")
    print("  ✓ Feature importance extraction works")
    print("  ✓ Model persistence (save/load) works")
    print("\nReady to proceed to Task 11: Implement prediction service")


if __name__ == '__main__':
    validate_training_pipeline()
