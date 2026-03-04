"""
Demo script for COVID-19 prediction service.

This script demonstrates how to use the PredictionService to make predictions
on new data after training a model.
"""

import numpy as np
from covid_prediction.data_loader import CovidDatasetLoader
from covid_prediction.preprocessor import CovidDataPreprocessor
from covid_prediction.feature_engineering import FeatureEngineer
from covid_prediction.training import TrainingPipeline
from covid_prediction.model_io import save_model
from covid_prediction.prediction import PredictionService


def main():
    print("=" * 80)
    print("COVID-19 Prediction Service Demo")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    loader = CovidDatasetLoader()
    df = loader.load_dataset("Data/corona_tested_individuals_ver_006.english.csv")
    print(f"   Loaded {len(df)} records")
    
    preprocessor = CovidDataPreprocessor()
    X, y = preprocessor.preprocess(df)
    print(f"   Preprocessed to {X.shape[0]} samples with {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    print(f"   Split: {len(X_train)} training, {len(X_test)} testing samples")
    
    # Step 2: Apply feature engineering
    print("\n2. Applying feature engineering...")
    algorithm = 'logistic_regression'
    feature_names = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
                     'age_60_and_above', 'gender', 'test_indication']
    
    feature_engineer = FeatureEngineer(algorithm=algorithm)
    X_train_eng, engineered_feature_names = feature_engineer.fit_transform(X_train, feature_names)
    print(f"   Engineered features: {X_train.shape[1]} -> {X_train_eng.shape[1]}")
    
    # Step 3: Train model
    print("\n3. Training model...")
    pipeline = TrainingPipeline()
    model = pipeline.train(
        X_train_eng, 
        y_train, 
        algorithm=algorithm,
        balance_classes=True
    )
    print(f"   Model trained successfully")
    
    # Step 4: Save model with metadata
    print("\n4. Saving model...")
    metadata = {
        'algorithm': algorithm,
        'training_date': '2024-01-15',
        'dataset_size': len(X_train),
        'feature_names': engineered_feature_names,
        'feature_engineering_applied': True,
        'original_feature_count': len(feature_names),
        'engineered_feature_count': len(engineered_feature_names),
        'test_indication_encoder': preprocessor.test_indication_encoder
    }
    model_path = save_model(model, metadata, feature_engineer=feature_engineer)
    print(f"   Model saved to: {model_path}")
    
    # Step 5: Load model and create prediction service
    print("\n5. Creating prediction service...")
    service = PredictionService(model_path)
    print(f"   Prediction service initialized")
    
    # Step 6: Make predictions on sample data
    print("\n6. Making predictions...")
    print("-" * 80)
    
    # Example 1: Patient with multiple symptoms
    patient1 = {
        'cough': 1,
        'fever': 1,
        'sore_throat': 1,
        'shortness_of_breath': 1,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'male',
        'test_indication': 'Contact with confirmed'
    }
    
    print("\nPatient 1 (Multiple symptoms, elderly, contact with confirmed):")
    print(f"  Symptoms: cough=1, fever=1, sore_throat=1, shortness_of_breath=1, head_ache=1")
    print(f"  Demographics: age_60_and_above=Yes, gender=male")
    print(f"  Test indication: Contact with confirmed")
    
    result1 = service.predict(patient1)
    print(f"\n  Prediction: {result1.predicted_class}")
    print(f"  Confidence: {result1.confidence:.2%}")
    print(f"  Timestamp: {result1.timestamp}")
    
    # Example 2: Patient with few symptoms
    patient2 = {
        'cough': 0,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 0,
        'age_60_and_above': 'No',
        'gender': 'female',
        'test_indication': 'Abroad'
    }
    
    print("\n" + "-" * 80)
    print("\nPatient 2 (Few symptoms, young, travel abroad):")
    print(f"  Symptoms: cough=0, fever=0, sore_throat=1, shortness_of_breath=0, head_ache=0")
    print(f"  Demographics: age_60_and_above=No, gender=female")
    print(f"  Test indication: Abroad")
    
    result2 = service.predict(patient2)
    print(f"\n  Prediction: {result2.predicted_class}")
    print(f"  Confidence: {result2.confidence:.2%}")
    print(f"  Timestamp: {result2.timestamp}")
    
    # Example 3: Test validation errors
    print("\n" + "-" * 80)
    print("\n7. Testing validation...")
    
    # Invalid symptom value
    invalid_patient = {
        'cough': 2,  # Invalid: should be 0 or 1
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'male',
        'test_indication': 'Contact with confirmed'
    }
    
    print("\nTesting invalid symptom value (cough=2)...")
    try:
        service.predict(invalid_patient)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Validation error caught: {str(e)[:80]}...")
    
    # Invalid gender
    invalid_patient2 = {
        'cough': 1,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'other',  # Invalid: should be 'male' or 'female'
        'test_indication': 'Contact with confirmed'
    }
    
    print("\nTesting invalid gender (gender='other')...")
    try:
        service.predict(invalid_patient2)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Validation error caught: {str(e)[:80]}...")
    
    # Empty test indication
    invalid_patient3 = {
        'cough': 1,
        'fever': 0,
        'sore_throat': 1,
        'shortness_of_breath': 0,
        'head_ache': 1,
        'age_60_and_above': 'Yes',
        'gender': 'male',
        'test_indication': ''  # Invalid: should not be empty
    }
    
    print("\nTesting empty test indication...")
    try:
        service.predict(invalid_patient3)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Validation error caught: {str(e)[:80]}...")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
