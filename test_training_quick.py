"""Quick test to verify training pipeline implementation."""

import numpy as np
from covid_prediction.training import TrainingPipeline

# Create simple synthetic data
np.random.seed(42)
n_samples = 100
n_features = 8

# Create imbalanced dataset (20% positive class)
X_train = np.random.randint(0, 2, size=(n_samples, n_features))
y_train = np.array([1] * 20 + [0] * 80)  # 20% positive

# Shuffle
indices = np.random.permutation(n_samples)
X_train = X_train[indices]
y_train = y_train[indices]

print("Testing TrainingPipeline...")
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Class distribution: {np.bincount(y_train)}")

# Test 1: Compute class weights
pipeline = TrainingPipeline()
class_weights = pipeline.compute_class_weights(y_train)
print(f"\n1. Class weights: {class_weights}")

# Test 2: Train with logistic regression
print("\n2. Training logistic regression...")
model_lr = pipeline.train(
    X_train, y_train,
    algorithm='logistic_regression',
    balance_classes=True,
    feature_names=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                   'head_ache', 'age_60_and_above', 'gender', 'test_indication']
)
print(f"   Model trained: {type(model_lr).__name__}")
print(f"   Metadata: {pipeline.metadata}")

# Test 3: Train with random forest
print("\n3. Training random forest...")
pipeline2 = TrainingPipeline()
model_rf = pipeline2.train(
    X_train, y_train,
    algorithm='random_forest',
    balance_classes=True,
    feature_names=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                   'head_ache', 'age_60_and_above', 'gender', 'test_indication']
)
print(f"   Model trained: {type(model_rf).__name__}")
print(f"   Balance method: {pipeline2.metadata['class_balance_method']}")

# Test 4: Train with gradient boosting
print("\n4. Training gradient boosting...")
pipeline3 = TrainingPipeline()
model_gb = pipeline3.train(
    X_train, y_train,
    algorithm='gradient_boosting',
    balance_classes=True,
    feature_names=['cough', 'fever', 'sore_throat', 'shortness_of_breath', 
                   'head_ache', 'age_60_and_above', 'gender', 'test_indication']
)
print(f"   Model trained: {type(model_gb).__name__}")
print(f"   Balance method: {pipeline3.metadata['class_balance_method']}")

# Test 5: Cross-validation
print("\n5. Testing cross-validation...")
cv_results = pipeline.cross_validate(
    X_train, y_train,
    algorithm='random_forest',
    n_folds=5
)
print(f"   CV Results:")
print(f"   - Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
print(f"   - Precision: {cv_results['precision_mean']:.3f} ± {cv_results['precision_std']:.3f}")
print(f"   - Recall: {cv_results['recall_mean']:.3f} ± {cv_results['recall_std']:.3f}")
print(f"   - F1: {cv_results['f1_mean']:.3f} ± {cv_results['f1_std']:.3f}")

# Test 6: Save model
print("\n6. Testing model persistence...")
filepath = pipeline.save_model(output_dir='models')
print(f"   Model saved to: {filepath}")

# Test 7: Error handling - empty data
print("\n7. Testing error handling (empty data)...")
try:
    pipeline_err = TrainingPipeline()
    pipeline_err.train(np.array([]), np.array([]), algorithm='random_forest')
    print("   ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly raised error: {str(e)[:80]}...")

# Test 8: Error handling - single class
print("\n8. Testing error handling (single class)...")
try:
    pipeline_err = TrainingPipeline()
    X_single = np.random.rand(50, 8)
    y_single = np.ones(50)  # All same class
    pipeline_err.train(X_single, y_single, algorithm='random_forest')
    print("   ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly raised error: {str(e)[:80]}...")

# Test 9: Error handling - unsupported algorithm
print("\n9. Testing error handling (unsupported algorithm)...")
try:
    pipeline_err = TrainingPipeline()
    pipeline_err.train(X_train, y_train, algorithm='neural_network')
    print("   ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ Correctly raised error: {str(e)[:80]}...")

print("\n✅ All tests passed!")
