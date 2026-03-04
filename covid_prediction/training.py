"""
Training pipeline for COVID-19 prediction model.

This module provides functionality to train machine learning models with
class balancing, cross-validation, and model persistence capabilities.
"""

import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


class TrainingPipeline:
    """Pipeline for training COVID-19 prediction models."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.model = None
        self.metadata = {}
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced dataset.
        
        Args:
            y: Target vector
            
        Returns:
            Dictionary mapping class labels to weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        algorithm: str = 'random_forest',
        balance_classes: bool = True,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[list] = None
    ) -> Any:
        """
        Train prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            algorithm: 'logistic_regression', 'random_forest', or 'gradient_boosting'
            balance_classes: Whether to apply class balancing
            hyperparameters: Algorithm-specific hyperparameters
            feature_names: List of feature names for metadata
            
        Returns:
            Trained model object
            
        Raises:
            ValueError: If training data is invalid or algorithm is unsupported
        """
        # Validate training data
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError(
                "[TrainingPipeline] ValueError: Training data is empty. "
                "Cannot train model with zero samples."
            )
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"[TrainingPipeline] ValueError: Mismatched X and y lengths. "
                f"X has {X_train.shape[0]} samples, y has {y_train.shape[0]} samples."
            )
        
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                "[TrainingPipeline] ValueError: Training data contains only one class. "
                "Need at least two classes for classification."
            )
        
        # Determine if class balancing is needed
        class_distribution = {int(cls): int(np.sum(y_train == cls)) for cls in np.unique(y_train)}
        total_samples = len(y_train)
        positive_ratio = class_distribution.get(1, 0) / total_samples if total_samples > 0 else 0
        
        # Apply class balancing if needed (positive class < 30%)
        balance_method = None
        if balance_classes and positive_ratio < 0.3:
            balance_method = self._determine_balance_method(algorithm)
        
        # Initialize hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        try:
            # Train model based on algorithm
            if algorithm == 'logistic_regression':
                self.model = self._train_logistic_regression(
                    X_train, y_train, hyperparameters, balance_method
                )
            elif algorithm == 'random_forest':
                self.model = self._train_random_forest(
                    X_train, y_train, hyperparameters, balance_method
                )
            elif algorithm == 'gradient_boosting':
                self.model = self._train_gradient_boosting(
                    X_train, y_train, hyperparameters, balance_method
                )
            else:
                raise ValueError(
                    f"[TrainingPipeline] ValueError: Unsupported algorithm '{algorithm}'. "
                    f"Supported algorithms: 'logistic_regression', 'random_forest', 'gradient_boosting'"
                )
            
            # Record training metadata
            self.metadata = {
                'algorithm': algorithm,
                'training_date': datetime.now().isoformat(),
                'dataset_size': int(X_train.shape[0]),
                'class_balance_method': balance_method,
                'feature_names': feature_names if feature_names else [f'feature_{i}' for i in range(X_train.shape[1])],
                'class_distribution': class_distribution,
                'hyperparameters': hyperparameters
            }
            
            return self.model
            
        except Exception as e:
            # Re-raise with descriptive message if not already formatted
            if not str(e).startswith('[TrainingPipeline]'):
                raise ValueError(
                    f"[TrainingPipeline] ValueError: Training failed - {str(e)}"
                )
            raise
    
    def _determine_balance_method(self, algorithm: str) -> str:
        """
        Determine the appropriate class balancing method for the algorithm.
        
        Args:
            algorithm: Model algorithm
            
        Returns:
            Balance method name
        """
        # Use class_weight for algorithms that support it
        if algorithm in ['logistic_regression', 'random_forest']:
            return 'class_weights'
        # Use SMOTE for gradient boosting
        elif algorithm == 'gradient_boosting':
            return 'smote'
        return 'class_weights'
    
    def _train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Dict[str, Any],
        balance_method: Optional[str]
    ) -> LogisticRegression:
        """Train logistic regression model."""
        params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        params.update(hyperparameters)
        
        # Apply class balancing
        if balance_method == 'class_weights':
            params['class_weight'] = 'balanced'
        elif balance_method == 'smote':
            X_train, y_train = self._apply_smote(X_train, y_train)
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Dict[str, Any],
        balance_method: Optional[str]
    ) -> RandomForestClassifier:
        """Train random forest model."""
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(hyperparameters)
        
        # Apply class balancing
        if balance_method == 'class_weights':
            params['class_weight'] = 'balanced'
        elif balance_method == 'smote':
            X_train, y_train = self._apply_smote(X_train, y_train)
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperparameters: Dict[str, Any],
        balance_method: Optional[str]
    ) -> GradientBoostingClassifier:
        """Train gradient boosting model."""
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        params.update(hyperparameters)
        
        # Apply class balancing (SMOTE for gradient boosting)
        if balance_method == 'smote':
            X_train, y_train = self._apply_smote(X_train, y_train)
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithm: str = 'random_forest',
        n_folds: int = 5,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            algorithm: Model algorithm to use
            n_folds: Number of folds
            hyperparameters: Algorithm-specific hyperparameters
            
        Returns:
            Dictionary with mean and std of metrics across folds
            
        Raises:
            ValueError: If data is invalid or algorithm is unsupported
        """
        # Validate data
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError(
                "[TrainingPipeline] ValueError: Data is empty. "
                "Cannot perform cross-validation with zero samples."
            )
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"[TrainingPipeline] ValueError: Mismatched X and y lengths. "
                f"X has {X.shape[0]} samples, y has {y.shape[0]} samples."
            )
        
        # Initialize hyperparameters
        if hyperparameters is None:
            hyperparameters = {}
        
        # Create model based on algorithm
        try:
            if algorithm == 'logistic_regression':
                params = {
                    'max_iter': 1000,
                    'random_state': 42,
                    'solver': 'lbfgs',
                    'class_weight': 'balanced'
                }
                params.update(hyperparameters)
                model = LogisticRegression(**params)
            elif algorithm == 'random_forest':
                params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
                params.update(hyperparameters)
                model = RandomForestClassifier(**params)
            elif algorithm == 'gradient_boosting':
                params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
                params.update(hyperparameters)
                model = GradientBoostingClassifier(**params)
            else:
                raise ValueError(
                    f"[TrainingPipeline] ValueError: Unsupported algorithm '{algorithm}'. "
                    f"Supported algorithms: 'logistic_regression', 'random_forest', 'gradient_boosting'"
                )
            
            # Perform cross-validation with multiple metrics
            scoring = ['accuracy', 'precision', 'recall', 'f1']
            results = {}
            
            for metric in scoring:
                scores = cross_val_score(model, X, y, cv=n_folds, scoring=metric)
                results[f'{metric}_mean'] = float(np.mean(scores))
                results[f'{metric}_std'] = float(np.std(scores))
            
            results['n_folds'] = n_folds
            results['algorithm'] = algorithm
            
            return results
            
        except Exception as e:
            # Re-raise with descriptive message if not already formatted
            if not str(e).startswith('[TrainingPipeline]'):
                raise ValueError(
                    f"[TrainingPipeline] ValueError: Cross-validation failed - {str(e)}"
                )
            raise
    
    def save_model(
        self,
        output_dir: str = 'models',
        feature_engineer: Optional[Any] = None
    ) -> str:
        """
        Persist trained model using joblib.
        
        Args:
            output_dir: Directory to save model
            feature_engineer: Optional FeatureEngineer object to save with model
            
        Returns:
            Path to saved model file
            
        Raises:
            ValueError: If no model has been trained
        """
        if self.model is None:
            raise ValueError(
                "[TrainingPipeline] ValueError: No model to save. "
                "Must train a model before saving."
            )
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        algorithm = self.metadata.get('algorithm', 'model')
        filename = f"{algorithm}_{timestamp}.joblib"
        filepath = os.path.join(output_dir, filename)
        
        # Package model with metadata and feature engineer
        model_package = {
            'model': self.model,
            'metadata': self.metadata,
            'feature_engineer': feature_engineer
        }
        
        # Save using joblib
        joblib.dump(model_package, filepath)
        
        return filepath
