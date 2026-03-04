"""
Feature engineering for COVID-19 prediction model.

This module provides functionality to create engineered features that improve
model performance, particularly for linear models like logistic regression.
Tree-based models (random forest, gradient boosting) work with raw features.
"""

import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class FeatureEngineer:
    """
    Feature engineer that applies algorithm-specific transformations.
    
    Strategy:
    - Logistic Regression: Apply ALL feature engineering (interactions, polynomial, scaling)
    - Random Forest: NO feature engineering (use raw features)
    - Gradient Boosting: NO feature engineering (use raw features)
    """
    
    def __init__(self, algorithm: str = 'logistic_regression'):
        """
        Initialize feature engineer with algorithm-specific configuration.
        
        Args:
            algorithm: 'logistic_regression', 'random_forest', or 'gradient_boosting'
                      Determines which feature engineering steps to apply
        """
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self._is_fitted = False
        self._feature_names = []
        self._original_feature_names = []
        
        # Symptom feature indices (first 5 features in the standard order)
        # Order: cough, fever, sore_throat, shortness_of_breath, head_ache
        self.symptom_indices = [0, 1, 2, 3, 4]
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Fit feature engineering transformations and transform training data.
        
        Args:
            X: Training feature matrix (n_samples, n_features)
            feature_names: List of original feature names
            
        Returns:
            Tuple of (transformed_features, new_feature_names)
            
        Note:
            - For logistic regression: applies interaction features, polynomial features, and scaling
            - For tree-based models: returns original features (no engineering needed)
        """
        self._original_feature_names = feature_names.copy()
        
        # For tree-based models, return original features without engineering
        if self.algorithm in ['random_forest', 'gradient_boosting']:
            self._is_fitted = True
            self._feature_names = feature_names.copy()
            return X.copy(), feature_names.copy()
        
        # For logistic regression, apply full feature engineering pipeline
        if self.algorithm == 'logistic_regression':
            # Step 1: Create interaction features for symptoms
            X_interactions = self.create_interaction_features(X, self.symptom_indices)
            
            # Step 2: Combine original features with interaction features
            X_with_interactions = np.hstack([X, X_interactions])
            
            # Step 3: Create polynomial features
            X_poly = self.create_polynomial_features(X_with_interactions, degree=2)
            
            # Step 4: Apply feature scaling
            X_scaled = self.apply_feature_scaling(X_poly)
            
            # Step 5: Generate feature names
            interaction_names = self._generate_interaction_names(feature_names)
            combined_names = feature_names + interaction_names
            poly_names = self._generate_polynomial_names(combined_names)
            
            self._feature_names = poly_names
            self._is_fitted = True
            
            return X_scaled, poly_names
        
        # Default: return original features
        self._is_fitted = True
        self._feature_names = feature_names.copy()
        return X.copy(), feature_names.copy()
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted transformations.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
            
        Raises:
            RuntimeError: If transform called before fit_transform
        """
        if not self._is_fitted:
            raise RuntimeError(
                "[FeatureEngineer] RuntimeError: Transform called before fit_transform. "
                "Must call fit_transform on training data first."
            )
        
        # For tree-based models, return original features
        if self.algorithm in ['random_forest', 'gradient_boosting']:
            return X.copy()
        
        # For logistic regression, apply the same transformations
        if self.algorithm == 'logistic_regression':
            # Step 1: Create interaction features
            X_interactions = self.create_interaction_features(X, self.symptom_indices)
            
            # Step 2: Combine with original features
            X_with_interactions = np.hstack([X, X_interactions])
            
            # Step 3: Apply polynomial features (using fitted transformer)
            X_poly = self.poly_features.transform(X_with_interactions)
            
            # Step 4: Apply scaling (using fitted scaler)
            X_scaled = self.scaler.transform(X_poly)
            
            return X_scaled
        
        # Default: return original features
        return X.copy()
    
    def create_interaction_features(self, X: np.ndarray, symptom_indices: List[int]) -> np.ndarray:
        """
        Create pairwise interaction features between symptoms.
        
        Interactions capture co-occurrence patterns like:
        - fever AND cough
        - shortness_of_breath AND fever
        - sore_throat AND cough
        
        Args:
            X: Feature matrix
            symptom_indices: Indices of symptom columns (binary features)
            
        Returns:
            Array of interaction features (n_samples, n_interactions)
            where n_interactions = n_symptoms * (n_symptoms - 1) / 2
        """
        interactions = []
        n_symptoms = len(symptom_indices)
        
        # Create pairwise interactions
        for i in range(n_symptoms):
            for j in range(i + 1, n_symptoms):
                idx_i = symptom_indices[i]
                idx_j = symptom_indices[j]
                # Interaction is the product of two binary features
                interaction = X[:, idx_i] * X[:, idx_j]
                interactions.append(interaction)
        
        # Stack all interactions into a single array
        if interactions:
            return np.column_stack(interactions)
        else:
            # Return empty array with correct shape if no interactions
            return np.empty((X.shape[0], 0))
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features to capture non-linear relationships.
        
        For degree=2, creates:
        - Original features: x1, x2, x3, ...
        - Squared features: x1^2, x2^2, x3^2, ...
        - Interaction features: x1*x2, x1*x3, x2*x3, ...
        
        Args:
            X: Feature matrix
            degree: Polynomial degree (typically 2 for binary features)
            
        Returns:
            Array with polynomial features
            
        Note:
            Uses sklearn.preprocessing.PolynomialFeatures with include_bias=False
        """
        # Fit and transform during fit_transform, only transform during transform
        if not self._is_fitted:
            return self.poly_features.fit_transform(X)
        else:
            return self.poly_features.transform(X)
    
    def apply_feature_scaling(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization (z-score normalization) to features.
        
        Transforms features to have mean=0 and std=1.
        Critical for logistic regression to ensure:
        - Faster convergence
        - Proper regularization
        - Comparable feature coefficients
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
            
        Note:
            Uses sklearn.preprocessing.StandardScaler
            Scaler is fitted during fit_transform and reused in transform
        """
        # Fit and transform during fit_transform, only transform during transform
        if not self._is_fitted:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all engineered features.
        
        Returns:
            List of feature names including:
            - Original features
            - Interaction features (e.g., 'fever_AND_cough')
            - Polynomial features (e.g., 'age_60_and_above^2')
        """
        return self._feature_names.copy()
    
    def _generate_interaction_names(self, feature_names: List[str]) -> List[str]:
        """
        Generate names for interaction features.
        
        Args:
            feature_names: Original feature names
            
        Returns:
            List of interaction feature names
        """
        interaction_names = []
        symptom_names = [feature_names[i] for i in self.symptom_indices]
        n_symptoms = len(symptom_names)
        
        for i in range(n_symptoms):
            for j in range(i + 1, n_symptoms):
                name = f"{symptom_names[i]}_AND_{symptom_names[j]}"
                interaction_names.append(name)
        
        return interaction_names
    
    def _generate_polynomial_names(self, feature_names: List[str]) -> List[str]:
        """
        Generate names for polynomial features.
        
        Args:
            feature_names: Feature names before polynomial expansion
            
        Returns:
            List of polynomial feature names
        """
        # Get feature names from PolynomialFeatures
        # This is a simplified version - sklearn's get_feature_names_out would be more accurate
        poly_names = []
        n_features = len(feature_names)
        
        # Original features
        for name in feature_names:
            poly_names.append(name)
        
        # Squared features
        for name in feature_names:
            poly_names.append(f"{name}^2")
        
        # Interaction features (pairwise)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                poly_names.append(f"{feature_names[i]}*{feature_names[j]}")
        
        return poly_names
