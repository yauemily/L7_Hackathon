"""
Data preprocessor for COVID-19 test data.

This module provides functionality to preprocess raw COVID test data,
including handling missing values, encoding categorical variables,
and splitting data into training and testing sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class CovidDataPreprocessor:
    """Preprocessor for COVID-19 test dataset."""
    
    def __init__(self):
        """Initialize the preprocessor with label encoders."""
        self.test_indication_encoder = LabelEncoder()
        self._is_fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'drop' or 'impute'
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy == 'drop':
            # Drop rows with missing values in age_60_and_above
            df_clean = df.dropna(subset=['age_60_and_above'])
            return df_clean
        elif strategy == 'impute':
            # Impute missing values with mode (most common value)
            if 'age_60_and_above' in df.columns:
                mode_value = df['age_60_and_above'].mode()[0] if not df['age_60_and_above'].mode().empty else 'No'
                df['age_60_and_above'] = df['age_60_and_above'].fillna(mode_value)
            return df
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'drop' or 'impute'.")
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables to numeric.
        
        Encoding scheme:
        - gender: male=0, female=1
        - age_60_and_above: No=0, Yes=1
        - test_indication: LabelEncoder (ordinal encoding)
        - corona_result: negative=0, positive=1
        
        Args:
            df: DataFrame with categorical columns
            fit: Whether to fit the encoders (True for training, False for test/prediction)
            
        Returns:
            DataFrame with encoded columns
        """
        df_encoded = df.copy()
        
        # Encode gender: male=0, female=1
        if 'gender' in df_encoded.columns:
            df_encoded['gender'] = df_encoded['gender'].map({'male': 0, 'female': 1})
        
        # Encode age_60_and_above: No=0, Yes=1
        if 'age_60_and_above' in df_encoded.columns:
            df_encoded['age_60_and_above'] = df_encoded['age_60_and_above'].map({'No': 0, 'Yes': 1})
        
        # Encode test_indication using LabelEncoder
        if 'test_indication' in df_encoded.columns:
            if fit:
                df_encoded['test_indication'] = self.test_indication_encoder.fit_transform(
                    df_encoded['test_indication'].astype(str)
                )
            else:
                df_encoded['test_indication'] = self.test_indication_encoder.transform(
                    df_encoded['test_indication'].astype(str)
                )
        
        # Encode corona_result: negative=0, positive=1
        if 'corona_result' in df_encoded.columns:
            df_encoded['corona_result'] = df_encoded['corona_result'].map({'negative': 0, 'positive': 1})
        
        return df_encoded

    def preprocess(self, df: pd.DataFrame, missing_value_strategy: str = 'drop') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess raw data for model training.
        
        Args:
            df: Raw DataFrame
            missing_value_strategy: Strategy for handling missing values ('drop' or 'impute')
            
        Returns:
            Tuple of (feature_matrix, target_vector)
        """
        # Step 1: Filter out rows with corona_result = 'other' (keep only positive/negative)
        df_filtered = df[df['corona_result'].isin(['positive', 'negative'])].copy()
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df_filtered, strategy=missing_value_strategy)
        
        # Step 3: Encode categorical variables
        df_encoded = self.encode_categorical(df_clean, fit=True)
        
        # Step 4: Ensure symptom features are binary (0/1)
        symptom_columns = ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache']
        for col in symptom_columns:
            if col in df_encoded.columns:
                # Convert to numeric and ensure binary values
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0).astype(int)
                # Ensure values are 0 or 1
                df_encoded[col] = df_encoded[col].apply(lambda x: 1 if x > 0 else 0)
        
        # Step 5: Separate features and target
        feature_columns = [
            'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache',
            'age_60_and_above', 'gender', 'test_indication'
        ]
        
        # Extract feature matrix (X) and target vector (y)
        X = df_encoded[feature_columns].values
        y = df_encoded['corona_result'].values
        
        self._is_fitted = True
        
        return X, y
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Stratification to maintain class distribution
        )
