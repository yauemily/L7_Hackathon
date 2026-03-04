"""Configuration file for COVID-19 Prediction Model."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PathConfig:
    """Configuration for file paths."""
    data_dir: str = "Data"
    dataset_file: str = "corona_tested_individuals_ver_006.english.csv"
    models_dir: str = "models"
    
    @property
    def dataset_path(self) -> str:
        """Get full path to dataset file."""
        return os.path.join(self.data_dir, self.dataset_file)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    test_size: float = 0.2
    random_state: int = 42
    missing_value_strategy: str = 'drop'  # 'drop' or 'impute'
    stratify: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training."""
    algorithm: str = 'random_forest'  # 'logistic_regression', 'random_forest', 'gradient_boosting'
    balance_classes: bool = True
    balance_method: str = 'class_weights'  # 'class_weights', 'smote', 'undersample'
    cross_validation_folds: int = 5
    
    # Hyperparameters by algorithm
    logistic_regression_params: Dict[str, Any] = None
    random_forest_params: Dict[str, Any] = None
    gradient_boosting_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default hyperparameters."""
        if self.logistic_regression_params is None:
            self.logistic_regression_params = {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs'
            }
        
        if self.random_forest_params is None:
            self.random_forest_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if self.gradient_boosting_params is None:
            self.gradient_boosting_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }


# Default configuration instances
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
