"""
Dataset loader for COVID-19 test data.

This module provides functionality to load and validate COVID test datasets
from CSV files, ensuring all required columns are present.
"""

import pandas as pd
from typing import List


class CovidDatasetLoader:
    """Loader for COVID-19 test dataset with schema validation."""
    
    # Required columns for the COVID dataset
    REQUIRED_COLUMNS = [
        'test_date',
        'cough',
        'fever',
        'sore_throat',
        'shortness_of_breath',
        'head_ache',
        'corona_result',
        'age_60_and_above',
        'gender',
        'test_indication'
    ]
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load COVID test dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing loaded data
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Validate schema after loading
        self.validate_schema(df, self.REQUIRED_COLUMNS)
        
        return df
    
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If columns are missing (with list of missing columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        return True
