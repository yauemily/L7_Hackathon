"""
Feature importance analysis module for COVID-19 prediction model.

This module provides the FeatureImportanceAnalyzer class for extracting,
ranking, and visualizing feature importance scores from trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyzes and visualizes feature importance from trained models.
    
    Supports:
    - Tree-based models (feature_importances_ attribute)
    - Linear models (coef_ attribute)
    - Ranking features by importance
    - Visualization of importance scores
    - Saving importance reports to file
    """
    
    def extract_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance scores from model.
        
        Supports two types of models:
        - Tree-based models (Random Forest, Gradient Boosting): use feature_importances_
        - Linear models (Logistic Regression): use absolute values of coef_
        
        Args:
            model: Trained model (must support feature_importances_ or coef_)
            feature_names: List of feature names corresponding to model features
            
        Returns:
            DataFrame with columns ['feature', 'importance'], sorted by importance
            in descending order
            
        Raises:
            ValueError: If model does not support feature importance extraction
        """
        # Check if model has feature_importances_ (tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            logger.info("Extracted feature importance from feature_importances_ attribute")
        
        # Check if model has coef_ (linear models)
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute values of coefficients
            # coef_ shape is (n_classes, n_features) for multiclass or (n_features,) for binary
            coef = model.coef_
            if coef.ndim == 2:
                # For binary classification, take the first row
                importance_scores = np.abs(coef[0])
            else:
                importance_scores = np.abs(coef)
            logger.info("Extracted feature importance from coef_ attribute (absolute values)")
        
        else:
            # Model doesn't support feature importance
            error_msg = (
                "[FeatureImportanceAnalyzer] ValueError: Model does not support feature importance. "
                "Model must have either 'feature_importances_' or 'coef_' attribute."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate that we have the right number of features
        if len(importance_scores) != len(feature_names):
            error_msg = (
                f"[FeatureImportanceAnalyzer] ValueError: Feature count mismatch. "
                f"Expected: {len(feature_names)} features, Got: {len(importance_scores)} importance scores"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance in descending order
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"Extracted importance for {len(feature_names)} features")
        
        return importance_df
    
    def visualize_importance(self, importance_df: pd.DataFrame, 
                           save_path: Optional[str] = None,
                           top_n: int = 20) -> None:
        """
        Create bar plot of feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            save_path: Optional path to save visualization (e.g., 'feature_importance.png')
            top_n: Number of top features to display (default: 20)
        """
        # Limit to top N features for readability
        plot_df = importance_df.head(top_n)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.3)))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(plot_df))
        ax.barh(y_pos, plot_df['importance'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['feature'])
        ax.invert_yaxis()  # Highest importance at the top
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {len(plot_df)} Feature Importance')
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance visualization to {save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
        
        logger.info(f"Created feature importance visualization for top {len(plot_df)} features")
    
    def save_report(self, importance_df: pd.DataFrame, save_path: str) -> None:
        """
        Save feature importance report to CSV file.
        
        Args:
            importance_df: DataFrame with feature importance scores
            save_path: Path to save CSV file (e.g., 'feature_importance_report.csv')
        """
        importance_df.to_csv(save_path, index=False)
        logger.info(f"Saved feature importance report to {save_path}")
