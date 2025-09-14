#!/usr/bin/env python3
"""
Competition Metrics for Hackathon Forecast 2025
Implementation of WMAPE and other relevant forecasting metrics
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Any
import warnings

def wmape(y_true: Union[np.ndarray, pd.Series], 
          y_pred: Union[np.ndarray, pd.Series],
          sample_weight: Union[np.ndarray, pd.Series] = None) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE)
    
    WMAPE = sum(|actual - forecast|) / sum(|actual|) * 100
    
    This is the PRIMARY METRIC for the hackathon competition.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values  
        sample_weight: Optional weights for each sample
        
    Returns:
        WMAPE as percentage (0-100)
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    # Handle sample weights
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight must have same length as y_true")
    else:
        sample_weight = np.ones_like(y_true)
    
    # Calculate weighted absolute errors and actuals
    abs_errors = np.abs(y_true - y_pred) * sample_weight
    abs_actuals = np.abs(y_true) * sample_weight
    
    # Avoid division by zero
    total_abs_actuals = np.sum(abs_actuals)
    if total_abs_actuals == 0:
        warnings.warn("Sum of absolute actuals is zero. WMAPE undefined.")
        return np.inf
    
    wmape_value = np.sum(abs_errors) / total_abs_actuals * 100
    
    return wmape_value

def mape(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series],
         epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Note: MAPE can be problematic with zero values, so we add epsilon
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE as percentage (0-100)
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero
    y_true_adj = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    ape = np.abs((y_true - y_pred) / y_true_adj) * 100
    return np.mean(ape)

def smape(y_true: Union[np.ndarray, pd.Series], 
          y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    
    sMAPE = 100 * mean(2 * |actual - forecast| / (|actual| + |forecast|))
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        sMAPE as percentage (0-100)
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape_value = 100 * np.mean(numerator / denominator)
    
    return smape_value

def mae(y_true: Union[np.ndarray, pd.Series], 
        y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def bias(y_true: Union[np.ndarray, pd.Series], 
         y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Forecast Bias (Mean Error)"""
    return np.mean(y_pred - y_true)

def wmape_by_group(y_true: Union[np.ndarray, pd.Series], 
                   y_pred: Union[np.ndarray, pd.Series],
                   groups: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate WMAPE for each group (e.g., by product category, store type)
    
    This is useful for understanding model performance across different segments
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        groups: Group identifiers
        
    Returns:
        Dictionary with WMAPE for each group
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    
    unique_groups = np.unique(groups)
    group_wmapes = {}
    
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:  # Ensure group has samples
            group_wmapes[str(group)] = wmape(y_true[mask], y_pred[mask])
    
    return group_wmapes

def volume_weighted_metrics(y_true: Union[np.ndarray, pd.Series], 
                           y_pred: Union[np.ndarray, pd.Series],
                           volumes: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate volume-weighted forecasting metrics
    
    This gives more weight to high-volume products, which is important for retail
    
    Args:
        y_true: Actual values
        y_pred: Predicted values  
        volumes: Volume weights (e.g., sales quantities)
        
    Returns:
        Dictionary of volume-weighted metrics
    """
    
    return {
        'volume_weighted_wmape': wmape(y_true, y_pred, sample_weight=volumes),
        'volume_weighted_mae': np.average(np.abs(y_true - y_pred), weights=volumes),
        'volume_weighted_bias': np.average(y_pred - y_true, weights=volumes)
    }

def forecast_accuracy_tiers(y_true: Union[np.ndarray, pd.Series], 
                           y_pred: Union[np.ndarray, pd.Series],
                           volume_tiers: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics by volume tiers (A/B/C products)
    
    This is crucial for retail forecasting as different product tiers
    have different importance and difficulty levels
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        volume_tiers: Tier labels ('A', 'B', 'C')
        
    Returns:
        Dictionary with metrics for each tier
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    volume_tiers = np.asarray(volume_tiers)
    
    tier_metrics = {}
    
    for tier in ['A', 'B', 'C']:
        mask = volume_tiers == tier
        if np.sum(mask) > 0:
            tier_metrics[f'tier_{tier}'] = {
                'wmape': wmape(y_true[mask], y_pred[mask]),
                'mape': mape(y_true[mask], y_pred[mask]),
                'mae': mae(y_true[mask], y_pred[mask]),
                'bias': bias(y_true[mask], y_pred[mask]),
                'count': np.sum(mask)
            }
    
    return tier_metrics

def time_series_cv_score(y_true: pd.DataFrame, 
                        y_pred: pd.DataFrame,
                        date_col: str,
                        n_splits: int = 5) -> Dict[str, float]:
    """
    Time series cross-validation scoring
    
    Performs walk-forward validation appropriate for time series data
    
    Args:
        y_true: DataFrame with actual values and dates
        y_pred: DataFrame with predicted values and dates  
        date_col: Name of date column
        n_splits: Number of CV splits
        
    Returns:
        Cross-validation metrics
    """
    
    # Sort by date
    y_true_sorted = y_true.sort_values(date_col)
    y_pred_sorted = y_pred.sort_values(date_col)
    
    n_samples = len(y_true_sorted)
    test_size = n_samples // (n_splits + 1)
    
    wmape_scores = []
    mape_scores = []
    mae_scores = []
    
    for i in range(n_splits):
        # Define train/test split
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
            
        # Get test data
        y_true_test = y_true_sorted.iloc[test_start:test_end]
        y_pred_test = y_pred_sorted.iloc[test_start:test_end]
        
        # Calculate metrics for this split
        if len(y_true_test) > 0 and len(y_pred_test) > 0:
            # Assuming prediction column name
            pred_col = [col for col in y_pred_test.columns if col != date_col][0]
            actual_col = [col for col in y_true_test.columns if col != date_col][0]
            
            wmape_scores.append(wmape(y_true_test[actual_col], y_pred_test[pred_col]))
            mape_scores.append(mape(y_true_test[actual_col], y_pred_test[pred_col]))
            mae_scores.append(mae(y_true_test[actual_col], y_pred_test[pred_col]))
    
    return {
        'cv_wmape_mean': np.mean(wmape_scores),
        'cv_wmape_std': np.std(wmape_scores),
        'cv_mape_mean': np.mean(mape_scores), 
        'cv_mape_std': np.std(mape_scores),
        'cv_mae_mean': np.mean(mae_scores),
        'cv_mae_std': np.std(mae_scores),
        'n_splits': len(wmape_scores)
    }

def retail_forecast_evaluation(y_true: pd.DataFrame,
                              y_pred: pd.DataFrame,
                              product_info: pd.DataFrame = None,
                              store_info: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation for retail forecasting
    
    Includes all relevant metrics for the hackathon competition
    
    Args:
        y_true: DataFrame with actual sales
        y_pred: DataFrame with predictions
        product_info: Optional product metadata
        store_info: Optional store metadata
        
    Returns:
        Comprehensive evaluation results
    """
    
    # Assume standard column names
    actual_col = 'actual' if 'actual' in y_true.columns else y_true.columns[0]
    pred_col = 'prediction' if 'prediction' in y_pred.columns else y_pred.columns[0]
    
    y_true_values = y_true[actual_col].values
    y_pred_values = y_pred[pred_col].values
    
    # Basic metrics
    results = {
        'primary_metric': {
            'wmape': wmape(y_true_values, y_pred_values)
        },
        'secondary_metrics': {
            'mape': mape(y_true_values, y_pred_values),
            'smape': smape(y_true_values, y_pred_values),
            'mae': mae(y_true_values, y_pred_values),
            'rmse': rmse(y_true_values, y_pred_values),
            'bias': bias(y_true_values, y_pred_values)
        },
        'data_info': {
            'n_predictions': len(y_true_values),
            'actual_mean': np.mean(y_true_values),
            'actual_std': np.std(y_true_values),
            'pred_mean': np.mean(y_pred_values),
            'pred_std': np.std(y_pred_values)
        }
    }
    
    # Add product-level analysis if product info available
    if product_info is not None and 'product_id' in y_true.columns:
        # Group by product category if available
        if 'category' in product_info.columns:
            merged = y_true.merge(y_pred, on='product_id', how='inner')
            merged = merged.merge(product_info[['product_id', 'category']], on='product_id', how='left')
            
            category_wmapes = wmape_by_group(
                merged[actual_col], 
                merged[pred_col], 
                merged['category']
            )
            results['category_analysis'] = category_wmapes
    
    return results


# Utility functions for competition submission
def calculate_submission_score(submission_df: pd.DataFrame,
                              ground_truth_df: pd.DataFrame,
                              id_cols: list = ['store_id', 'product_id', 'date']) -> float:
    """
    Calculate final submission score (WMAPE)
    
    This simulates the competition scoring function
    """
    
    # Merge predictions with ground truth
    merged = ground_truth_df.merge(
        submission_df, 
        on=id_cols, 
        how='inner',
        suffixes=('_actual', '_pred')
    )
    
    if len(merged) == 0:
        raise ValueError("No matching records found between submission and ground truth")
    
    # Get value columns
    actual_col = [col for col in merged.columns if col.endswith('_actual')][0]
    pred_col = [col for col in merged.columns if col.endswith('_pred')][0]
    
    return wmape(merged[actual_col], merged[pred_col])


if __name__ == "__main__":
    # Test the metrics
    
    # Sample data
    np.random.seed(42)
    y_true = np.random.exponential(100, 1000)  # Retail-like data
    y_pred = y_true + np.random.normal(0, 10, 1000)  # Add some noise
    
    print("Testing Competition Metrics")
    print("=" * 50)
    
    # Test WMAPE (primary metric)
    wmape_score = wmape(y_true, y_pred)
    print(f"WMAPE: {wmape_score:.2f}%")
    
    # Test other metrics
    mape_score = mape(y_true, y_pred)
    mae_score = mae(y_true, y_pred)
    rmse_score = rmse(y_true, y_pred)
    
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MAE: {mae_score:.2f}")
    print(f"RMSE: {rmse_score:.2f}")
    
    # Test group-wise WMAPE
    groups = np.random.choice(['A', 'B', 'C'], 1000)
    group_wmapes = wmape_by_group(y_true, y_pred, groups)
    
    print("\nWMAPE by Group:")
    for group, wmape_val in group_wmapes.items():
        print(f"  Group {group}: {wmape_val:.2f}%")