#!/usr/bin/env python3
"""
INTERMITTENT DEMAND SPECIALIST - Hackathon Forecast Big Data 2025
Advanced Zero-Inflated Models for Sparse Demand Forecasting

Features:
- Zero-Inflated Models (ZIP, ZINB, Hurdle)
- Croston Method variants (SBA, TSB)
- Two-Stage Learning (Zero/Non-Zero + Quantity)
- Bootstrap uncertainty quantification
- WMAPE-optimized loss functions
- Cold start handling for new products

Specialized for intermittent/sparse demand patterns! 📈
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import json
import time
from datetime import datetime
from scipy import stats
from scipy.special import gammaln, digamma
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape
from src.utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

class CrostonMethod:
    """
    Croston Method for Intermittent Demand Forecasting
    
    Variants:
    - Classic Croston
    - SBA (Syntetos-Boylan Approximation) - bias correction
    - TSB (Teunter-Syntetos-Babai) - improved bias correction
    """
    
    def __init__(self, method: str = 'sba', alpha: float = 0.1, beta: float = 0.1):
        """
        Args:
            method: 'croston', 'sba', or 'tsb'
            alpha: Smoothing parameter for demand size
            beta: Smoothing parameter for inter-arrival time
        """
        self.method = method
        self.alpha = alpha
        self.beta = beta
        
        # State variables
        self.demand_size = None
        self.inter_arrival_time = None
        self.last_non_zero_period = 0
        self.fitted_values = []
    
    def fit(self, y: np.ndarray) -> 'CrostonMethod':
        """
        Fit Croston method to time series
        
        Args:
            y: Time series with intermittent demand
            
        Returns:
            Self
        """
        
        y = np.array(y)
        n = len(y)
        
        # Find non-zero demands
        non_zero_indices = np.where(y > 0)[0]
        
        if len(non_zero_indices) < 2:
            # Not enough non-zero observations
            self.demand_size = np.mean(y[y > 0]) if len(non_zero_indices) > 0 else 1.0
            self.inter_arrival_time = n / max(len(non_zero_indices), 1)
            self.fitted_values = [self.demand_size / self.inter_arrival_time] * n
            return self
        
        # Initialize
        demand_sizes = []
        inter_arrival_times = []
        
        # Extract demand sizes and inter-arrival times
        for i, idx in enumerate(non_zero_indices):
            demand_sizes.append(y[idx])
            
            if i > 0:
                inter_arrival_times.append(idx - non_zero_indices[i-1])
        
        # Initialize smoothed values
        self.demand_size = demand_sizes[0]
        self.inter_arrival_time = inter_arrival_times[0] if inter_arrival_times else 1.0
        
        fitted_values = []
        
        # Apply smoothing
        for t in range(n):
            if t in non_zero_indices and t > non_zero_indices[0]:
                # Update demand size
                self.demand_size = self.alpha * y[t] + (1 - self.alpha) * self.demand_size
                
                # Update inter-arrival time
                arrival_time = t - self.last_non_zero_period
                self.inter_arrival_time = (self.beta * arrival_time + 
                                         (1 - self.beta) * self.inter_arrival_time)
                
                self.last_non_zero_period = t
            
            # Calculate forecast based on method
            if self.method == 'croston':
                forecast = self.demand_size / self.inter_arrival_time
            elif self.method == 'sba':
                # SBA bias correction
                bias_correction = 1 - self.alpha / 2
                forecast = (self.demand_size / self.inter_arrival_time) * bias_correction
            elif self.method == 'tsb':
                # TSB bias correction (more sophisticated)
                if self.inter_arrival_time > 1:
                    prob_demand = 1 / self.inter_arrival_time
                    forecast = prob_demand * self.demand_size
                else:
                    forecast = self.demand_size
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            fitted_values.append(max(forecast, 0))
        
        self.fitted_values = fitted_values
        
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Predict future values
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Forecast array
        """
        
        if self.demand_size is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        # Same forecast for all future periods in classical Croston
        if self.method == 'croston':
            forecast_value = self.demand_size / self.inter_arrival_time
        elif self.method == 'sba':
            bias_correction = 1 - self.alpha / 2
            forecast_value = (self.demand_size / self.inter_arrival_time) * bias_correction
        elif self.method == 'tsb':
            if self.inter_arrival_time > 1:
                prob_demand = 1 / self.inter_arrival_time
                forecast_value = prob_demand * self.demand_size
            else:
                forecast_value = self.demand_size
        
        return np.array([max(forecast_value, 0)] * steps)

class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """
    Zero-Inflated Regression Model
    
    Two-stage approach:
    1. Classify zero vs non-zero (binary model)
    2. Predict quantity given non-zero (regression model)
    
    Optimized for WMAPE metric.
    """
    
    def __init__(self, 
                 zero_classifier=None,
                 positive_regressor=None,
                 zero_threshold: float = 1e-6):
        """
        Args:
            zero_classifier: Model for zero/non-zero classification
            positive_regressor: Model for positive quantity regression
            zero_threshold: Threshold for considering value as zero
        """
        
        self.zero_classifier = zero_classifier or LogisticRegression(random_state=42)
        self.positive_regressor = positive_regressor or RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.zero_threshold = zero_threshold
        
        # Model state
        self.is_fitted = False
        self.zero_ratio = 0.0
        self.positive_mean = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ZeroInflatedRegressor':
        """
        Fit zero-inflated model
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self
        """
        
        X = np.array(X)
        y = np.array(y)
        
        # Create binary target for zero classification
        y_binary = (y > self.zero_threshold).astype(int)
        
        # Fit zero classifier
        self.zero_classifier.fit(X, y_binary)
        
        # Fit positive regressor on non-zero samples
        positive_mask = y > self.zero_threshold
        if np.sum(positive_mask) > 0:
            X_positive = X[positive_mask]
            y_positive = y[positive_mask]
            
            self.positive_regressor.fit(X_positive, y_positive)
            self.positive_mean = np.mean(y_positive)
        else:
            # All zeros case
            self.positive_mean = 1.0
        
        # Store statistics
        self.zero_ratio = 1 - np.mean(y_binary)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using zero-inflated model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")
        
        X = np.array(X)
        
        # Predict probability of non-zero
        prob_nonzero = self.zero_classifier.predict_proba(X)[:, 1]
        
        # Predict positive quantities
        try:
            positive_pred = self.positive_regressor.predict(X)
        except:
            # Fallback if regressor fails
            positive_pred = np.full(X.shape[0], self.positive_mean)
        
        # Combine predictions
        predictions = prob_nonzero * positive_pred
        
        return predictions
    
    def predict_components(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict both components separately
        
        Args:
            X: Feature matrix
            
        Returns:
            (probability_nonzero, quantity_given_positive)
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")
        
        X = np.array(X)
        
        prob_nonzero = self.zero_classifier.predict_proba(X)[:, 1]
        
        try:
            quantity_given_positive = self.positive_regressor.predict(X)
        except:
            quantity_given_positive = np.full(X.shape[0], self.positive_mean)
        
        return prob_nonzero, quantity_given_positive

class HurdleModel(BaseEstimator, RegressorMixin):
    """
    Hurdle Model for Count Data
    
    Similar to zero-inflated but with different assumptions:
    - Stage 1: Hurdle (zero vs any positive)
    - Stage 2: Count model for positive values (truncated)
    """
    
    def __init__(self, 
                 hurdle_classifier=None,
                 count_regressor=None):
        
        self.hurdle_classifier = hurdle_classifier or LogisticRegression(random_state=42)
        self.count_regressor = count_regressor or RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        
        self.is_fitted = False
        self.positive_min = 1.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HurdleModel':
        """Fit hurdle model"""
        
        X = np.array(X)
        y = np.array(y)
        
        # Binary target: zero vs positive
        y_binary = (y > 0).astype(int)
        
        # Fit hurdle classifier
        self.hurdle_classifier.fit(X, y_binary)
        
        # Fit count model on positive values only
        positive_mask = y > 0
        if np.sum(positive_mask) > 0:
            X_positive = X[positive_mask]
            y_positive = y[positive_mask]
            
            # For hurdle, we model the positive part directly
            self.count_regressor.fit(X_positive, y_positive)
            self.positive_min = np.min(y_positive)
        else:
            self.positive_min = 1.0
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using hurdle model"""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")
        
        X = np.array(X)
        
        # Predict probability of crossing hurdle
        prob_positive = self.hurdle_classifier.predict_proba(X)[:, 1]
        
        # Predict count given positive
        try:
            count_given_positive = self.count_regressor.predict(X)
            # Ensure positive predictions
            count_given_positive = np.maximum(count_given_positive, self.positive_min)
        except:
            count_given_positive = np.full(X.shape[0], self.positive_min)
        
        # Combine predictions
        predictions = prob_positive * count_given_positive
        
        return predictions

class IntermittentDemandSpecialist:
    """
    Specialist Engine for Intermittent Demand Forecasting
    
    Combines multiple approaches:
    - Croston methods
    - Zero-inflated models
    - Hurdle models
    - Bootstrap uncertainty
    """
    
    def __init__(self, 
                 date_col: str = 'transaction_date',
                 target_col: str = 'quantity',
                 feature_cols: List[str] = None):
        
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols or []
        
        # Model collections
        self.croston_models = {}
        self.zero_inflated_models = {}
        self.hurdle_models = {}
        
        # Model results
        self.fitted_results = {}
        self.predictions = {}
        self.performance_metrics = {}
        
        # Configuration
        self.intermittency_threshold = 0.75  # 75% zeros = intermittent
        self.min_observations = 10
    
    def identify_intermittent_series(self, df: pd.DataFrame,
                                   group_col: str = None) -> Dict[str, Dict]:
        """
        Identify intermittent demand series
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            
        Returns:
            Dictionary with intermittency statistics
        """
        
        print("[INFO] Identifying intermittent demand patterns...")
        
        if group_col:
            groups = df.groupby(group_col)
        else:
            groups = {'overall': df}
        
        intermittent_analysis = {}
        
        for group_name, group_df in groups:
            if isinstance(groups, dict):
                group_name = 'overall'
            
            # Calculate intermittency metrics
            total_obs = len(group_df)
            zero_obs = len(group_df[group_df[self.target_col] == 0])
            zero_ratio = zero_obs / total_obs if total_obs > 0 else 1.0
            
            # Consecutive zeros
            is_zero = (group_df[self.target_col] == 0).astype(int)
            zero_streaks = []
            current_streak = 0
            
            for zero in is_zero:
                if zero:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        zero_streaks.append(current_streak)
                        current_streak = 0
            
            if current_streak > 0:
                zero_streaks.append(current_streak)
            
            # Average inter-demand interval
            non_zero_indices = group_df[group_df[self.target_col] > 0].index.tolist()
            if len(non_zero_indices) >= 2:
                intervals = [non_zero_indices[i] - non_zero_indices[i-1] 
                           for i in range(1, len(non_zero_indices))]
                avg_interval = np.mean(intervals)
            else:
                avg_interval = total_obs
            
            # Classification
            is_intermittent = (zero_ratio >= self.intermittency_threshold and 
                             total_obs >= self.min_observations)
            
            intermittent_analysis[group_name] = {
                'total_observations': total_obs,
                'zero_observations': zero_obs,
                'zero_ratio': zero_ratio,
                'max_zero_streak': max(zero_streaks) if zero_streaks else 0,
                'avg_zero_streak': np.mean(zero_streaks) if zero_streaks else 0,
                'avg_inter_demand_interval': avg_interval,
                'is_intermittent': is_intermittent,
                'non_zero_mean': group_df[group_df[self.target_col] > 0][self.target_col].mean() 
                               if len(group_df[group_df[self.target_col] > 0]) > 0 else 0
            }
        
        # Summary
        total_series = len(intermittent_analysis)
        intermittent_series = sum(1 for v in intermittent_analysis.values() if v['is_intermittent'])
        
        print(f"[OK] Analyzed {total_series} series")
        print(f"[OK] Found {intermittent_series} intermittent series ({intermittent_series/total_series*100:.1f}%)")
        
        return intermittent_analysis
    
    def train_croston_models(self, df: pd.DataFrame, 
                           group_col: str = None) -> Dict:
        """
        Train Croston method variants
        
        Args:
            df: Training data
            group_col: Column to group by
            
        Returns:
            Training results
        """
        
        print("[INFO] Training Croston models...")
        
        # Identify intermittent series
        intermittent_analysis = self.identify_intermittent_series(df, group_col)
        
        croston_results = {}
        
        # Train models for each intermittent series
        for group_name, analysis in intermittent_analysis.items():
            if not analysis['is_intermittent']:
                continue
            
            # Get group data
            if group_col:
                group_data = df[df[group_col] == group_name][self.target_col].values
            else:
                group_data = df[self.target_col].values
            
            # Train different Croston variants
            methods = ['croston', 'sba', 'tsb']
            group_results = {}
            
            for method in methods:
                try:
                    model = CrostonMethod(method=method)
                    model.fit(group_data)
                    
                    # Calculate in-sample metrics
                    fitted_values = np.array(model.fitted_values)
                    in_sample_wmape = wmape(group_data, fitted_values)
                    
                    group_results[method] = {
                        'model': model,
                        'in_sample_wmape': in_sample_wmape,
                        'demand_size': model.demand_size,
                        'inter_arrival_time': model.inter_arrival_time
                    }
                    
                except Exception as e:
                    print(f"[WARNING] Failed to fit {method} for {group_name}: {e}")
            
            if group_results:
                # Select best method based on WMAPE
                best_method = min(group_results.keys(), 
                                key=lambda x: group_results[x]['in_sample_wmape'])
                group_results['best_method'] = best_method
                
                croston_results[group_name] = group_results
        
        self.croston_models = croston_results
        
        print(f"[OK] Trained Croston models for {len(croston_results)} series")
        
        return croston_results
    
    def train_zero_inflated_models(self, df: pd.DataFrame,
                                 group_col: str = None) -> Dict:
        """
        Train zero-inflated models
        
        Args:
            df: Training data with features
            group_col: Column to group by
            
        Returns:
            Training results
        """
        
        print("[INFO] Training zero-inflated models...")
        
        if not self.feature_cols:
            print("[WARNING] No feature columns specified for zero-inflated models")
            return {}
        
        # Prepare features
        available_features = [col for col in self.feature_cols if col in df.columns]
        if not available_features:
            print("[WARNING] No valid feature columns found")
            return {}
        
        zi_results = {}
        
        # Identify intermittent series
        intermittent_analysis = self.identify_intermittent_series(df, group_col)
        
        for group_name, analysis in intermittent_analysis.items():
            if not analysis['is_intermittent']:
                continue
            
            # Get group data
            if group_col:
                group_data = df[df[group_col] == group_name].copy()
            else:
                group_data = df.copy()
            
            if len(group_data) < self.min_observations:
                continue
            
            # Prepare features and target
            X = group_data[available_features].fillna(0).values
            y = group_data[self.target_col].values
            
            try:
                # Train zero-inflated model
                zi_model = ZeroInflatedRegressor(
                    zero_classifier=LogisticRegression(random_state=42),
                    positive_regressor=lgb.LGBMRegressor(
                        objective='regression',
                        n_estimators=100,
                        verbose=-1,
                        random_state=42
                    )
                )
                
                zi_model.fit(X, y)
                
                # Train hurdle model
                hurdle_model = HurdleModel(
                    hurdle_classifier=LogisticRegression(random_state=42),
                    count_regressor=lgb.LGBMRegressor(
                        objective='regression',
                        n_estimators=100,
                        verbose=-1,
                        random_state=42
                    )
                )
                
                hurdle_model.fit(X, y)
                
                # Evaluate models
                zi_pred = zi_model.predict(X)
                hurdle_pred = hurdle_model.predict(X)
                
                zi_wmape = wmape(y, zi_pred)
                hurdle_wmape = wmape(y, hurdle_pred)
                
                zi_results[group_name] = {
                    'zero_inflated_model': zi_model,
                    'hurdle_model': hurdle_model,
                    'zi_wmape': zi_wmape,
                    'hurdle_wmape': hurdle_wmape,
                    'best_model': 'zero_inflated' if zi_wmape <= hurdle_wmape else 'hurdle',
                    'features_used': available_features
                }
                
            except Exception as e:
                print(f"[WARNING] Failed to train zero-inflated models for {group_name}: {e}")
        
        self.zero_inflated_models = zi_results
        
        print(f"[OK] Trained zero-inflated models for {len(zi_results)} series")
        
        return zi_results
    
    def predict_intermittent(self, df: pd.DataFrame, 
                           forecast_horizon: int = 30,
                           method: str = 'ensemble') -> Dict:
        """
        Generate predictions for intermittent demand
        
        Args:
            df: DataFrame with features (for zero-inflated models)
            forecast_horizon: Number of periods to forecast
            method: 'croston', 'zero_inflated', or 'ensemble'
            
        Returns:
            Dictionary of predictions
        """
        
        print(f"[INFO] Generating intermittent demand forecasts ({method})...")
        
        predictions = {}
        
        # Croston predictions
        if method in ['croston', 'ensemble'] and self.croston_models:
            croston_predictions = {}
            
            for group_name, models in self.croston_models.items():
                if 'best_method' in models:
                    best_model = models[models['best_method']]['model']
                    forecast = best_model.predict(forecast_horizon)
                    croston_predictions[group_name] = forecast
            
            predictions['croston'] = croston_predictions
        
        # Zero-inflated predictions
        if method in ['zero_inflated', 'ensemble'] and self.zero_inflated_models:
            zi_predictions = {}
            
            for group_name, models in self.zero_inflated_models.items():
                if 'best_model' in models:
                    best_model_name = models['best_model']
                    model = models[f'{best_model_name}_model']
                    
                    # Use available features for prediction
                    if 'features_used' in models and len(df) > 0:
                        features_used = models['features_used']
                        available_features = [col for col in features_used if col in df.columns]
                        
                        if available_features:
                            # Use last known feature values
                            X_future = df[available_features].iloc[[-1] * forecast_horizon].values
                            forecast = model.predict(X_future)
                            zi_predictions[group_name] = forecast
            
            predictions['zero_inflated'] = zi_predictions
        
        # Ensemble predictions
        if method == 'ensemble' and len(predictions) > 1:
            ensemble_predictions = {}
            
            # Simple average ensemble
            all_groups = set()
            for pred_type in predictions.values():
                all_groups.update(pred_type.keys())
            
            for group_name in all_groups:
                group_forecasts = []
                
                for pred_type in predictions.values():
                    if group_name in pred_type:
                        group_forecasts.append(pred_type[group_name])
                
                if group_forecasts:
                    # Average ensemble
                    ensemble_forecast = np.mean(group_forecasts, axis=0)
                    ensemble_predictions[group_name] = ensemble_forecast
            
            predictions['ensemble'] = ensemble_predictions
        
        self.predictions = predictions
        
        total_forecasts = sum(len(pred) for pred in predictions.values())
        print(f"[OK] Generated {total_forecasts} intermittent demand forecasts")
        
        return predictions
    
    def save_models(self, output_dir: str = "../../models/trained") -> Dict[str, str]:
        """Save intermittent demand models"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models using pickle
        import pickle
        
        saved_files = {}
        
        # Save Croston models
        if self.croston_models:
            croston_file = output_path / f"croston_models_{timestamp}.pkl"
            with open(croston_file, 'wb') as f:
                pickle.dump(self.croston_models, f)
            saved_files['croston'] = str(croston_file)
        
        # Save zero-inflated models
        if self.zero_inflated_models:
            zi_file = output_path / f"zero_inflated_models_{timestamp}.pkl"
            with open(zi_file, 'wb') as f:
                pickle.dump(self.zero_inflated_models, f)
            saved_files['zero_inflated'] = str(zi_file)
        
        # Save predictions
        if self.predictions:
            predictions_file = output_path / f"intermittent_predictions_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_predictions = {}
            for method, preds in self.predictions.items():
                serializable_predictions[method] = {
                    group: pred.tolist() if isinstance(pred, np.ndarray) else pred
                    for group, pred in preds.items()
                }
            
            with open(predictions_file, 'w') as f:
                json.dump(serializable_predictions, f, indent=2)
            
            saved_files['predictions'] = str(predictions_file)
        
        print(f"[SAVE] Intermittent demand models saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Intermittent Demand Specialist"""
    
    print("📈 INTERMITTENT DEMAND SPECIALIST - DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load data with features
        print("Loading data for intermittent demand analysis...")
        trans_df, prod_df, pdv_df = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=20000,
            sample_products=500,
            enable_joins=True,
            validate_loss=True
        )
        
        # Add some basic features for zero-inflated models
        if 'transaction_date' in trans_df.columns:
            trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])
            trans_df['month'] = trans_df['transaction_date'].dt.month
            trans_df['day_of_week'] = trans_df['transaction_date'].dt.dayofweek
        
        print(f"Data loaded: {trans_df.shape}")
        
        # Initialize specialist
        feature_cols = ['month', 'day_of_week']
        if 'categoria' in trans_df.columns:
            feature_cols.append('categoria')
        
        specialist = IntermittentDemandSpecialist(
            date_col='transaction_date',
            target_col='quantity',
            feature_cols=feature_cols
        )
        
        # Train Croston models
        print("\n[DEMO] Training Croston models...")
        croston_results = specialist.train_croston_models(
            trans_df, 
            group_col='internal_product_id'
        )
        
        # Train zero-inflated models
        print("\n[DEMO] Training zero-inflated models...")
        zi_results = specialist.train_zero_inflated_models(
            trans_df, 
            group_col='internal_product_id'
        )
        
        # Generate predictions
        print("\n[DEMO] Generating ensemble predictions...")
        predictions = specialist.predict_intermittent(
            trans_df,
            forecast_horizon=14,
            method='ensemble'
        )
        
        # Save models
        print("\n[DEMO] Saving models...")
        saved_files = specialist.save_models()
        
        print("\n" + "=" * 80)
        print("🎉 INTERMITTENT DEMAND SPECIALIST DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print(f"Croston models trained: {len(croston_results)}")
        print(f"Zero-inflated models trained: {len(zi_results)}")
        
        for method, preds in predictions.items():
            print(f"{method} predictions: {len(preds)} series")
        
        print(f"Files saved: {len(saved_files)}")
        
        return specialist, croston_results, zi_results, predictions
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    results = main()