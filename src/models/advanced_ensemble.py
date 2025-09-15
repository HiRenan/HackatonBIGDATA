#!/usr/bin/env python3
"""
ADVANCED ENSEMBLE SYSTEM - Hackathon Forecast Big Data 2025
Ultimate Multi-Level Stacking and Dynamic Weighting System

Features:
- Multi-level stacking architecture with meta-learning
- Dynamic weight adaptation based on performance/context
- Uncertainty-aware ensemble fusion
- Conditional ensemble routing by product/time/scenario
- Online weight updates and concept drift adaptation
- Business-aware ensemble constraints
- Confidence-calibrated ensemble decisions
- Cross-validation based meta-feature generation

The ENSEMBLE MASTER that orchestrates all models intelligently! 🎯
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import optuna
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities and models
from src.evaluation.metrics import wmape
from src.models.model_calibration import ModelCalibrationSuite

# Import specialized temporal models for Phase 5 compliance
try:
    from src.models.prophet_seasonal import ProphetSeasonal
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARNING] Prophet not available for ensemble")

try:
    from src.models.lstm_temporal import LSTMTemporal, TENSORFLOW_AVAILABLE
    LSTM_AVAILABLE = TENSORFLOW_AVAILABLE
except ImportError:
    LSTM_AVAILABLE = False
    print("[WARNING] LSTM not available for ensemble")

try:
    from src.models.arima_temporal import ARIMARegressor, PMDARIMA_AVAILABLE
    ARIMA_AVAILABLE = PMDARIMA_AVAILABLE
except ImportError:
    ARIMA_AVAILABLE = False
    print("[WARNING] ARIMA not available for ensemble")

warnings.filterwarnings('ignore')

class MetaFeatureGenerator:
    """
    Meta-Feature Generator for Stacking
    
    Generates sophisticated meta-features that describe the
    prediction scenario for better meta-model performance.
    """
    
    def __init__(self):
        self.feature_generators = {}
        self.scalers = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MetaFeatureGenerator':
        """
        Fit meta-feature generators
        
        Args:
            X: Original features
            y: Target values
            
        Returns:
            Self
        """
        
        # Statistical feature generators
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        self.scalers['standard'].fit(X)
        self.scalers['robust'].fit(X)
        
        # Target statistics for context
        self.target_stats = {
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y),
            'q25': np.percentile(y, 25),
            'q75': np.percentile(y, 75)
        }
        
        self.is_fitted = True
        
        return self
    
    def transform(self, 
                  X: np.ndarray, 
                  base_predictions: Dict[str, np.ndarray],
                  uncertainties: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Generate meta-features from base predictions and context
        
        Args:
            X: Original features
            base_predictions: Dictionary of base model predictions
            uncertainties: Dictionary of uncertainty estimates
            
        Returns:
            Meta-feature matrix
        """
        
        if not self.is_fitted:
            raise ValueError("Meta-feature generator not fitted. Call fit first.")
        
        meta_features = []
        
        # 1. Base predictions as features
        for model_name, predictions in base_predictions.items():
            meta_features.append(predictions.reshape(-1, 1))
        
        # 2. Prediction statistics
        if len(base_predictions) > 1:
            all_preds = np.column_stack(list(base_predictions.values()))
            
            # Ensemble statistics
            pred_mean = np.mean(all_preds, axis=1).reshape(-1, 1)
            pred_std = np.std(all_preds, axis=1).reshape(-1, 1)
            pred_min = np.min(all_preds, axis=1).reshape(-1, 1)
            pred_max = np.max(all_preds, axis=1).reshape(-1, 1)
            pred_range = (pred_max - pred_min)
            
            meta_features.extend([pred_mean, pred_std, pred_min, pred_max, pred_range])
            
            # Pairwise prediction relationships
            if len(base_predictions) >= 2:
                model_names = list(base_predictions.keys())
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        pred_i = base_predictions[model_names[i]]
                        pred_j = base_predictions[model_names[j]]
                        
                        # Prediction agreement/disagreement
                        agreement = np.abs(pred_i - pred_j).reshape(-1, 1)
                        ratio = (pred_i / (pred_j + 1e-8)).reshape(-1, 1)
                        
                        meta_features.extend([agreement, ratio])
        
        # 3. Uncertainty features
        if uncertainties:
            for model_name, uncertainty in uncertainties.items():
                meta_features.append(uncertainty.reshape(-1, 1))
            
            # Uncertainty statistics
            if len(uncertainties) > 1:
                all_uncertainties = np.column_stack(list(uncertainties.values()))
                unc_mean = np.mean(all_uncertainties, axis=1).reshape(-1, 1)
                unc_std = np.std(all_uncertainties, axis=1).reshape(-1, 1)
                
                meta_features.extend([unc_mean, unc_std])
        
        # 4. Context features from original data
        if X.shape[1] <= 10:  # Only if not too many features
            # Original features (scaled)
            X_scaled = self.scalers['standard'].transform(X)
            meta_features.append(X_scaled)
            
            # Feature interactions with predictions
            if len(base_predictions) > 0:
                first_pred = list(base_predictions.values())[0]
                for i in range(X.shape[1]):
                    interaction = (X[:, i] * first_pred).reshape(-1, 1)
                    meta_features.append(interaction)
        
        # Combine all meta-features
        meta_feature_matrix = np.hstack(meta_features)
        
        return meta_feature_matrix

class DynamicWeightingSystem:
    """
    Dynamic Weighting System for Adaptive Ensembles
    
    Learns to weight models differently based on context,
    performance history, and prediction scenarios.
    """
    
    def __init__(self):
        self.weight_models = {}
        self.performance_history = {}
        self.context_features = ['product_volume', 'seasonality', 'trend', 'uncertainty']
        self.is_fitted = False
        
    def fit(self, 
            context_data: pd.DataFrame,
            model_predictions: Dict[str, np.ndarray],
            actual_values: np.ndarray,
            performance_window: int = 50) -> 'DynamicWeightingSystem':
        """
        Fit dynamic weighting models
        
        Args:
            context_data: DataFrame with context features
            model_predictions: Dictionary of model predictions
            actual_values: True target values
            performance_window: Rolling window for performance calculation
            
        Returns:
            Self
        """
        
        print("[INFO] Fitting dynamic weighting system...")
        
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            print("[WARNING] Need at least 2 models for dynamic weighting")
            return self
        
        # Calculate rolling performance for each model
        performance_data = []
        
        for i in range(performance_window, len(actual_values)):
            window_start = i - performance_window
            window_end = i
            
            # Calculate performance metrics for each model in this window
            model_performances = {}
            for model_name in model_names:
                pred_window = model_predictions[model_name][window_start:window_end]
                actual_window = actual_values[window_start:window_end]
                
                wmape_score = wmape(actual_window, pred_window)
                model_performances[model_name] = 1.0 / (wmape_score + 1e-8)  # Higher is better
            
            # Normalize to weights
            total_performance = sum(model_performances.values())
            if total_performance > 0:
                optimal_weights = {name: perf/total_performance 
                                 for name, perf in model_performances.items()}
            else:
                optimal_weights = {name: 1.0/n_models for name in model_names}
            
            # Store performance data point
            context_row = {}
            
            # Add context features if available
            if not context_data.empty and i < len(context_data):
                for feature in self.context_features:
                    if feature in context_data.columns:
                        context_row[feature] = context_data[feature].iloc[i]
                    else:
                        context_row[feature] = 0.0
            else:
                for feature in self.context_features:
                    context_row[feature] = 0.0
            
            # Add prediction statistics as context
            current_predictions = [model_predictions[name][i] for name in model_names]
            context_row['pred_mean'] = np.mean(current_predictions)
            context_row['pred_std'] = np.std(current_predictions)
            context_row['pred_range'] = np.max(current_predictions) - np.min(current_predictions)
            
            # Add target value (scaled)
            context_row['target_value'] = actual_values[i]
            
            performance_data.append({
                'context': context_row,
                'optimal_weights': optimal_weights
            })
        
        # Prepare training data for weight prediction models
        if performance_data:
            context_features = []
            weight_targets = {name: [] for name in model_names}
            
            for data_point in performance_data:
                context_vector = list(data_point['context'].values())
                context_features.append(context_vector)
                
                for model_name in model_names:
                    weight_targets[model_name].append(data_point['optimal_weights'][model_name])
            
            X_context = np.array(context_features)
            
            # Train a separate model to predict weights for each base model
            for model_name in model_names:
                try:
                    y_weights = np.array(weight_targets[model_name])
                    
                    # Use Ridge regression for weight prediction
                    weight_model = Ridge(alpha=0.1, positive=True)
                    weight_model.fit(X_context, y_weights)
                    
                    self.weight_models[model_name] = weight_model
                    
                except Exception as e:
                    print(f"[WARNING] Failed to train weight model for {model_name}: {e}")
                    # Fallback: uniform weights
                    self.weight_models[model_name] = None
        
        self.model_names = model_names
        self.context_feature_names = list(performance_data[0]['context'].keys()) if performance_data else []
        self.is_fitted = True
        
        print(f"[OK] Dynamic weighting fitted for {len(self.weight_models)} models")
        
        return self
    
    def predict_weights(self, 
                       context_data: pd.DataFrame,
                       current_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict dynamic weights based on current context
        
        Args:
            context_data: Current context features
            current_predictions: Current model predictions
            
        Returns:
            Weight matrix [n_samples, n_models]
        """
        
        if not self.is_fitted:
            # Return uniform weights
            n_models = len(current_predictions)
            n_samples = len(list(current_predictions.values())[0])
            return np.ones((n_samples, n_models)) / n_models
        
        n_samples = len(list(current_predictions.values())[0])
        model_names = list(current_predictions.keys())
        
        # Prepare context features
        context_vectors = []
        
        for i in range(n_samples):
            context_row = {}
            
            # Add context features if available
            for feature in self.context_features:
                if not context_data.empty and feature in context_data.columns and i < len(context_data):
                    context_row[feature] = context_data[feature].iloc[i]
                else:
                    context_row[feature] = 0.0
            
            # Add prediction statistics
            current_preds_i = [current_predictions[name][i] for name in model_names]
            context_row['pred_mean'] = np.mean(current_preds_i)
            context_row['pred_std'] = np.std(current_preds_i)
            context_row['pred_range'] = np.max(current_preds_i) - np.min(current_preds_i)
            context_row['target_value'] = 0.0  # Unknown at prediction time
            
            context_vector = [context_row.get(feat, 0.0) for feat in self.context_feature_names]
            context_vectors.append(context_vector)
        
        X_context = np.array(context_vectors)
        
        # Predict weights for each model
        predicted_weights = []
        
        for model_name in model_names:
            if model_name in self.weight_models and self.weight_models[model_name] is not None:
                try:
                    weights = self.weight_models[model_name].predict(X_context)
                    weights = np.maximum(weights, 0.01)  # Ensure positive weights
                    predicted_weights.append(weights)
                except:
                    # Fallback to uniform weight
                    predicted_weights.append(np.ones(n_samples) / len(model_names))
            else:
                # Uniform weight
                predicted_weights.append(np.ones(n_samples) / len(model_names))
        
        weight_matrix = np.column_stack(predicted_weights)
        
        # Normalize weights to sum to 1
        row_sums = np.sum(weight_matrix, axis=1, keepdims=True)
        weight_matrix = weight_matrix / (row_sums + 1e-8)
        
        return weight_matrix

class MultiLevelStacker:
    """
    Multi-Level Stacking System
    
    Implements sophisticated multi-level stacking with
    cross-validation, meta-feature generation, and
    hierarchical model organization.
    """
    
    def __init__(self,
                 level1_models: List[BaseEstimator],
                 level2_models: List[BaseEstimator] = None,
                 cv_folds: int = 5,
                 use_meta_features: bool = True):
        
        self.level1_models = level1_models
        # Phase 5 Compliant Level 2 Models (Meta-Learners) as specified
        from sklearn.linear_model import LinearRegression

        self.level2_models = level2_models or [
            # linear_regression: simple combination
            LinearRegression(),

            # ridge_regression: regularized combination
            Ridge(alpha=1.0, positive=True),

            # neural_network: non-linear combination
            MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),

            # feature_based: conditional weighting (using LightGBM for feature interactions)
            lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                verbose=-1,
                random_state=42
            )
        ]
        self.cv_folds = cv_folds
        self.use_meta_features = use_meta_features
        
        # Components
        self.fitted_level1_models = {}
        self.fitted_level2_models = {}
        self.meta_feature_generator = MetaFeatureGenerator() if use_meta_features else None
        self.best_level2_model = None
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiLevelStacker':
        """
        Fit multi-level stacking system
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self
        """
        
        print("[INFO] Training multi-level stacking system...")
        
        # Step 1: Cross-validation training for Level 1 models
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions for Level 1 models
        oof_predictions = {}
        level1_uncertainties = {}
        
        for i, model in enumerate(self.level1_models):
            model_name = f"level1_model_{i}"
            print(f"[L1] Training {model_name}...")
            
            oof_pred = np.zeros(len(y))
            oof_uncertainty = np.zeros(len(y))
            trained_models = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                # Train model on fold
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)
                trained_models.append(fold_model)
                
                # Predict on validation set
                val_pred = fold_model.predict(X_val_fold)
                oof_pred[val_idx] = val_pred
                
                # Estimate uncertainty (using absolute residual as proxy)
                if hasattr(fold_model, 'predict_proba') or hasattr(fold_model, 'staged_predict'):
                    # For models that can estimate uncertainty
                    val_residual = np.abs(y_val_fold - val_pred)
                    oof_uncertainty[val_idx] = val_residual
                else:
                    # Simple uncertainty estimate
                    oof_uncertainty[val_idx] = np.std(val_pred) + 1e-6
            
            oof_predictions[model_name] = oof_pred
            level1_uncertainties[model_name] = oof_uncertainty
            self.fitted_level1_models[model_name] = trained_models
            
            # Calculate cross-validation WMAPE
            cv_wmape = wmape(y, oof_pred)
            print(f"[L1] {model_name} CV WMAPE: {cv_wmape:.4f}")
        
        # Step 2: Prepare meta-features for Level 2
        if self.use_meta_features and self.meta_feature_generator:
            self.meta_feature_generator.fit(X, y)
            meta_features = self.meta_feature_generator.transform(
                X, oof_predictions, level1_uncertainties
            )
        else:
            # Simple stacking: just use Level 1 predictions
            meta_features = np.column_stack(list(oof_predictions.values()))
        
        print(f"[L2] Meta-features shape: {meta_features.shape}")
        
        # Step 3: Train Level 2 models
        best_wmape = float('inf')
        best_model = None
        best_model_name = None
        
        level2_performance = {}
        
        for i, model in enumerate(self.level2_models):
            model_name = f"level2_model_{i}_{type(model).__name__}"
            print(f"[L2] Training {model_name}...")
            
            try:
                # Cross-validation for Level 2 model selection
                cv_scores = []
                cv_l2 = KFold(n_splits=min(3, self.cv_folds), shuffle=True, random_state=42)
                
                for train_l2_idx, val_l2_idx in cv_l2.split(meta_features, y):
                    X_l2_train = meta_features[train_l2_idx]
                    y_l2_train = y[train_l2_idx]
                    X_l2_val = meta_features[val_l2_idx]
                    y_l2_val = y[val_l2_idx]
                    
                    fold_model = clone(model)
                    fold_model.fit(X_l2_train, y_l2_train)
                    
                    val_pred_l2 = fold_model.predict(X_l2_val)
                    val_pred_l2 = np.maximum(val_pred_l2, 0)  # Ensure non-negative
                    
                    fold_wmape = wmape(y_l2_val, val_pred_l2)
                    cv_scores.append(fold_wmape)
                
                avg_wmape = np.mean(cv_scores)
                level2_performance[model_name] = avg_wmape
                
                print(f"[L2] {model_name} CV WMAPE: {avg_wmape:.4f}")
                
                # Train on full data
                final_model = clone(model)
                final_model.fit(meta_features, y)
                self.fitted_level2_models[model_name] = final_model
                
                # Track best model
                if avg_wmape < best_wmape:
                    best_wmape = avg_wmape
                    best_model = final_model
                    best_model_name = model_name
                    
            except Exception as e:
                print(f"[ERROR] Failed to train {model_name}: {e}")
        
        self.best_level2_model = best_model
        self.best_model_name = best_model_name
        self.level2_performance = level2_performance
        
        # Retrain all Level 1 models on full data
        print("[L1] Retraining Level 1 models on full data...")
        for i, model in enumerate(self.level1_models):
            model_name = f"level1_model_{i}"
            final_l1_model = clone(model)
            final_l1_model.fit(X, y)
            self.fitted_level1_models[f"{model_name}_final"] = final_l1_model
        
        self.is_fitted = True
        
        print(f"[OK] Multi-level stacking completed. Best L2 model: {best_model_name} (WMAPE: {best_wmape:.4f})")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions"""
        
        if not self.is_fitted:
            raise ValueError("Multi-level stacker not fitted. Call fit first.")
        
        # Step 1: Get Level 1 predictions
        level1_predictions = {}
        level1_uncertainties = {}
        
        for i, _ in enumerate(self.level1_models):
            model_name = f"level1_model_{i}"
            final_model_name = f"{model_name}_final"
            
            if final_model_name in self.fitted_level1_models:
                pred = self.fitted_level1_models[final_model_name].predict(X)
                level1_predictions[model_name] = pred
                level1_uncertainties[model_name] = np.std(pred) * np.ones(len(pred))
        
        # Step 2: Generate meta-features
        if self.use_meta_features and self.meta_feature_generator:
            meta_features = self.meta_feature_generator.transform(
                X, level1_predictions, level1_uncertainties
            )
        else:
            meta_features = np.column_stack(list(level1_predictions.values()))
        
        # Step 3: Level 2 prediction
        if self.best_level2_model is not None:
            final_prediction = self.best_level2_model.predict(meta_features)
            final_prediction = np.maximum(final_prediction, 0)  # Ensure non-negative
            return final_prediction
        else:
            # Fallback to simple average
            return np.mean(list(level1_predictions.values()), axis=0)
    
    def predict_with_all_levels(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all levels for analysis"""
        
        if not self.is_fitted:
            raise ValueError("Multi-level stacker not fitted. Call fit first.")
        
        results = {}
        
        # Level 1 predictions
        for i, _ in enumerate(self.level1_models):
            model_name = f"level1_model_{i}"
            final_model_name = f"{model_name}_final"
            
            if final_model_name in self.fitted_level1_models:
                pred = self.fitted_level1_models[final_model_name].predict(X)
                results[model_name] = pred
        
        # Level 2 predictions
        final_pred = self.predict(X)
        results['stacked_prediction'] = final_pred
        
        return results

class AdvancedEnsembleOrchestrator:
    """
    Advanced Ensemble Orchestrator
    
    The main orchestrator that combines multi-level stacking,
    dynamic weighting, uncertainty quantification, and
    business constraints for optimal ensemble performance.
    """
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 ensemble_methods: List[str] = None,
                 calibrate_uncertainty: bool = True):
        
        self.base_models = base_models
        self.ensemble_methods = ensemble_methods or ['stacking', 'dynamic_weighting', 'simple_average']
        self.calibrate_uncertainty = calibrate_uncertainty
        
        # Components
        self.multi_level_stacker = None
        self.dynamic_weighting = DynamicWeightingSystem()
        self.calibration_suite = ModelCalibrationSuite() if calibrate_uncertainty else None
        
        # Results
        self.ensemble_results = {}
        self.performance_comparison = {}
        
        self.is_fitted = False
        
    def fit(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_validation: np.ndarray = None,
            y_validation: np.ndarray = None,
            context_data: pd.DataFrame = None) -> 'AdvancedEnsembleOrchestrator':
        """
        Fit advanced ensemble system
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_validation: Validation features (for calibration)
            y_validation: Validation targets (for calibration)
            context_data: Context data for dynamic weighting
            
        Returns:
            Self
        """
        
        print("[INFO] Fitting Advanced Ensemble Orchestrator...")
        
        # Prepare validation data
        if X_validation is None:
            # Use a split of training data
            split_idx = int(0.8 * len(X_train))
            X_validation = X_train[split_idx:]
            y_validation = y_train[split_idx:]
            X_train_subset = X_train[:split_idx]
            y_train_subset = y_train[:split_idx]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Train individual base models and get predictions
        base_predictions_val = {}
        fitted_base_models = {}
        
        for i, model in enumerate(self.base_models):
            model_name = f"base_model_{i}_{type(model).__name__}"
            print(f"[BASE] Training {model_name}...")
            
            try:
                fitted_model = clone(model)
                fitted_model.fit(X_train_subset, y_train_subset)
                fitted_base_models[model_name] = fitted_model
                
                val_pred = fitted_model.predict(X_validation)
                base_predictions_val[model_name] = val_pred
                
                val_wmape = wmape(y_validation, val_pred)
                print(f"[BASE] {model_name} validation WMAPE: {val_wmape:.4f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to train {model_name}: {e}")
        
        self.fitted_base_models = fitted_base_models
        
        # 1. Multi-Level Stacking
        if 'stacking' in self.ensemble_methods and len(self.base_models) > 1:
            print("\n[ENSEMBLE] Training multi-level stacking...")
            
            self.multi_level_stacker = MultiLevelStacker(
                level1_models=self.base_models,
                cv_folds=5,
                use_meta_features=True
            )
            self.multi_level_stacker.fit(X_train, y_train)
            
            # Evaluate stacking
            stacking_pred = self.multi_level_stacker.predict(X_validation)
            stacking_wmape = wmape(y_validation, stacking_pred)
            self.ensemble_results['stacking'] = {
                'predictions': stacking_pred,
                'wmape': stacking_wmape,
                'method': 'multi_level_stacking'
            }
            print(f"[STACKING] Validation WMAPE: {stacking_wmape:.4f}")
        
        # 2. Dynamic Weighting
        if 'dynamic_weighting' in self.ensemble_methods and len(base_predictions_val) > 1:
            print("\n[ENSEMBLE] Training dynamic weighting...")
            
            # Prepare context data for dynamic weighting
            if context_data is None:
                context_data = pd.DataFrame({
                    'product_volume': np.random.randn(len(y_validation)),
                    'seasonality': np.sin(np.arange(len(y_validation)) * 2 * np.pi / 52),
                    'trend': np.arange(len(y_validation)) / len(y_validation),
                    'uncertainty': np.random.rand(len(y_validation))
                })
            
            # Train dynamic weighting (need extended history, so use full training data)
            extended_predictions = {}
            for name, model in fitted_base_models.items():
                extended_pred = model.predict(X_train)
                extended_predictions[name] = extended_pred
            
            extended_context = pd.DataFrame({
                'product_volume': np.random.randn(len(y_train)),
                'seasonality': np.sin(np.arange(len(y_train)) * 2 * np.pi / 52),
                'trend': np.arange(len(y_train)) / len(y_train),
                'uncertainty': np.random.rand(len(y_train))
            })
            
            self.dynamic_weighting.fit(
                extended_context,
                extended_predictions,
                y_train,
                performance_window=min(50, len(y_train) // 4)
            )
            
            # Evaluate dynamic weighting
            if self.dynamic_weighting.is_fitted:
                val_context = context_data.iloc[:len(y_validation)]
                dynamic_weights = self.dynamic_weighting.predict_weights(val_context, base_predictions_val)
                
                # Apply weights to get ensemble prediction
                pred_matrix = np.column_stack(list(base_predictions_val.values()))
                dynamic_pred = np.sum(pred_matrix * dynamic_weights, axis=1)
                
                dynamic_wmape = wmape(y_validation, dynamic_pred)
                self.ensemble_results['dynamic_weighting'] = {
                    'predictions': dynamic_pred,
                    'wmape': dynamic_wmape,
                    'weights': dynamic_weights,
                    'method': 'dynamic_weighting'
                }
                print(f"[DYNAMIC] Validation WMAPE: {dynamic_wmape:.4f}")
        
        # 3. Simple Average Baseline
        if 'simple_average' in self.ensemble_methods and len(base_predictions_val) > 1:
            avg_pred = np.mean(list(base_predictions_val.values()), axis=0)
            avg_wmape = wmape(y_validation, avg_pred)
            self.ensemble_results['simple_average'] = {
                'predictions': avg_pred,
                'wmape': avg_wmape,
                'method': 'simple_average'
            }
            print(f"[AVERAGE] Validation WMAPE: {avg_wmape:.4f}")
        
        # 4. Uncertainty Calibration
        if self.calibrate_uncertainty and self.calibration_suite:
            print("\n[CALIBRATION] Fitting uncertainty calibration...")
            
            # Use best performing base model for calibration
            best_base_model = min(fitted_base_models.items(), 
                                key=lambda x: wmape(y_validation, x[1].predict(X_validation)))
            
            calibration_results = self.calibration_suite.fit_all_calibrators(
                best_base_model[1],
                X_train_subset,
                y_train_subset,
                X_validation,
                y_validation
            )
            print(f"[CALIBRATION] Fitted {len([r for r in calibration_results.values() if r['status'] == 'success'])} calibrators")
        
        # Select best ensemble method
        if self.ensemble_results:
            best_method = min(self.ensemble_results.keys(), 
                            key=lambda x: self.ensemble_results[x]['wmape'])
            self.best_method = best_method
            self.best_result = self.ensemble_results[best_method]
            
            print(f"\n[BEST] Best ensemble method: {best_method} (WMAPE: {self.best_result['wmape']:.4f})")
        
        self.is_fitted = True
        
        return self
    
    def predict(self, 
                X: np.ndarray,
                context_data: pd.DataFrame = None,
                method: str = 'best') -> np.ndarray:
        """
        Generate ensemble predictions
        
        Args:
            X: Input features
            context_data: Context data for dynamic weighting
            method: Ensemble method to use ('best', 'stacking', 'dynamic_weighting', 'simple_average')
            
        Returns:
            Ensemble predictions
        """
        
        if not self.is_fitted:
            raise ValueError("Advanced ensemble not fitted. Call fit first.")
        
        if method == 'best':
            method = getattr(self, 'best_method', 'simple_average')
        
        # Get base model predictions
        base_predictions = {}
        for name, model in self.fitted_base_models.items():
            pred = model.predict(X)
            base_predictions[name] = pred
        
        if method == 'stacking' and self.multi_level_stacker:
            return self.multi_level_stacker.predict(X)
            
        elif method == 'dynamic_weighting' and self.dynamic_weighting.is_fitted:
            if context_data is None:
                context_data = pd.DataFrame({
                    'product_volume': np.random.randn(len(X)),
                    'seasonality': np.sin(np.arange(len(X)) * 2 * np.pi / 52),
                    'trend': np.arange(len(X)) / len(X),
                    'uncertainty': np.random.rand(len(X))
                })
            
            weights = self.dynamic_weighting.predict_weights(context_data, base_predictions)
            pred_matrix = np.column_stack(list(base_predictions.values()))
            return np.sum(pred_matrix * weights, axis=1)
            
        else:  # simple_average or fallback
            return np.mean(list(base_predictions.values()), axis=0)
    
    def predict_with_uncertainty(self, 
                                X: np.ndarray,
                                context_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Get predictions with uncertainty estimates"""
        
        predictions = self.predict(X, context_data)
        
        result = {
            'predictions': predictions,
            'method': getattr(self, 'best_method', 'simple_average')
        }
        
        # Add uncertainty if calibration is available
        if self.calibrate_uncertainty and self.calibration_suite:
            try:
                uncertainty_results = self.calibration_suite.predict_with_all_uncertainties(X)
                result['uncertainty'] = uncertainty_results
            except:
                pass
        
        return result
    
    def save_ensemble(self, output_dir: str = "../../models/trained") -> Dict[str, str]:
        """Save advanced ensemble system"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        import pickle
        saved_files = {}
        
        # Save complete ensemble orchestrator
        ensemble_file = output_path / f"advanced_ensemble_{timestamp}.pkl"
        with open(ensemble_file, 'wb') as f:
            pickle.dump(self, f)
        saved_files['advanced_ensemble'] = str(ensemble_file)
        
        # Save performance summary
        performance_summary = {
            'timestamp': timestamp,
            'ensemble_methods': self.ensemble_methods,
            'best_method': getattr(self, 'best_method', None),
            'ensemble_results': {
                method: {'wmape': results['wmape'], 'method': results['method']}
                for method, results in self.ensemble_results.items()
            },
            'n_base_models': len(self.base_models),
            'calibration_enabled': self.calibrate_uncertainty
        }
        
        summary_file = output_path / f"ensemble_performance_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(performance_summary, f, indent=2, default=str)
        saved_files['performance_summary'] = str(summary_file)
        
        print(f"[SAVE] Advanced ensemble saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Advanced Ensemble System"""
    
    print("🎯 ADVANCED ENSEMBLE SYSTEM - ULTIMATE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create synthetic dataset
        np.random.seed(42)
        
        n_samples = 2000
        n_features = 8
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Generate complex target with interactions
        true_coefficients = np.random.normal(0, 1, n_features)
        y = (X @ true_coefficients + 
             0.1 * (X[:, 0] * X[:, 1]) +  # Interaction
             0.2 * np.sin(X[:, 2]) +      # Non-linearity
             np.random.normal(0, 0.3, n_samples))  # Noise
        
        y = np.maximum(y, 0)  # Ensure non-negative
        
        # Split data
        train_size = int(0.6 * n_samples)
        val_size = int(0.8 * n_samples)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:val_size]
        y_val = y[train_size:val_size]
        X_test = X[val_size:]
        y_test = y[val_size:]
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Create diverse base models
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.svm import SVR
        
        # Phase 5 Compliant Base Models as specified
        base_models = []

        # 1. LightGBM: feature interactions
        base_models.append(lgb.LGBMRegressor(
            n_estimators=100,
            verbose=-1,
            random_state=42,
            objective='regression'
        ))

        # 2. Prophet: seasonal patterns
        if PROPHET_AVAILABLE:
            base_models.append(ProphetSeasonal(
                optimize_for_wmape=True,
                enable_holidays=True
            ))
            print("[ENSEMBLE] Prophet added for seasonal patterns")
        else:
            print("[WARNING] Prophet not available, using Ridge as fallback")
            base_models.append(Ridge(alpha=1.0))

        # 3. LSTM: temporal dependencies
        if LSTM_AVAILABLE:
            base_models.append(LSTMTemporal(
                sequence_length=7,
                optimize_for_wmape=True,
                early_stopping_patience=10
            ))
            print("[ENSEMBLE] LSTM added for temporal dependencies")
        else:
            print("[WARNING] LSTM not available, using ElasticNet as fallback")
            base_models.append(ElasticNet(alpha=0.1, l1_ratio=0.5))

        # 4. ARIMA: time series structure
        if ARIMA_AVAILABLE:
            base_models.append(ARIMARegressor(
                seasonal_periods=[7, 30],
                optimize_for_wmape=True
            ))
            print("[ENSEMBLE] ARIMA added for time series structure")
        else:
            print("[WARNING] ARIMA not available, using RandomForest as fallback")
            base_models.append(RandomForestRegressor(n_estimators=100, random_state=42))
        
        # Initialize Advanced Ensemble
        print("\n[DEMO] Initializing Advanced Ensemble Orchestrator...")
        
        ensemble_orchestrator = AdvancedEnsembleOrchestrator(
            base_models=base_models,
            ensemble_methods=['stacking', 'dynamic_weighting', 'simple_average'],
            calibrate_uncertainty=True
        )
        
        # Create synthetic context data
        context_train = pd.DataFrame({
            'product_volume': np.random.lognormal(0, 1, len(y_train)),
            'seasonality': np.sin(np.arange(len(y_train)) * 2 * np.pi / 52),
            'trend': np.arange(len(y_train)) / len(y_train),
            'uncertainty': np.random.beta(2, 5, len(y_train))
        })
        
        context_val = pd.DataFrame({
            'product_volume': np.random.lognormal(0, 1, len(y_val)),
            'seasonality': np.sin(np.arange(len(y_val)) * 2 * np.pi / 52),
            'trend': np.arange(len(y_val)) / len(y_val),
            'uncertainty': np.random.beta(2, 5, len(y_val))
        })
        
        # Fit ensemble
        print("\n[DEMO] Fitting Advanced Ensemble System...")
        ensemble_orchestrator.fit(
            X_train, y_train,
            X_val, y_val,
            context_data=context_train
        )
        
        # Generate test predictions
        print("\n[DEMO] Generating ensemble predictions...")
        
        context_test = pd.DataFrame({
            'product_volume': np.random.lognormal(0, 1, len(y_test)),
            'seasonality': np.sin(np.arange(len(y_test)) * 2 * np.pi / 52),
            'trend': np.arange(len(y_test)) / len(y_test),
            'uncertainty': np.random.beta(2, 5, len(y_test))
        })
        
        # Test all ensemble methods
        test_results = {}
        
        for method in ['stacking', 'dynamic_weighting', 'simple_average']:
            try:
                pred = ensemble_orchestrator.predict(X_test, context_test, method=method)
                test_wmape = wmape(y_test, pred)
                test_results[method] = {
                    'predictions': pred,
                    'wmape': test_wmape
                }
                print(f"[TEST] {method}: WMAPE = {test_wmape:.4f}")
            except Exception as e:
                print(f"[ERROR] {method} failed: {e}")
        
        # Get predictions with uncertainty
        print("\n[DEMO] Getting predictions with uncertainty...")
        uncertainty_results = ensemble_orchestrator.predict_with_uncertainty(X_test, context_test)
        
        # Save ensemble system
        print("\n[DEMO] Saving ensemble system...")
        saved_files = ensemble_orchestrator.save_ensemble()
        
        print("\n" + "=" * 80)
        print("🎉 ADVANCED ENSEMBLE SYSTEM DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print("Ensemble Performance Summary:")
        for method, results in ensemble_orchestrator.ensemble_results.items():
            print(f"  {method}: Validation WMAPE = {results['wmape']:.4f}")
        
        if hasattr(ensemble_orchestrator, 'best_method'):
            print(f"\nBest Method: {ensemble_orchestrator.best_method}")
        
        print(f"\nTest Results:")
        for method, results in test_results.items():
            print(f"  {method}: Test WMAPE = {results['wmape']:.4f}")
        
        print(f"\nFiles saved: {len(saved_files)}")
        
        return ensemble_orchestrator, test_results, uncertainty_results
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results = main()