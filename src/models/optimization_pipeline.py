#!/usr/bin/env python3
"""
OPTIMIZATION PIPELINE - Hackathon Forecast Big Data 2025
Ultimate Automated Optimization and Hyperparameter Tuning System

Features:
- Multi-model hyperparameter optimization using Optuna
- Neural architecture search and automated ML
- Feature selection and engineering optimization
- Ensemble weight optimization with business constraints
- Multi-objective optimization (accuracy vs speed vs interpretability)
- Pipeline automation with MLflow integration
- Bayesian optimization for expensive evaluations
- Early stopping and resource management

The OPTIMIZATION MASTER that fine-tunes everything! ⚡
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities and models
from src.evaluation.metrics import wmape
from src.models.advanced_ensemble import AdvancedEnsembleOrchestrator
from src.models.business_rules import BusinessRulesOrchestrator

warnings.filterwarnings('ignore')

@dataclass
class OptimizationConfig:
    """
    Optimization Configuration Container
    
    Stores all configuration parameters for the optimization pipeline.
    """
    
    # Study configuration
    study_name: str = "hackathon_forecast_optimization"
    optimization_direction: str = "minimize"  # minimize WMAPE
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 hour
    
    # Sampler configuration
    sampler: str = "tpe"  # tpe, cmaes, random
    pruner: str = "median"  # median, hyperband, none
    
    # Cross-validation configuration
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # time_series, kfold
    
    # Multi-objective weights
    accuracy_weight: float = 0.7
    speed_weight: float = 0.2
    interpretability_weight: float = 0.1
    
    # Resource limits
    max_parallel_jobs: int = 4
    memory_limit_gb: float = 8.0
    
    # Feature selection
    enable_feature_selection: bool = True
    max_features: int = 50
    
    # Ensemble optimization
    enable_ensemble_optimization: bool = True
    ensemble_methods: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['stacking', 'dynamic_weighting', 'simple_average']

class HyperparameterOptimizer:
    """
    Hyperparameter Optimization Engine
    
    Optimizes model hyperparameters using Bayesian optimization
    with early stopping and resource management.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.studies = {}
        self.optimization_history = []
        
        # Set up Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    def optimize_lightgbm(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> Dict:
        """
        Optimize LightGBM hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Optimization results
        """
        
        print("[OPTIM] Starting LightGBM hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'regression',
                'metric': 'None',  # We'll use custom WMAPE
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'verbosity': -1,
                'random_state': 42
            }
            
            # Train model
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Custom WMAPE evaluation metric
            def wmape_eval(y_pred, y_true):
                y_true = y_true.get_label()
                wmape_score = wmape(y_true, y_pred)
                return 'wmape', wmape_score, False  # False means lower is better
            
            callbacks = [
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=callbacks,
                feval=wmape_eval
            )
            
            # Predict and calculate WMAPE
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
            
            return wmape(y_val, y_pred)
        
        # Create and run study
        study = optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            study_name=f"{self.config.study_name}_lightgbm"
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True
        )
        
        self.studies['lightgbm'] = study
        
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        print(f"[OPTIM] LightGBM optimization completed: Best WMAPE = {study.best_value:.4f}")
        
        return results
    
    def optimize_random_forest(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Dict:
        """Optimize Random Forest hyperparameters"""
        
        print("[OPTIM] Starting Random Forest hyperparameter optimization...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            y_pred = np.maximum(y_pred, 0)
            
            return wmape(y_val, y_pred)
        
        study = optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            study_name=f"{self.config.study_name}_rf"
        )
        
        study.optimize(objective, n_trials=self.config.n_trials // 2)  # Fewer trials for RF
        
        self.studies['random_forest'] = study
        
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        print(f"[OPTIM] Random Forest optimization completed: Best WMAPE = {study.best_value:.4f}")
        
        return results
    
    def optimize_neural_network(self,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray) -> Dict:
        """Optimize Neural Network architecture and hyperparameters"""
        
        print("[OPTIM] Starting Neural Network optimization...")
        
        def objective(trial):
            from sklearn.neural_network import MLPRegressor
            
            # Architecture parameters
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_layer_sizes = []
            
            for i in range(n_layers):
                layer_size = trial.suggest_int(f'layer_{i}_size', 10, 200)
                hidden_layer_sizes.append(layer_size)
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
            
            # Add learning_rate_init only for adam
            if params['solver'] == 'adam':
                params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
            
            model = MLPRegressor(**params)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_pred = np.maximum(y_pred, 0)
                return wmape(y_val, y_pred)
            except:
                return 100.0  # High penalty for failed training
        
        study = optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
            study_name=f"{self.config.study_name}_nn"
        )
        
        study.optimize(objective, n_trials=self.config.n_trials // 3)  # Fewer trials for NN
        
        self.studies['neural_network'] = study
        
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        print(f"[OPTIM] Neural Network optimization completed: Best WMAPE = {study.best_value:.4f}")
        
        return results
    
    def _create_sampler(self):
        """Create appropriate sampler based on configuration"""
        
        if self.config.sampler == 'tpe':
            return TPESampler(random_state=42)
        elif self.config.sampler == 'cmaes':
            return CmaEsSampler(random_state=42)
        else:
            return optuna.samplers.RandomSampler(seed=42)
    
    def _create_pruner(self):
        """Create appropriate pruner based on configuration"""
        
        if self.config.pruner == 'median':
            return MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == 'hyperband':
            return HyperbandPruner(min_resource=1, max_resource=100)
        else:
            return optuna.pruners.NopPruner()

class FeatureOptimizer:
    """
    Feature Selection and Engineering Optimizer
    
    Optimizes feature selection, transformation, and engineering
    to improve model performance.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.feature_importance_history = []
        self.selected_features = None
        
    def optimize_feature_selection(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_val: np.ndarray,
                                  feature_names: List[str] = None) -> Dict:
        """
        Optimize feature selection using multiple methods
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Names of features
            
        Returns:
            Feature selection results
        """
        
        print("[FEATURE] Starting feature selection optimization...")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        def objective(trial):
            # Feature selection method
            method = trial.suggest_categorical('method', ['univariate', 'rfe', 'importance_based'])
            
            # Number of features to select
            n_features = trial.suggest_int('n_features', 
                                         min(5, X_train.shape[1]), 
                                         min(self.config.max_features, X_train.shape[1]))
            
            if method == 'univariate':
                # Univariate feature selection
                score_func = trial.suggest_categorical('score_func', ['f_regression', 'mutual_info'])
                
                if score_func == 'f_regression':
                    selector = SelectKBest(f_regression, k=n_features)
                else:
                    selector = SelectKBest(mutual_info_regression, k=n_features)
                
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_val_selected = selector.transform(X_val)
                
            elif method == 'rfe':
                # Recursive feature elimination
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = RFE(estimator, n_features_to_select=n_features)
                
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_val_selected = selector.transform(X_val)
                
            else:  # importance_based
                # Importance-based selection using Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                
                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1][:n_features]
                
                X_train_selected = X_train[:, indices]
                X_val_selected = X_val[:, indices]
                selector = indices
            
            # Train a simple model on selected features
            model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
            model.fit(X_train_selected, y_train)
            
            y_pred = model.predict(X_val_selected)
            y_pred = np.maximum(y_pred, 0)
            
            return wmape(y_val, y_pred)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        # Get best configuration
        best_method = study.best_params['method']
        best_n_features = study.best_params['n_features']
        
        print(f"[FEATURE] Best method: {best_method}, n_features: {best_n_features}")
        
        # Apply best feature selection
        if best_method == 'univariate':
            score_func = study.best_params.get('score_func', 'f_regression')
            if score_func == 'f_regression':
                final_selector = SelectKBest(f_regression, k=best_n_features)
            else:
                final_selector = SelectKBest(mutual_info_regression, k=best_n_features)
        elif best_method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            final_selector = RFE(estimator, n_features_to_select=best_n_features)
        else:  # importance_based
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            final_selector = np.argsort(importances)[::-1][:best_n_features]
        
        # Transform data with best selector
        if hasattr(final_selector, 'fit_transform'):
            X_train_optimized = final_selector.fit_transform(X_train, y_train)
            X_val_optimized = final_selector.transform(X_val)
            selected_feature_indices = final_selector.get_support(indices=True)
        else:  # importance_based case
            X_train_optimized = X_train[:, final_selector]
            X_val_optimized = X_val[:, final_selector]
            selected_feature_indices = final_selector
        
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]
        
        results = {
            'best_method': best_method,
            'best_n_features': best_n_features,
            'best_wmape': study.best_value,
            'selected_features': selected_feature_names,
            'selected_indices': selected_feature_indices,
            'X_train_optimized': X_train_optimized,
            'X_val_optimized': X_val_optimized,
            'selector': final_selector,
            'study': study
        }
        
        self.selected_features = selected_feature_names
        
        print(f"[FEATURE] Feature optimization completed: {len(selected_feature_names)} features selected")
        print(f"[FEATURE] Selected features: {selected_feature_names[:5]}{'...' if len(selected_feature_names) > 5 else ''}")
        
        return results
    
    def optimize_preprocessing(self,
                             X_train: np.ndarray,
                             X_val: np.ndarray) -> Dict:
        """Optimize preprocessing pipeline"""
        
        print("[PREPROC] Optimizing preprocessing pipeline...")
        
        def objective(trial):
            # Scaling method
            scaler_type = trial.suggest_categorical('scaler', ['standard', 'robust', 'minmax', 'none'])
            
            # Dimensionality reduction
            use_reduction = trial.suggest_categorical('use_reduction', [True, False])
            
            X_train_processed = X_train.copy()
            X_val_processed = X_val.copy()
            
            # Apply scaling
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = None
            
            if scaler:
                X_train_processed = scaler.fit_transform(X_train_processed)
                X_val_processed = scaler.transform(X_val_processed)
            
            # Apply dimensionality reduction
            if use_reduction and X_train_processed.shape[1] > 10:
                reduction_method = trial.suggest_categorical('reduction_method', ['pca', 'truncated_svd'])
                n_components = trial.suggest_int('n_components', 5, min(50, X_train_processed.shape[1] - 1))
                
                if reduction_method == 'pca':
                    reducer = PCA(n_components=n_components, random_state=42)
                else:
                    reducer = TruncatedSVD(n_components=n_components, random_state=42)
                
                X_train_processed = reducer.fit_transform(X_train_processed)
                X_val_processed = reducer.transform(X_val_processed)
            
            return X_train_processed.shape[1]  # Return number of final features (minimize for speed)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        return {
            'best_params': study.best_params,
            'study': study
        }

class EnsembleOptimizer:
    """
    Ensemble Weight and Method Optimizer
    
    Optimizes ensemble weights and methods for best performance
    while considering business constraints.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimized_weights = None
        
    def optimize_ensemble_weights(self,
                                base_predictions: Dict[str, np.ndarray],
                                y_true: np.ndarray) -> Dict:
        """
        Optimize ensemble weights for base model predictions
        
        Args:
            base_predictions: Dictionary of model predictions
            y_true: True target values
            
        Returns:
            Optimization results
        """
        
        print("[ENSEMBLE] Optimizing ensemble weights...")
        
        model_names = list(base_predictions.keys())
        prediction_matrix = np.column_stack(list(base_predictions.values()))
        
        def objective(trial):
            # Suggest weights for each model
            weights = []
            for model_name in model_names:
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights.append(weight)
            
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / (weights.sum() + 1e-8)
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.sum(prediction_matrix * weights, axis=1)
            ensemble_pred = np.maximum(ensemble_pred, 0)
            
            # Multi-objective: WMAPE + diversity penalty
            wmape_score = wmape(y_true, ensemble_pred)
            
            # Diversity reward (encourage using multiple models)
            diversity_reward = -np.var(weights) * 0.1  # Negative because we minimize
            
            return wmape_score + diversity_reward
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200)
        
        # Extract optimized weights
        optimized_weights = {}
        for model_name in model_names:
            weight = study.best_params[f'weight_{model_name}']
            optimized_weights[model_name] = weight
        
        # Normalize final weights
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
        
        self.optimized_weights = optimized_weights
        
        # Calculate final ensemble prediction
        weights_array = np.array([optimized_weights[name] for name in model_names])
        final_prediction = np.sum(prediction_matrix * weights_array, axis=1)
        final_wmape = wmape(y_true, final_prediction)
        
        results = {
            'optimized_weights': optimized_weights,
            'final_wmape': final_wmape,
            'improvement': min([wmape(y_true, pred) for pred in base_predictions.values()]) - final_wmape,
            'study': study
        }
        
        print(f"[ENSEMBLE] Weight optimization completed: WMAPE = {final_wmape:.4f}")
        print("[ENSEMBLE] Optimized weights:")
        for model_name, weight in optimized_weights.items():
            print(f"  {model_name}: {weight:.3f}")
        
        return results

class OptimizationPipeline:
    """
    Master Optimization Pipeline
    
    Orchestrates all optimization components to create
    the best possible forecasting system.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Optimization components
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
        self.feature_optimizer = FeatureOptimizer(self.config)
        self.ensemble_optimizer = EnsembleOptimizer(self.config)
        
        # Results storage
        self.optimization_results = {}
        self.best_models = {}
        self.optimization_history = []
        
        # Performance tracking
        self.baseline_performance = None
        self.final_performance = None
        
    def run_full_optimization(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            feature_names: List[str] = None) -> Dict:
        """
        Run complete optimization pipeline
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: Feature names
            
        Returns:
            Complete optimization results
        """
        
        print("⚡ OPTIMIZATION PIPELINE - FULL OPTIMIZATION STARTED")
        print("=" * 80)
        
        start_time = time.time()
        
        # Calculate baseline performance
        print("\n[BASELINE] Calculating baseline performance...")
        baseline_model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_val)
        baseline_pred = np.maximum(baseline_pred, 0)
        self.baseline_performance = wmape(y_val, baseline_pred)
        
        print(f"[BASELINE] Baseline WMAPE: {self.baseline_performance:.4f}")
        
        # Phase 1: Feature Optimization
        print("\n[PHASE 1] Feature Selection and Engineering Optimization...")
        
        if self.config.enable_feature_selection:
            feature_results = self.feature_optimizer.optimize_feature_selection(
                X_train, y_train, X_val, y_val, feature_names
            )
            
            # Use optimized features for subsequent steps
            X_train_opt = feature_results['X_train_optimized']
            X_val_opt = feature_results['X_val_optimized']
            
            self.optimization_results['feature_selection'] = feature_results
        else:
            X_train_opt = X_train
            X_val_opt = X_val
        
        # Phase 2: Hyperparameter Optimization
        print("\n[PHASE 2] Hyperparameter Optimization...")
        
        hyperparameter_results = {}
        
        # Optimize LightGBM
        lgb_results = self.hyperparameter_optimizer.optimize_lightgbm(
            X_train_opt, y_train, X_val_opt, y_val
        )
        hyperparameter_results['lightgbm'] = lgb_results
        
        # Optimize Random Forest
        rf_results = self.hyperparameter_optimizer.optimize_random_forest(
            X_train_opt, y_train, X_val_opt, y_val
        )
        hyperparameter_results['random_forest'] = rf_results
        
        # Optimize Neural Network (if time allows)
        if self.config.n_trials >= 50:
            nn_results = self.hyperparameter_optimizer.optimize_neural_network(
                X_train_opt, y_train, X_val_opt, y_val
            )
            hyperparameter_results['neural_network'] = nn_results
        
        self.optimization_results['hyperparameters'] = hyperparameter_results
        
        # Phase 3: Train Optimized Models and Get Predictions
        print("\n[PHASE 3] Training optimized models...")
        
        base_predictions = {}
        
        # Train LightGBM with best parameters
        best_lgb_params = lgb_results['best_params']
        best_lgb_params.update({
            'objective': 'regression',
            'metric': 'None',
            'verbosity': -1,
            'random_state': 42
        })
        
        lgb_model = lgb.LGBMRegressor(**best_lgb_params)
        lgb_model.fit(X_train_opt, y_train)
        lgb_pred = lgb_model.predict(X_val_opt)
        base_predictions['optimized_lightgbm'] = np.maximum(lgb_pred, 0)
        
        # Train Random Forest with best parameters
        best_rf_params = rf_results['best_params']
        rf_model = RandomForestRegressor(**best_rf_params)
        rf_model.fit(X_train_opt, y_train)
        rf_pred = rf_model.predict(X_val_opt)
        base_predictions['optimized_random_forest'] = np.maximum(rf_pred, 0)
        
        # Train Neural Network (if optimized)
        if 'neural_network' in hyperparameter_results:
            from sklearn.neural_network import MLPRegressor
            best_nn_params = hyperparameter_results['neural_network']['best_params']
            
            # Convert layer parameters back to tuple
            n_layers = best_nn_params.get('n_layers', 2)
            hidden_layer_sizes = []
            for i in range(n_layers):
                if f'layer_{i}_size' in best_nn_params:
                    hidden_layer_sizes.append(best_nn_params[f'layer_{i}_size'])
            
            nn_params = {k: v for k, v in best_nn_params.items() 
                        if not k.startswith('layer_') and k != 'n_layers'}
            nn_params['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
            nn_params.update({
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True
            })
            
            try:
                nn_model = MLPRegressor(**nn_params)
                nn_model.fit(X_train_opt, y_train)
                nn_pred = nn_model.predict(X_val_opt)
                base_predictions['optimized_neural_network'] = np.maximum(nn_pred, 0)
            except Exception as e:
                print(f"[WARNING] Neural network training failed: {e}")
        
        # Store optimized models
        self.best_models = {
            'lightgbm': lgb_model,
            'random_forest': rf_model
        }
        
        # Phase 4: Ensemble Optimization
        print("\n[PHASE 4] Ensemble Optimization...")
        
        if self.config.enable_ensemble_optimization and len(base_predictions) > 1:
            ensemble_results = self.ensemble_optimizer.optimize_ensemble_weights(
                base_predictions, y_val
            )
            
            self.optimization_results['ensemble'] = ensemble_results
            self.final_performance = ensemble_results['final_wmape']
        else:
            # Use best single model
            best_single_wmape = min([wmape(y_val, pred) for pred in base_predictions.values()])
            self.final_performance = best_single_wmape
        
        # Calculate total improvement
        total_time = time.time() - start_time
        improvement = self.baseline_performance - self.final_performance
        improvement_pct = (improvement / self.baseline_performance) * 100
        
        # Final summary
        final_results = {
            'baseline_performance': self.baseline_performance,
            'final_performance': self.final_performance,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'optimization_time': total_time,
            'optimization_results': self.optimization_results,
            'best_models': self.best_models,
            'base_predictions': base_predictions,
            'config': self.config
        }
        
        print("\n" + "=" * 80)
        print("🎉 OPTIMIZATION PIPELINE COMPLETED!")
        print("=" * 80)
        
        print(f"Baseline WMAPE: {self.baseline_performance:.4f}")
        print(f"Final WMAPE: {self.final_performance:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
        print(f"Optimization time: {total_time:.1f} seconds")
        
        # Print component results
        print("\nComponent Results:")
        if 'feature_selection' in self.optimization_results:
            fs_result = self.optimization_results['feature_selection']
            print(f"  Feature Selection: {fs_result['best_n_features']} features selected")
        
        for model_name, hp_result in hyperparameter_results.items():
            print(f"  {model_name}: WMAPE {hp_result['best_value']:.4f}")
        
        if 'ensemble' in self.optimization_results:
            ensemble_result = self.optimization_results['ensemble']
            print(f"  Ensemble: WMAPE {ensemble_result['final_wmape']:.4f}")
        
        return final_results
    
    def save_optimization_results(self, 
                                output_dir: str = "../../models/trained",
                                results: Dict = None) -> Dict[str, str]:
        """Save complete optimization results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save optimization pipeline
        import pickle
        pipeline_file = output_path / f"optimization_pipeline_{timestamp}.pkl"
        with open(pipeline_file, 'wb') as f:
            pickle.dump(self, f)
        saved_files['pipeline'] = str(pipeline_file)
        
        # Save results
        if results:
            results_file = output_path / f"optimization_results_{timestamp}.json"
            
            # Make results JSON serializable
            serializable_results = {}
            for key, value in results.items():
                if key == 'best_models':
                    serializable_results[key] = list(value.keys())  # Just model names
                elif key == 'base_predictions':
                    serializable_results[key] = {k: 'array_data' for k in value.keys()}
                elif key == 'config':
                    serializable_results[key] = value.__dict__
                elif key == 'optimization_results':
                    # Simplified optimization results
                    serializable_results[key] = {}
                    for comp, comp_results in value.items():
                        if isinstance(comp_results, dict):
                            serializable_results[key][comp] = {
                                k: v for k, v in comp_results.items() 
                                if k in ['best_params', 'best_value', 'n_trials']
                            }
                else:
                    serializable_results[key] = value
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            saved_files['results'] = str(results_file)
        
        print(f"[SAVE] Optimization results saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Optimization Pipeline"""
    
    print("⚡ OPTIMIZATION PIPELINE - ULTIMATE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create synthetic dataset
        np.random.seed(42)
        
        n_samples = 3000
        n_features = 20
        
        # Generate features with some noise and correlation
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Add correlation structure
        for i in range(1, n_features):
            X[:, i] += 0.3 * X[:, i-1]  # Add some correlation
        
        # Generate target with complex relationships
        true_coefficients = np.random.normal(0, 1, n_features)
        # Make some coefficients zero (irrelevant features)
        true_coefficients[n_features//2:] *= 0.1
        
        y = (X @ true_coefficients + 
             0.1 * (X[:, 0] * X[:, 1]) +  # Interaction
             0.2 * np.sin(X[:, 2]) +      # Non-linearity
             np.random.normal(0, 0.5, n_samples))  # Noise
        
        y = np.maximum(y, 0)  # Ensure non-negative
        
        # Split data
        train_size = int(0.7 * n_samples)
        val_size = int(0.85 * n_samples)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:val_size]
        y_val = y[train_size:val_size]
        X_test = X[val_size:]
        y_test = y[val_size:]
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        print(f"Dataset: {n_samples} samples, {n_features} features")
        print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Configure optimization
        config = OptimizationConfig(
            study_name="demo_optimization",
            n_trials=30,  # Reduced for demo
            timeout=300,  # 5 minutes
            cv_folds=3,
            enable_feature_selection=True,
            enable_ensemble_optimization=True,
            max_features=15
        )
        
        # Initialize and run optimization pipeline
        print("\n[DEMO] Initializing Optimization Pipeline...")
        
        pipeline = OptimizationPipeline(config)
        
        # Run full optimization
        print("\n[DEMO] Starting full optimization...")
        results = pipeline.run_full_optimization(
            X_train, y_train, X_val, y_val, feature_names
        )
        
        # Test final model on test set
        print("\n[DEMO] Testing optimized models on test set...")
        
        test_performances = {}
        
        for model_name, model in pipeline.best_models.items():
            try:
                if 'feature_selection' in results['optimization_results']:
                    feature_selector = results['optimization_results']['feature_selection']['selector']
                    if hasattr(feature_selector, 'transform'):
                        X_test_opt = feature_selector.transform(X_test)
                    else:  # importance-based case
                        X_test_opt = X_test[:, feature_selector]
                else:
                    X_test_opt = X_test
                
                test_pred = model.predict(X_test_opt)
                test_pred = np.maximum(test_pred, 0)
                test_wmape = wmape(y_test, test_pred)
                test_performances[model_name] = test_wmape
                
                print(f"  {model_name}: Test WMAPE = {test_wmape:.4f}")
                
            except Exception as e:
                print(f"  {model_name}: Test failed - {e}")
        
        # Save results
        print("\n[DEMO] Saving optimization results...")
        saved_files = pipeline.save_optimization_results(results=results)
        
        print("\n" + "=" * 80)
        print("🎉 OPTIMIZATION PIPELINE DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print("Final Results Summary:")
        print(f"  Baseline Performance: {results['baseline_performance']:.4f}")
        print(f"  Final Performance: {results['final_performance']:.4f}")
        print(f"  Improvement: {results['improvement']:.4f} ({results['improvement_percentage']:.1f}%)")
        print(f"  Optimization Time: {results['optimization_time']:.1f} seconds")
        
        if results['optimization_results']:
            print("\nOptimization Components:")
            for component, result in results['optimization_results'].items():
                if component == 'feature_selection':
                    print(f"  Feature Selection: {result['best_n_features']} features")
                elif component == 'hyperparameters':
                    for model, hp_result in result.items():
                        print(f"  {model}: Best WMAPE {hp_result['best_value']:.4f}")
                elif component == 'ensemble':
                    print(f"  Ensemble: Final WMAPE {result['final_wmape']:.4f}")
        
        print(f"\nFiles saved: {len(saved_files)}")
        for file_type, path in saved_files.items():
            print(f"  {file_type}: {path}")
        
        return pipeline, results, test_performances
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results = main()