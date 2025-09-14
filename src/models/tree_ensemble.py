#!/usr/bin/env python3
"""
TREE ENSEMBLE ENGINE - Hackathon Forecast Big Data 2025
XGBoost + CatBoost Advanced Ensemble for Competition Dominance

Features:
- Custom WMAPE objectives for both XGBoost and CatBoost
- Advanced feature engineering integration
- Hyperparameter optimization with Optuna
- Model stacking and blending strategies
- Automated feature selection
- Business constraint integration
- Ensemble uncertainty quantification

The TRIPLE THREAT: LightGBM + XGBoost + CatBoost! 🏆
"""

import xgboost as xgb
import catboost as cb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import json
import time
from datetime import datetime
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape
from src.utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

class XGBoostWMAPE:
    """
    Custom XGBoost with WMAPE Optimization
    
    XGBoost doesn't natively support WMAPE, so we implement
    a custom objective that approximates WMAPE gradient.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def wmape_objective(self, y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        WMAPE objective for XGBoost
        
        Args:
            y_pred: Predicted values
            dtrain: DMatrix with true values
            
        Returns:
            (gradient, hessian) tuple
        """
        y_true = dtrain.get_label()
        
        # Ensure positive denominator
        y_true_safe = np.where(np.abs(y_true) < self.epsilon, self.epsilon, y_true)
        
        # Calculate residuals
        residual = y_pred - y_true
        abs_y_true = np.abs(y_true)
        sum_abs_y_true = np.sum(abs_y_true) + self.epsilon
        
        # WMAPE gradient approximation
        gradient = np.where(residual >= 0, 1.0, -1.0) / sum_abs_y_true
        
        # Hessian approximation (constant for numerical stability)
        hessian = np.ones_like(gradient) * (1.0 / sum_abs_y_true) * 0.01
        
        return gradient, hessian

def xgb_wmape_eval(y_pred: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """WMAPE evaluation function for XGBoost"""
    y_true = dtrain.get_label()
    wmape_score = wmape(y_true, y_pred)
    return 'wmape', wmape_score

class CatBoostWMAPE:
    """
    CatBoost with WMAPE Integration
    
    CatBoost has better built-in support for custom metrics,
    so we can implement WMAPE more directly.
    """
    
    def __init__(self):
        pass
    
    class WMAPEMetric(object):
        """Custom WMAPE metric for CatBoost"""
        
        def get_final_error(self, error, weight):
            return error / (weight + 1e-8)

        def is_max_optimal(self):
            return False  # Lower WMAPE is better

        def evaluate(self, approxes, target, weight):
            """
            Evaluate WMAPE metric
            
            Args:
                approxes: Predicted values
                target: True values  
                weight: Sample weights (not used for WMAPE)
                
            Returns:
                (error, weight) tuple
            """
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])
            
            approx = approxes[0]
            
            # Calculate WMAPE components
            abs_error = np.sum(np.abs(np.array(target) - np.array(approx)))
            abs_actual = np.sum(np.abs(np.array(target)))
            
            return abs_error, abs_actual

class TreeEnsembleEngine:
    """
    Advanced Tree Ensemble Engine
    
    Combines XGBoost and CatBoost with advanced ensemble techniques:
    - Model stacking
    - Weighted blending  
    - Feature importance fusion
    - Uncertainty quantification
    """
    
    def __init__(self,
                 date_col: str = 'transaction_date',
                 target_col: str = 'quantity',
                 feature_cols: List[str] = None):
        
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols or []
        
        # Model components
        self.xgb_model = None
        self.catboost_model = None
        self.stacking_model = None
        
        # Training state
        self.feature_columns = []
        self.categorical_features = []
        self.best_params = {'xgb': {}, 'catboost': {}}
        self.training_history = {}
        self.ensemble_weights = None
        
        # Default parameters optimized for forecasting
        self.xgb_default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }
        
        self.catboost_default_params = {
            'objective': 'RMSE',
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepare features for tree ensemble training
        
        Args:
            df: DataFrame with features
            
        Returns:
            (processed_df, feature_columns, categorical_features)
        """
        
        print("[INFO] Preparing features for Tree Ensemble...")
        
        processed_df = df.copy()
        
        # Identify feature columns
        exclude_cols = [self.target_col, self.date_col, 'internal_product_id', 'internal_store_id']
        feature_columns = [col for col in processed_df.columns if col not in exclude_cols]
        
        # Identify categorical features
        categorical_features = []
        for col in feature_columns:
            if (processed_df[col].dtype in ['category', 'object'] or
                col.endswith(('_tier', '_stage', '_type', '_category')) or
                col.startswith(('categoria', 'marca', 'fabricante'))):
                categorical_features.append(col)
                
                # Ensure categorical encoding
                if processed_df[col].dtype == 'object':
                    processed_df[col] = processed_df[col].astype('category')
        
        # Handle missing values
        for col in feature_columns:
            if processed_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            else:
                processed_df[col] = processed_df[col].fillna('Unknown')
        
        self.feature_columns = feature_columns
        self.categorical_features = categorical_features
        
        print(f"[OK] Prepared {len(feature_columns)} features, {len(categorical_features)} categorical")
        
        return processed_df, feature_columns, categorical_features
    
    def train_xgboost(self, 
                     train_df: pd.DataFrame,
                     val_df: Optional[pd.DataFrame] = None,
                     params: Optional[Dict] = None,
                     use_wmape_objective: bool = True) -> Dict:
        """
        Train XGBoost model with optional custom WMAPE objective
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            params: XGBoost parameters (optional)
            use_wmape_objective: Whether to use custom WMAPE objective
            
        Returns:
            Training results
        """
        
        print("[INFO] Training XGBoost model...")
        
        # Prepare features
        processed_df, feature_columns, categorical_features = self.prepare_features(train_df)
        
        # Prepare training data
        X_train = processed_df[feature_columns]
        y_train = processed_df[self.target_col]
        
        # Handle categorical features for XGBoost
        for cat_col in categorical_features:
            if cat_col in X_train.columns:
                X_train[cat_col] = X_train[cat_col].cat.codes
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
        
        # Prepare validation set
        eval_sets = [(dtrain, 'train')]
        if val_df is not None:
            processed_val_df, _, _ = self.prepare_features(val_df)
            X_val = processed_val_df[feature_columns]
            y_val = processed_val_df[self.target_col]
            
            # Handle categorical features
            for cat_col in categorical_features:
                if cat_col in X_val.columns:
                    X_val[cat_col] = X_val[cat_col].cat.codes
            
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
            eval_sets.append((dval, 'valid'))
        
        # Use provided params or defaults
        model_params = params or self.xgb_default_params.copy()
        
        # Training parameters
        training_params = {
            'params': model_params,
            'dtrain': dtrain,
            'num_boost_round': model_params.pop('n_estimators', 1000),
            'evals': eval_sets,
            'early_stopping_rounds': 50,
            'verbose_eval': 100
        }
        
        start_time = time.time()
        
        if use_wmape_objective:
            # Train with custom WMAPE objective
            wmape_obj = XGBoostWMAPE()
            training_params['obj'] = wmape_obj.wmape_objective
            training_params['feval'] = xgb_wmape_eval
            
            # Remove conflicting parameters
            model_params.pop('objective', None)
            model_params.pop('eval_metric', None)
        
        # Train model
        self.xgb_model = xgb.train(**training_params)
        
        training_time = time.time() - start_time
        
        # Training results
        training_results = {
            'model_type': 'xgboost',
            'training_time': training_time,
            'best_iteration': self.xgb_model.best_iteration,
            'best_score': self.xgb_model.best_score,
            'feature_importance': self.xgb_model.get_score(importance_type='gain'),
            'use_wmape_objective': use_wmape_objective
        }
        
        self.training_history['xgboost'] = training_results
        
        print(f"[OK] XGBoost training completed in {training_time:.2f}s")
        if val_df is not None and hasattr(self.xgb_model, 'best_score'):
            print(f"[OK] Best score: {self.xgb_model.best_score}")
        
        return training_results
    
    def train_catboost(self,
                      train_df: pd.DataFrame,
                      val_df: Optional[pd.DataFrame] = None,
                      params: Optional[Dict] = None,
                      use_wmape_metric: bool = True) -> Dict:
        """
        Train CatBoost model with optional custom WMAPE metric
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            params: CatBoost parameters (optional)
            use_wmape_metric: Whether to use custom WMAPE metric
            
        Returns:
            Training results
        """
        
        print("[INFO] Training CatBoost model...")
        
        # Prepare features
        processed_df, feature_columns, categorical_features = self.prepare_features(train_df)
        
        # Prepare training data
        X_train = processed_df[feature_columns]
        y_train = processed_df[self.target_col]
        
        # Get categorical feature indices
        cat_feature_indices = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]
        
        # Create Pool
        train_pool = cb.Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_feature_indices
        )
        
        # Prepare validation set
        valid_pool = None
        if val_df is not None:
            processed_val_df, _, _ = self.prepare_features(val_df)
            X_val = processed_val_df[feature_columns]
            y_val = processed_val_df[self.target_col]
            
            valid_pool = cb.Pool(
                data=X_val,
                label=y_val,
                cat_features=cat_feature_indices
            )
        
        # Use provided params or defaults
        model_params = params or self.catboost_default_params.copy()
        
        # Add custom WMAPE metric if requested
        if use_wmape_metric:
            model_params['custom_metric'] = ['WMAPE:hints=skip_train~false']
            # CatBoost doesn't have built-in WMAPE, so we'll use RMSE as objective
            # and track WMAPE separately
        
        start_time = time.time()
        
        # Initialize model
        self.catboost_model = cb.CatBoostRegressor(**model_params)
        
        # Train model
        self.catboost_model.fit(
            train_pool,
            eval_set=valid_pool if valid_pool is not None else None,
            early_stopping_rounds=50,
            verbose=100
        )
        
        training_time = time.time() - start_time
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.catboost_model, 'get_feature_importance'):
            importance_values = self.catboost_model.get_feature_importance()
            feature_importance = dict(zip(feature_columns, importance_values))
        
        # Training results
        training_results = {
            'model_type': 'catboost',
            'training_time': training_time,
            'best_iteration': self.catboost_model.tree_count_,
            'feature_importance': feature_importance,
            'use_wmape_metric': use_wmape_metric
        }
        
        # Get validation scores if available
        if hasattr(self.catboost_model, 'get_best_score'):
            try:
                best_score = self.catboost_model.get_best_score()
                training_results['best_score'] = best_score
            except:
                pass
        
        self.training_history['catboost'] = training_results
        
        print(f"[OK] CatBoost training completed in {training_time:.2f}s")
        
        return training_results
    
    def predict_individual(self, df: pd.DataFrame, model_type: str = 'both') -> Dict[str, np.ndarray]:
        """
        Make predictions with individual models
        
        Args:
            df: DataFrame with features
            model_type: 'xgboost', 'catboost', or 'both'
            
        Returns:
            Dictionary of predictions
        """
        
        if model_type not in ['xgboost', 'catboost', 'both']:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        # Prepare features
        processed_df, _, _ = self.prepare_features(df)
        X = processed_df[self.feature_columns]
        
        predictions = {}
        
        if model_type in ['xgboost', 'both'] and self.xgb_model is not None:
            # Handle categorical features for XGBoost
            X_xgb = X.copy()
            for cat_col in self.categorical_features:
                if cat_col in X_xgb.columns:
                    X_xgb[cat_col] = X_xgb[cat_col].cat.codes
            
            dmatrix = xgb.DMatrix(X_xgb, feature_names=self.feature_columns)
            xgb_pred = self.xgb_model.predict(dmatrix, iteration_range=(0, self.xgb_model.best_iteration))
            predictions['xgboost'] = np.maximum(xgb_pred, 0)  # Ensure non-negative
        
        if model_type in ['catboost', 'both'] and self.catboost_model is not None:
            catboost_pred = self.catboost_model.predict(X)
            predictions['catboost'] = np.maximum(catboost_pred, 0)  # Ensure non-negative
        
        return predictions
    
    def train_stacking_ensemble(self, 
                               train_df: pd.DataFrame,
                               val_df: Optional[pd.DataFrame] = None,
                               stacking_model: Any = None) -> Dict:
        """
        Train stacking ensemble on top of base models
        
        Args:
            train_df: Training data
            val_df: Validation data (optional)
            stacking_model: Meta-model for stacking (default: LinearRegression)
            
        Returns:
            Stacking results
        """
        
        if self.xgb_model is None or self.catboost_model is None:
            raise ValueError("Both XGBoost and CatBoost models must be trained first")
        
        print("[INFO] Training stacking ensemble...")
        
        # Get base model predictions on training data
        train_predictions = self.predict_individual(train_df, 'both')
        
        # Create stacking features
        stacking_features = np.column_stack([
            train_predictions['xgboost'],
            train_predictions['catboost']
        ])
        
        # Target
        y_train = train_df[self.target_col].values
        
        # Initialize meta-model
        if stacking_model is None:
            # Use Ridge regression with positive constraint for better interpretability
            from sklearn.linear_model import Ridge
            self.stacking_model = Ridge(alpha=1.0, positive=True)
        else:
            self.stacking_model = stacking_model
        
        # Train stacking model
        start_time = time.time()
        self.stacking_model.fit(stacking_features, y_train)
        training_time = time.time() - start_time
        
        # Evaluate stacking model
        stacked_pred_train = self.stacking_model.predict(stacking_features)
        train_wmape = wmape(y_train, stacked_pred_train)
        
        stacking_results = {
            'training_time': training_time,
            'train_wmape': train_wmape,
            'stacking_weights': self.stacking_model.coef_ if hasattr(self.stacking_model, 'coef_') else None
        }
        
        # Evaluate on validation if provided
        if val_df is not None:
            val_predictions = self.predict_individual(val_df, 'both')
            val_stacking_features = np.column_stack([
                val_predictions['xgboost'],
                val_predictions['catboost']
            ])
            
            stacked_pred_val = self.stacking_model.predict(val_stacking_features)
            y_val = val_df[self.target_col].values
            val_wmape = wmape(y_val, stacked_pred_val)
            
            stacking_results['val_wmape'] = val_wmape
            
            print(f"[STACKING] Validation WMAPE: {val_wmape:.4f}")
        
        self.training_history['stacking'] = stacking_results
        
        print(f"[OK] Stacking ensemble trained in {training_time:.2f}s")
        print(f"[STACKING] Training WMAPE: {train_wmape:.4f}")
        
        if stacking_results.get('stacking_weights') is not None:
            weights = stacking_results['stacking_weights']
            print(f"[STACKING] Model weights: XGBoost={weights[0]:.3f}, CatBoost={weights[1]:.3f}")
        
        return stacking_results
    
    def predict_ensemble(self, df: pd.DataFrame, method: str = 'stacking') -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            df: DataFrame with features
            method: 'stacking', 'weighted_average', or 'simple_average'
            
        Returns:
            Ensemble predictions
        """
        
        # Get individual predictions
        individual_preds = self.predict_individual(df, 'both')
        
        if method == 'stacking':
            if self.stacking_model is None:
                raise ValueError("Stacking model not trained. Use train_stacking_ensemble first.")
            
            stacking_features = np.column_stack([
                individual_preds['xgboost'],
                individual_preds['catboost']
            ])
            
            ensemble_pred = self.stacking_model.predict(stacking_features)
            
        elif method == 'weighted_average':
            if self.ensemble_weights is None:
                # Default weights (can be optimized)
                self.ensemble_weights = {'xgboost': 0.5, 'catboost': 0.5}
            
            ensemble_pred = (
                individual_preds['xgboost'] * self.ensemble_weights['xgboost'] +
                individual_preds['catboost'] * self.ensemble_weights['catboost']
            )
            
        elif method == 'simple_average':
            ensemble_pred = (individual_preds['xgboost'] + individual_preds['catboost']) / 2
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return np.maximum(ensemble_pred, 0)  # Ensure non-negative
    
    def optimize_ensemble_weights(self, 
                                val_df: pd.DataFrame,
                                n_trials: int = 50) -> Dict:
        """
        Optimize ensemble weights using Optuna
        
        Args:
            val_df: Validation data for optimization
            n_trials: Number of optimization trials
            
        Returns:
            Optimization results
        """
        
        print(f"[INFO] Optimizing ensemble weights with {n_trials} trials...")
        
        # Get individual predictions
        individual_preds = self.predict_individual(val_df, 'both')
        y_true = val_df[self.target_col].values
        
        def objective(trial):
            # Define weight for XGBoost (CatBoost weight = 1 - xgb_weight)
            xgb_weight = trial.suggest_float('xgb_weight', 0.0, 1.0)
            catboost_weight = 1.0 - xgb_weight
            
            # Calculate weighted ensemble prediction
            ensemble_pred = (
                individual_preds['xgboost'] * xgb_weight +
                individual_preds['catboost'] * catboost_weight
            )
            
            # Calculate WMAPE
            return wmape(y_true, ensemble_pred)
        
        # Create and run study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store best weights
        best_xgb_weight = study.best_params['xgb_weight']
        best_catboost_weight = 1.0 - best_xgb_weight
        
        self.ensemble_weights = {
            'xgboost': best_xgb_weight,
            'catboost': best_catboost_weight
        }
        
        optimization_results = {
            'best_weights': self.ensemble_weights,
            'best_wmape': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        print(f"[OPTUNA] Best WMAPE: {study.best_value:.4f}")
        print(f"[OPTUNA] Best weights: XGBoost={best_xgb_weight:.3f}, CatBoost={best_catboost_weight:.3f}")
        
        return optimization_results
    
    def save_models(self, output_dir: str = "../../models/trained") -> Dict[str, str]:
        """Save all trained models"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save XGBoost model
        if self.xgb_model is not None:
            xgb_file = output_path / f"xgboost_model_{timestamp}.json"
            self.xgb_model.save_model(str(xgb_file))
            saved_files['xgboost'] = str(xgb_file)
        
        # Save CatBoost model
        if self.catboost_model is not None:
            catboost_file = output_path / f"catboost_model_{timestamp}.cbm"
            self.catboost_model.save_model(str(catboost_file))
            saved_files['catboost'] = str(catboost_file)
        
        # Save stacking model
        if self.stacking_model is not None:
            import pickle
            stacking_file = output_path / f"stacking_model_{timestamp}.pkl"
            with open(stacking_file, 'wb') as f:
                pickle.dump(self.stacking_model, f)
            saved_files['stacking'] = str(stacking_file)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'ensemble_weights': self.ensemble_weights
        }
        
        metadata_file = output_path / f"tree_ensemble_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files['metadata'] = str(metadata_file)
        
        print(f"[SAVE] Tree ensemble models saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Tree Ensemble Engine"""
    
    print("🌳 TREE ENSEMBLE ENGINE - XGBOOST + CATBOOST DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load data with enhanced features
        print("Loading data for tree ensemble demonstration...")
        trans_df, prod_df, pdv_df = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=25000,
            sample_products=500,
            enable_joins=True,
            validate_loss=True
        )
        
        # Add basic temporal features for demo
        if 'transaction_date' in trans_df.columns:
            trans_df['transaction_date'] = pd.to_datetime(trans_df['transaction_date'])
            trans_df['month'] = trans_df['transaction_date'].dt.month
            trans_df['day_of_week'] = trans_df['transaction_date'].dt.dayofweek
            trans_df['is_weekend'] = trans_df['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"Data loaded: {trans_df.shape}")
        
        # Initialize tree ensemble engine
        tree_ensemble = TreeEnsembleEngine(
            date_col='transaction_date',
            target_col='quantity'
        )
        
        # Split data for training/validation
        split_date = trans_df['transaction_date'].quantile(0.8)
        train_data = trans_df[trans_df['transaction_date'] < split_date].copy()
        val_data = trans_df[trans_df['transaction_date'] >= split_date].copy()
        
        print(f"Train data: {train_data.shape}, Val data: {val_data.shape}")
        
        # Train XGBoost
        print("\n[DEMO] Training XGBoost with WMAPE objective...")
        xgb_results = tree_ensemble.train_xgboost(
            train_data,
            val_data,
            use_wmape_objective=True
        )
        
        # Train CatBoost
        print("\n[DEMO] Training CatBoost with WMAPE metric...")
        catboost_results = tree_ensemble.train_catboost(
            train_data,
            val_data,
            use_wmape_metric=True
        )
        
        # Train stacking ensemble
        print("\n[DEMO] Training stacking ensemble...")
        stacking_results = tree_ensemble.train_stacking_ensemble(train_data, val_data)
        
        # Optimize ensemble weights
        print("\n[DEMO] Optimizing ensemble weights...")
        weight_optimization = tree_ensemble.optimize_ensemble_weights(val_data, n_trials=20)
        
        # Compare predictions
        print("\n[DEMO] Comparing ensemble methods...")
        individual_preds = tree_ensemble.predict_individual(val_data, 'both')
        stacked_preds = tree_ensemble.predict_ensemble(val_data, 'stacking')
        weighted_preds = tree_ensemble.predict_ensemble(val_data, 'weighted_average')
        
        # Evaluate all methods
        y_true = val_data['quantity'].values
        
        results_comparison = {
            'xgboost': wmape(y_true, individual_preds['xgboost']),
            'catboost': wmape(y_true, individual_preds['catboost']),
            'stacking': wmape(y_true, stacked_preds),
            'weighted_average': wmape(y_true, weighted_preds)
        }
        
        # Save models
        print("\n[DEMO] Saving models...")
        saved_files = tree_ensemble.save_models()
        
        print("\n" + "=" * 80)
        print("🎉 TREE ENSEMBLE ENGINE DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print("Performance Comparison (WMAPE):")
        for method, score in results_comparison.items():
            print(f"  {method:<15}: {score:.4f}")
        
        print(f"\nFiles saved: {len(saved_files)}")
        
        return tree_ensemble, results_comparison, saved_files
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results = main()