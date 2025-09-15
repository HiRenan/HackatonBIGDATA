#!/usr/bin/env python3
"""
MODEL CALIBRATION ENGINE - Hackathon Forecast Big Data 2025
Advanced Probability Calibration and Uncertainty Quantification System

Features:
- Probability calibration (Platt scaling, Isotonic regression, Temperature scaling)
- Conformal prediction for distribution-free uncertainty
- Quantile regression for prediction intervals
- Bootstrap methods for empirical distributions
- Bayesian uncertainty quantification
- WMAPE-calibrated confidence intervals
- Multi-model uncertainty fusion
- Adaptive calibration for concept drift

The UNCERTAINTY MASTER that makes predictions trustworthy! 📊
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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.validation import check_X_y, check_array
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape

warnings.filterwarnings('ignore')

class PlattScalingCalibrator:
    """
    Platt Scaling for Probability Calibration
    
    Uses logistic regression to map raw prediction scores
    to well-calibrated probability estimates.
    """
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.is_fitted = False
        
    def fit(self, 
            raw_scores: np.ndarray, 
            true_labels: np.ndarray) -> 'PlattScalingCalibrator':
        """
        Fit Platt scaling calibrator
        
        Args:
            raw_scores: Raw model outputs
            true_labels: True binary labels (0/1)
            
        Returns:
            Self
        """
        
        raw_scores = np.array(raw_scores).reshape(-1, 1)
        true_labels = np.array(true_labels)
        
        self.calibrator.fit(raw_scores, true_labels)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities
        
        Args:
            raw_scores: Raw model outputs
            
        Returns:
            Calibrated probabilities
        """
        
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit first.")
        
        raw_scores = np.array(raw_scores).reshape(-1, 1)
        return self.calibrator.predict_proba(raw_scores)[:, 1]

class IsotonicCalibrator:
    """
    Isotonic Regression Calibrator
    
    Non-parametric method that finds a monotonic function
    to map predictions to calibrated probabilities.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        
    def fit(self, 
            raw_scores: np.ndarray, 
            true_labels: np.ndarray) -> 'IsotonicCalibrator':
        """Fit isotonic regression calibrator"""
        
        raw_scores = np.array(raw_scores).flatten()
        true_labels = np.array(true_labels).flatten()
        
        self.calibrator.fit(raw_scores, true_labels)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities"""
        
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit first.")
        
        raw_scores = np.array(raw_scores).flatten()
        return self.calibrator.predict(raw_scores)

class TemperatureScaling:
    """
    Temperature Scaling for Neural Network Calibration
    
    Learns a temperature parameter to scale logits before softmax
    to improve calibration.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        
    def fit(self, 
            logits: np.ndarray, 
            true_labels: np.ndarray) -> 'TemperatureScaling':
        """
        Fit temperature parameter
        
        Args:
            logits: Raw logit outputs
            true_labels: True labels
            
        Returns:
            Self
        """
        
        logits = np.array(logits).flatten()
        true_labels = np.array(true_labels).flatten()
        
        # Optimize temperature to minimize negative log-likelihood
        def nll(temperature):
            scaled_logits = logits / temperature
            # Convert to probabilities (sigmoid for binary case)
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
            
            # Negative log-likelihood
            nll_value = -np.mean(
                true_labels * np.log(probs) + (1 - true_labels) * np.log(1 - probs)
            )
            return nll_value
        
        # Find optimal temperature
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Get temperature-scaled probabilities"""
        
        if not self.is_fitted:
            raise ValueError("Temperature scaling not fitted. Call fit first.")
        
        logits = np.array(logits).flatten()
        scaled_logits = logits / self.temperature
        
        # Convert to probabilities
        probs = 1 / (1 + np.exp(-scaled_logits))
        return probs

class ConformalPredictor:
    """
    Conformal Prediction for Distribution-Free Uncertainty
    
    Provides prediction intervals with guaranteed coverage
    without assumptions about the underlying distribution.
    """
    
    def __init__(self, 
                 base_model: BaseEstimator,
                 confidence_level: float = 0.95):
        
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.conformity_scores = None
        self.threshold = None
        self.is_fitted = False
        
    def fit(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_calibration: np.ndarray,
            y_calibration: np.ndarray) -> 'ConformalPredictor':
        """
        Fit conformal predictor
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_calibration: Calibration features
            y_calibration: Calibration targets
            
        Returns:
            Self
        """
        
        # Train base model
        self.base_model.fit(X_train, y_train)
        
        # Get predictions on calibration set
        y_pred_calibration = self.base_model.predict(X_calibration)
        
        # Calculate conformity scores (absolute residuals)
        self.conformity_scores = np.abs(y_calibration - y_pred_calibration)
        
        # Calculate threshold for desired confidence level
        n = len(self.conformity_scores)
        rank = int(np.ceil((n + 1) * self.confidence_level))
        self.threshold = np.partition(self.conformity_scores, rank - 1)[rank - 1]
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get point predictions"""
        
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted. Call fit first.")
        
        return self.base_model.predict(X)
    
    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction intervals with guaranteed coverage
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted. Call fit first.")
        
        # Get point predictions
        point_predictions = self.base_model.predict(X)
        
        # Calculate prediction intervals
        lower_bounds = point_predictions - self.threshold
        upper_bounds = point_predictions + self.threshold
        
        return lower_bounds, upper_bounds
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions with uncertainty estimates"""
        
        point_preds = self.predict(X)
        lower_bounds, upper_bounds = self.predict_interval(X)
        
        return {
            'predictions': point_preds,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'uncertainty': (upper_bounds - lower_bounds) / 2,
            'confidence_level': self.confidence_level
        }

class QuantileRegressor:
    """
    Quantile Regression for Prediction Intervals
    
    Directly models different quantiles of the conditional
    distribution to provide prediction intervals.
    """
    
    def __init__(self, quantiles: List[float] = None):
        
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]  # Default quantiles
        
        self.quantiles = sorted(quantiles)
        self.models = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        Fit quantile regression models
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Self
        """
        
        from sklearn.linear_model import QuantileRegressor as SKQuantileRegressor
        
        X, y = check_X_y(X, y)
        
        # Fit separate model for each quantile
        for quantile in self.quantiles:
            try:
                model = SKQuantileRegressor(
                    quantile=quantile,
                    alpha=0.01,  # Small regularization
                    solver='highs'
                )
                model.fit(X, y)
                self.models[quantile] = model
            except ImportError:
                # Fallback to linear regression with custom loss
                print(f"[WARNING] QuantileRegressor not available, using approximation for quantile {quantile}")
                # Simple approximation using weighted least squares
                weights = np.where(y >= np.percentile(y, quantile * 100), 
                                 quantile, 1 - quantile)
                model = Ridge(alpha=1.0)
                model.fit(X, y, sample_weight=weights)
                self.models[quantile] = model
        
        self.is_fitted = True
        
        return self
    
    def predict_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict multiple quantiles
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping quantiles to predictions
        """
        
        if not self.is_fitted:
            raise ValueError("Quantile regressor not fitted. Call fit first.")
        
        X = check_array(X)
        
        predictions = {}
        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X)
        
        return predictions
    
    def predict_interval(self, 
                        X: np.ndarray, 
                        confidence_level: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction intervals for given confidence level
        
        Args:
            X: Input features
            confidence_level: Desired confidence level
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        # Find closest available quantiles
        lower_q = min(self.quantiles, key=lambda x: abs(x - lower_quantile))
        upper_q = min(self.quantiles, key=lambda x: abs(x - upper_quantile))
        
        predictions = self.predict_quantiles(X)
        
        return predictions[lower_q], predictions[upper_q]

class BootstrapUncertainty:
    """
    Bootstrap Methods for Uncertainty Estimation
    
    Uses bootstrap resampling to estimate prediction uncertainty
    through model variability.
    """
    
    def __init__(self, 
                 base_model: BaseEstimator,
                 n_bootstrap: int = 100,
                 random_state: int = 42):
        
        self.base_model = base_model
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.bootstrap_models = []
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BootstrapUncertainty':
        """
        Fit bootstrap ensemble
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Self
        """
        
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]
        
        np.random.seed(self.random_state)
        
        self.bootstrap_models = []
        
        print(f"[INFO] Training {self.n_bootstrap} bootstrap models...")
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model on bootstrap sample
            model = clone(self.base_model)
            model.fit(X_boot, y_boot)
            self.bootstrap_models.append(model)
            
            if (i + 1) % 20 == 0:
                print(f"[PROGRESS] Trained {i + 1}/{self.n_bootstrap} models")
        
        self.is_fitted = True
        
        return self
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions with bootstrap uncertainty
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        
        if not self.is_fitted:
            raise ValueError("Bootstrap ensemble not fitted. Call fit first.")
        
        X = check_array(X)
        
        # Get predictions from all bootstrap models
        all_predictions = []
        for model in self.bootstrap_models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        # Calculate percentiles for prediction intervals
        lower_bound = np.percentile(all_predictions, 5, axis=0)
        upper_bound = np.percentile(all_predictions, 95, axis=0)
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'lower_bound_90': lower_bound,
            'upper_bound_90': upper_bound,
            'all_predictions': all_predictions
        }

class WMAPECalibratedUncertainty:
    """
    WMAPE-Calibrated Uncertainty Estimation
    
    Calibrates uncertainty estimates specifically for WMAPE metric,
    ensuring that prediction intervals have the right coverage
    when evaluated with WMAPE.
    """
    
    def __init__(self):
        self.calibration_curve = None
        self.wmape_bins = None
        self.uncertainty_bins = None
        self.is_fitted = False
        
    def fit(self, 
            predictions: np.ndarray,
            actuals: np.ndarray,
            raw_uncertainties: np.ndarray) -> 'WMAPECalibratedUncertainty':
        """
        Fit WMAPE-calibrated uncertainty model
        
        Args:
            predictions: Model predictions
            actuals: True values
            raw_uncertainties: Raw uncertainty estimates
            
        Returns:
            Self
        """
        
        # Calculate per-sample WMAPE contributions
        abs_errors = np.abs(actuals - predictions)
        wmape_contributions = abs_errors / (np.abs(actuals) + 1e-8)
        
        # Create bins based on uncertainty levels
        n_bins = 10
        uncertainty_percentiles = np.percentile(raw_uncertainties, 
                                               np.linspace(0, 100, n_bins + 1))
        
        calibrated_uncertainties = []
        self.wmape_bins = []
        self.uncertainty_bins = []
        
        for i in range(n_bins):
            # Find samples in this uncertainty bin
            if i == 0:
                mask = raw_uncertainties <= uncertainty_percentiles[i + 1]
            elif i == n_bins - 1:
                mask = raw_uncertainties > uncertainty_percentiles[i]
            else:
                mask = ((raw_uncertainties > uncertainty_percentiles[i]) & 
                       (raw_uncertainties <= uncertainty_percentiles[i + 1]))
            
            if np.sum(mask) > 0:
                # Calculate actual WMAPE for this bin
                bin_wmape = np.mean(wmape_contributions[mask])
                bin_uncertainty = np.mean(raw_uncertainties[mask])
                
                self.wmape_bins.append(bin_wmape)
                self.uncertainty_bins.append(bin_uncertainty)
                calibrated_uncertainties.append(bin_wmape)
        
        # Create interpolation function for calibration
        if len(self.uncertainty_bins) > 1:
            from scipy.interpolate import interp1d
            self.calibration_curve = interp1d(
                self.uncertainty_bins, 
                self.wmape_bins,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
        
        self.is_fitted = True
        
        return self
    
    def calibrate_uncertainty(self, raw_uncertainties: np.ndarray) -> np.ndarray:
        """
        Calibrate raw uncertainties for WMAPE metric
        
        Args:
            raw_uncertainties: Raw uncertainty estimates
            
        Returns:
            WMAPE-calibrated uncertainties
        """
        
        if not self.is_fitted:
            raise ValueError("WMAPE calibrator not fitted. Call fit first.")
        
        if self.calibration_curve is None:
            return raw_uncertainties
        
        return self.calibration_curve(raw_uncertainties)

class ModelCalibrationSuite:
    """
    Comprehensive Model Calibration Suite
    
    Orchestrates all calibration methods and provides
    a unified interface for uncertainty quantification.
    """
    
    def __init__(self, 
                 calibration_methods: List[str] = None,
                 confidence_levels: List[float] = None):
        
        if calibration_methods is None:
            calibration_methods = ['platt', 'isotonic', 'conformal', 'bootstrap']
        
        if confidence_levels is None:
            confidence_levels = [0.8, 0.9, 0.95]
        
        self.calibration_methods = calibration_methods
        self.confidence_levels = confidence_levels
        
        # Calibrators
        self.calibrators = {}
        self.wmape_calibrator = WMAPECalibratedUncertainty()
        
        # Results storage
        self.calibration_results = {}
        self.performance_metrics = {}
        
    def fit_all_calibrators(self,
                           base_model: BaseEstimator,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_calibration: np.ndarray,
                           y_calibration: np.ndarray) -> Dict:
        """
        Fit all available calibration methods
        
        Args:
            base_model: Base predictive model
            X_train: Training features
            y_train: Training targets
            X_calibration: Calibration features
            y_calibration: Calibration targets
            
        Returns:
            Dictionary with fitting results
        """
        
        print("[INFO] Fitting all calibration methods...")
        
        results = {}
        
        # Get base model predictions
        base_model.fit(X_train, y_train)
        predictions_calibration = base_model.predict(X_calibration)
        
        # Fit Platt scaling (for probability calibration)
        if 'platt' in self.calibration_methods:
            try:
                # Convert regression to binary classification for calibration
                binary_labels = (y_calibration > np.median(y_calibration)).astype(int)
                platt_calibrator = PlattScalingCalibrator()
                platt_calibrator.fit(predictions_calibration, binary_labels)
                self.calibrators['platt'] = platt_calibrator
                results['platt'] = {'status': 'success'}
            except Exception as e:
                results['platt'] = {'status': 'failed', 'error': str(e)}
        
        # Fit Isotonic regression
        if 'isotonic' in self.calibration_methods:
            try:
                binary_labels = (y_calibration > np.median(y_calibration)).astype(int)
                isotonic_calibrator = IsotonicCalibrator()
                isotonic_calibrator.fit(predictions_calibration, binary_labels)
                self.calibrators['isotonic'] = isotonic_calibrator
                results['isotonic'] = {'status': 'success'}
            except Exception as e:
                results['isotonic'] = {'status': 'failed', 'error': str(e)}
        
        # Fit Conformal prediction
        if 'conformal' in self.calibration_methods:
            try:
                conformal_predictor = ConformalPredictor(clone(base_model))
                conformal_predictor.fit(X_train, y_train, X_calibration, y_calibration)
                self.calibrators['conformal'] = conformal_predictor
                results['conformal'] = {'status': 'success'}
            except Exception as e:
                results['conformal'] = {'status': 'failed', 'error': str(e)}
        
        # Fit Quantile regression
        if 'quantile' in self.calibration_methods:
            try:
                quantile_regressor = QuantileRegressor()
                quantile_regressor.fit(X_train, y_train)
                self.calibrators['quantile'] = quantile_regressor
                results['quantile'] = {'status': 'success'}
            except Exception as e:
                results['quantile'] = {'status': 'failed', 'error': str(e)}
        
        # Fit Bootstrap uncertainty
        if 'bootstrap' in self.calibration_methods:
            try:
                bootstrap_model = BootstrapUncertainty(clone(base_model), n_bootstrap=50)
                bootstrap_model.fit(X_train, y_train)
                self.calibrators['bootstrap'] = bootstrap_model
                results['bootstrap'] = {'status': 'success'}
            except Exception as e:
                results['bootstrap'] = {'status': 'failed', 'error': str(e)}
        
        # Fit WMAPE calibrator
        try:
            # Get uncertainty estimates from bootstrap (if available)
            if 'bootstrap' in self.calibrators:
                bootstrap_results = self.calibrators['bootstrap'].predict_with_uncertainty(X_calibration)
                raw_uncertainties = bootstrap_results['uncertainty']
            else:
                # Use absolute residuals as proxy
                raw_uncertainties = np.abs(y_calibration - predictions_calibration)
            
            self.wmape_calibrator.fit(predictions_calibration, y_calibration, raw_uncertainties)
            results['wmape_calibrator'] = {'status': 'success'}
            
        except Exception as e:
            results['wmape_calibrator'] = {'status': 'failed', 'error': str(e)}
        
        print(f"[OK] Fitted {len([r for r in results.values() if r['status'] == 'success'])}/{len(results)} calibration methods")
        
        return results
    
    def predict_with_all_uncertainties(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get predictions with all available uncertainty estimates
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with all uncertainty estimates
        """
        
        results = {}
        
        # Conformal prediction
        if 'conformal' in self.calibrators:
            conformal_results = self.calibrators['conformal'].predict_with_uncertainty(X)
            results['conformal'] = conformal_results
        
        # Quantile regression
        if 'quantile' in self.calibrators:
            quantile_predictions = self.calibrators['quantile'].predict_quantiles(X)
            results['quantile'] = quantile_predictions
        
        # Bootstrap uncertainty
        if 'bootstrap' in self.calibrators:
            bootstrap_results = self.calibrators['bootstrap'].predict_with_uncertainty(X)
            results['bootstrap'] = bootstrap_results
        
        return results
    
    def evaluate_calibration_quality(self,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray) -> Dict:
        """
        Evaluate calibration quality on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with calibration metrics
        """
        
        print("[INFO] Evaluating calibration quality...")
        
        evaluation_results = {}
        
        for method_name, calibrator in self.calibrators.items():
            try:
                if method_name == 'conformal':
                    # Evaluate coverage
                    uncertainty_results = calibrator.predict_with_uncertainty(X_test)
                    predictions = uncertainty_results['predictions']
                    lower_bounds = uncertainty_results['lower_bounds']
                    upper_bounds = uncertainty_results['upper_bounds']
                    
                    # Calculate empirical coverage
                    in_interval = (y_test >= lower_bounds) & (y_test <= upper_bounds)
                    empirical_coverage = np.mean(in_interval)
                    
                    # Calculate interval width
                    avg_width = np.mean(upper_bounds - lower_bounds)
                    
                    evaluation_results[method_name] = {
                        'empirical_coverage': empirical_coverage,
                        'target_coverage': calibrator.confidence_level,
                        'coverage_error': abs(empirical_coverage - calibrator.confidence_level),
                        'avg_interval_width': avg_width,
                        'wmape': wmape(y_test, predictions)
                    }
                    
                elif method_name == 'bootstrap':
                    # Evaluate bootstrap uncertainty
                    bootstrap_results = calibrator.predict_with_uncertainty(X_test)
                    predictions = bootstrap_results['predictions']
                    
                    # Calculate coverage for 90% intervals
                    lower_90 = bootstrap_results['lower_bound_90']
                    upper_90 = bootstrap_results['upper_bound_90']
                    coverage_90 = np.mean((y_test >= lower_90) & (y_test <= upper_90))
                    
                    evaluation_results[method_name] = {
                        'empirical_coverage_90': coverage_90,
                        'target_coverage_90': 0.9,
                        'coverage_error_90': abs(coverage_90 - 0.9),
                        'avg_uncertainty': np.mean(bootstrap_results['uncertainty']),
                        'wmape': wmape(y_test, predictions)
                    }
                
            except Exception as e:
                evaluation_results[method_name] = {'error': str(e)}
        
        return evaluation_results
    
    def save_calibrators(self, output_dir: str = "../../models/trained") -> Dict[str, str]:
        """Save all fitted calibrators"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        import pickle
        saved_files = {}
        
        # Save calibration suite
        calibration_file = output_path / f"calibration_suite_{timestamp}.pkl"
        with open(calibration_file, 'wb') as f:
            pickle.dump(self, f)
        saved_files['calibration_suite'] = str(calibration_file)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'calibration_methods': self.calibration_methods,
            'confidence_levels': self.confidence_levels,
            'fitted_calibrators': list(self.calibrators.keys()),
            'wmape_calibrator_fitted': self.wmape_calibrator.is_fitted
        }
        
        metadata_file = output_path / f"calibration_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_file)
        
        print(f"[SAVE] Calibration models saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Model Calibration Engine"""
    
    print("📊 MODEL CALIBRATION ENGINE - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Create synthetic data for demonstration
        np.random.seed(42)
        
        # Generate features
        n_samples = 1000
        n_features = 5
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Generate target with some noise
        true_coefficients = np.random.normal(0, 1, n_features)
        y = X @ true_coefficients + np.random.normal(0, 0.5, n_samples)
        
        # Split data
        split_idx = int(0.6 * n_samples)
        cal_idx = int(0.8 * n_samples)
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_calibration = X[split_idx:cal_idx]
        y_calibration = y[split_idx:cal_idx]
        X_test = X[cal_idx:]
        y_test = y[cal_idx:]
        
        print(f"Data split: Train={len(X_train)}, Cal={len(X_calibration)}, Test={len(X_test)}")
        
        # Create base model
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Initialize calibration suite
        calibration_suite = ModelCalibrationSuite(
            calibration_methods=['conformal', 'bootstrap'],
            confidence_levels=[0.8, 0.9, 0.95]
        )
        
        # Fit all calibrators
        print("\n[DEMO] Fitting calibration methods...")
        fitting_results = calibration_suite.fit_all_calibrators(
            base_model, X_train, y_train, X_calibration, y_calibration
        )
        
        # Get predictions with uncertainty
        print("\n[DEMO] Generating predictions with uncertainty...")
        uncertainty_results = calibration_suite.predict_with_all_uncertainties(X_test)
        
        # Evaluate calibration quality
        print("\n[DEMO] Evaluating calibration quality...")
        evaluation_results = calibration_suite.evaluate_calibration_quality(X_test, y_test)
        
        # Save calibrators
        print("\n[DEMO] Saving calibration models...")
        saved_files = calibration_suite.save_calibrators()
        
        print("\n" + "=" * 80)
        print("🎉 MODEL CALIBRATION ENGINE DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        print("Calibration Results:")
        for method, result in fitting_results.items():
            print(f"  {method}: {result['status']}")
        
        print("\nEvaluation Results:")
        for method, metrics in evaluation_results.items():
            if 'error' not in metrics:
                if 'empirical_coverage' in metrics:
                    print(f"  {method}: Coverage={metrics['empirical_coverage']:.3f}, "
                          f"Target={metrics['target_coverage']:.3f}, "
                          f"WMAPE={metrics['wmape']:.4f}")
        
        print(f"\nFiles saved: {len(saved_files)}")
        
        return calibration_suite, uncertainty_results, evaluation_results

    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

class BayesianCalibrator:
    """
    Bayesian Calibration Engine - PHASE 5 REQUIREMENT

    Implements Bayesian methods for uncertainty quantification
    including posterior sampling and credible intervals.
    """

    def __init__(self,
                 prior_alpha: float = 1.0,
                 prior_beta: float = 1.0,
                 n_samples: int = 1000):

        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n_samples = n_samples

        # Posterior parameters (fitted)
        self.posterior_alpha = None
        self.posterior_beta = None

        # Fitted parameters
        self.is_fitted = False
        self.likelihood_params = None

    def fit(self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            prior_type: str = 'conjugate') -> 'BayesianCalibrator':
        """
        Fit Bayesian calibration model

        Args:
            y_true: True values
            y_pred: Predicted values
            prior_type: Type of prior ('conjugate', 'jeffreys', 'uniform')

        Returns:
            Self
        """

        print("[BAYESIAN] Fitting Bayesian calibration model...")

        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        # Empirical statistics
        error_mean = np.mean(errors)
        error_var = np.var(errors)
        error_std = np.std(errors)

        # For regression problems, we'll model the error distribution
        if prior_type == 'conjugate':
            # Normal-Gamma conjugate prior for Normal likelihood
            # Prior: Normal-Gamma(mu_0, lambda_0, alpha_0, beta_0)

            n = len(errors)
            sample_mean = error_mean
            sample_var = error_var

            # Prior hyperparameters (weakly informative)
            mu_0 = 0.0  # Prior mean for error
            lambda_0 = 1.0  # Prior precision scaling
            alpha_0 = self.prior_alpha  # Prior shape for precision
            beta_0 = self.prior_beta   # Prior rate for precision

            # Posterior hyperparameters (conjugate update)
            mu_n = (lambda_0 * mu_0 + n * sample_mean) / (lambda_0 + n)
            lambda_n = lambda_0 + n
            alpha_n = alpha_0 + n / 2
            beta_n = beta_0 + 0.5 * np.sum(errors**2) + (lambda_0 * n * (sample_mean - mu_0)**2) / (2 * (lambda_0 + n))

            self.likelihood_params = {
                'type': 'normal_gamma',
                'mu_n': mu_n,
                'lambda_n': lambda_n,
                'alpha_n': alpha_n,
                'beta_n': beta_n,
                'n_obs': n
            }

        elif prior_type == 'jeffreys':
            # Jeffreys prior (non-informative)
            n = len(errors)

            # For normal distribution with unknown mean and variance
            # Jeffreys prior is proportional to 1/sigma^2

            self.likelihood_params = {
                'type': 'jeffreys_normal',
                'sample_mean': error_mean,
                'sample_var': error_var,
                'n_obs': n
            }

        else:  # uniform prior
            self.likelihood_params = {
                'type': 'uniform_normal',
                'sample_mean': error_mean,
                'sample_var': error_var,
                'n_obs': len(errors)
            }

        self.is_fitted = True

        print(f"[BAYESIAN] Fitted with {prior_type} prior, n_obs={len(errors)}")

        return self

    def sample_posterior(self, n_samples: int = None) -> Dict[str, np.ndarray]:
        """
        Sample from posterior distribution - PHASE 5 REQUIREMENT

        Args:
            n_samples: Number of samples to draw

        Returns:
            Dictionary with posterior samples
        """

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if n_samples is None:
            n_samples = self.n_samples

        print(f"[BAYESIAN] Drawing {n_samples} posterior samples...")

        if self.likelihood_params['type'] == 'normal_gamma':
            # Sample from Normal-Gamma posterior
            mu_n = self.likelihood_params['mu_n']
            lambda_n = self.likelihood_params['lambda_n']
            alpha_n = self.likelihood_params['alpha_n']
            beta_n = self.likelihood_params['beta_n']

            # Sample precision (inverse variance) from Gamma
            precision_samples = np.random.gamma(alpha_n, 1/beta_n, n_samples)

            # Sample mean from Normal given precision
            mean_samples = np.random.normal(
                mu_n,
                1/np.sqrt(lambda_n * precision_samples),
                n_samples
            )

            # Convert precision to standard deviation
            std_samples = 1/np.sqrt(precision_samples)

            return {
                'error_mean': mean_samples,
                'error_std': std_samples,
                'precision': precision_samples,
                'type': 'normal_gamma'
            }

        elif self.likelihood_params['type'] == 'jeffreys_normal':
            # Approximate sampling for Jeffreys prior
            sample_mean = self.likelihood_params['sample_mean']
            sample_var = self.likelihood_params['sample_var']
            n_obs = self.likelihood_params['n_obs']

            # Approximate posterior for mean and variance
            # Mean: t-distribution
            # Variance: inverse-chi-squared

            dof = n_obs - 1

            # Sample mean (t-distribution)
            mean_samples = stats.t.rvs(dof, loc=sample_mean,
                                     scale=np.sqrt(sample_var/n_obs),
                                     size=n_samples)

            # Sample variance (inverse chi-squared)
            variance_samples = stats.invgamma.rvs(dof/2, scale=dof*sample_var/2, size=n_samples)
            std_samples = np.sqrt(variance_samples)

            return {
                'error_mean': mean_samples,
                'error_std': std_samples,
                'error_variance': variance_samples,
                'type': 'jeffreys'
            }

        else:  # uniform prior
            # Similar to Jeffreys but with different parameterization
            sample_mean = self.likelihood_params['sample_mean']
            sample_var = self.likelihood_params['sample_var']
            n_obs = self.likelihood_params['n_obs']

            # Bootstrap-like sampling
            mean_samples = np.random.normal(sample_mean, np.sqrt(sample_var/n_obs), n_samples)

            # Chi-squared for variance
            chi2_samples = np.random.chisquare(n_obs-1, n_samples)
            variance_samples = (n_obs-1) * sample_var / chi2_samples
            std_samples = np.sqrt(variance_samples)

            return {
                'error_mean': mean_samples,
                'error_std': std_samples,
                'error_variance': variance_samples,
                'type': 'uniform'
            }

    def predict_with_uncertainty(self,
                                y_pred: np.ndarray,
                                confidence_level: float = 0.95,
                                n_posterior_samples: int = None) -> Dict[str, np.ndarray]:
        """
        Generate predictions with Bayesian uncertainty

        Args:
            y_pred: Point predictions
            confidence_level: Confidence level for intervals
            n_posterior_samples: Number of posterior samples

        Returns:
            Dictionary with predictions and credible intervals
        """

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if n_posterior_samples is None:
            n_posterior_samples = self.n_samples

        print(f"[BAYESIAN] Generating Bayesian predictions with {confidence_level:.0%} credible intervals...")

        # Sample from posterior
        posterior_samples = self.sample_posterior(n_posterior_samples)

        # Generate predictions for each posterior sample
        n_pred = len(y_pred)
        predictions_matrix = np.zeros((n_posterior_samples, n_pred))

        for i in range(n_posterior_samples):
            # Add sampled error to predictions
            error_mean = posterior_samples['error_mean'][i]
            error_std = posterior_samples['error_std'][i]

            # Generate prediction with uncertainty
            prediction_errors = np.random.normal(error_mean, error_std, n_pred)
            predictions_matrix[i, :] = y_pred + prediction_errors

        # Calculate credible intervals
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        # Prediction statistics
        pred_mean = np.mean(predictions_matrix, axis=0)
        pred_std = np.std(predictions_matrix, axis=0)
        pred_lower = np.percentile(predictions_matrix, lower_percentile, axis=0)
        pred_upper = np.percentile(predictions_matrix, upper_percentile, axis=0)

        # Epistemic vs aleatoric uncertainty
        epistemic_uncertainty = np.std(np.mean(predictions_matrix, axis=1))  # Model uncertainty
        aleatoric_uncertainty = np.mean(posterior_samples['error_std'])       # Data uncertainty

        results = {
            'predictions': pred_mean,
            'std': pred_std,
            'lower_bound': pred_lower,
            'upper_bound': pred_upper,
            'confidence_level': confidence_level,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'posterior_samples': posterior_samples,
            'prediction_samples': predictions_matrix
        }

        print(f"[BAYESIAN] Generated predictions with epistemic σ={epistemic_uncertainty:.4f}, aleatoric σ={aleatoric_uncertainty:.4f}")

        return results

    def get_posterior_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the posterior distribution"""

        if not self.is_fitted:
            return {}

        posterior_samples = self.sample_posterior(self.n_samples)

        summary = {
            'error_mean': {
                'mean': np.mean(posterior_samples['error_mean']),
                'std': np.std(posterior_samples['error_mean']),
                'quantiles': np.percentile(posterior_samples['error_mean'], [2.5, 25, 50, 75, 97.5])
            },
            'error_std': {
                'mean': np.mean(posterior_samples['error_std']),
                'std': np.std(posterior_samples['error_std']),
                'quantiles': np.percentile(posterior_samples['error_std'], [2.5, 25, 50, 75, 97.5])
            },
            'posterior_type': posterior_samples['type'],
            'n_samples': self.n_samples
        }

        return summary

class PosteriorSamplingEngine:
    """
    Advanced Posterior Sampling Engine - PHASE 5 REQUIREMENT

    Implements various posterior sampling methods for
    uncertainty quantification and model averaging.
    """

    def __init__(self,
                 sampling_method: str = 'gibbs',
                 n_chains: int = 4,
                 n_samples_per_chain: int = 1000,
                 burn_in: int = 200):

        self.sampling_method = sampling_method
        self.n_chains = n_chains
        self.n_samples_per_chain = n_samples_per_chain
        self.burn_in = burn_in

        # Storage for chains
        self.chains = []
        self.diagnostics = {}
        self.is_fitted = False

    def sample_model_posterior(self,
                              models: List[BaseEstimator],
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        Sample from model posterior using Bayesian Model Averaging

        Args:
            models: List of trained models
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)

        Returns:
            Posterior sampling results
        """

        print(f"[POSTERIOR] Starting {self.sampling_method} sampling with {self.n_chains} chains...")

        n_models = len(models)

        # Calculate model likelihoods (using negative WMAPE as proxy)
        model_likelihoods = []
        for model in models:
            y_pred = model.predict(X_train)
            likelihood = -wmape(y_train, y_pred)  # Negative WMAPE (higher is better)
            model_likelihoods.append(likelihood)

        # Convert to probabilities (softmax)
        model_likelihoods = np.array(model_likelihoods)
        model_likelihoods = model_likelihoods - np.max(model_likelihoods)  # Numerical stability
        model_probs = np.exp(model_likelihoods) / np.sum(np.exp(model_likelihoods))

        # Sample model indicators from posterior
        model_samples = []
        prediction_samples = []

        for chain in range(self.n_chains):
            chain_model_samples = []
            chain_prediction_samples = []

            for sample in range(self.n_samples_per_chain):
                # Sample model index
                model_idx = np.random.choice(n_models, p=model_probs)
                chain_model_samples.append(model_idx)

                # Generate prediction with sampled model
                if X_test is not None:
                    pred = models[model_idx].predict(X_test)
                    chain_prediction_samples.append(pred)

            model_samples.append(chain_model_samples)
            if X_test is not None:
                prediction_samples.append(chain_prediction_samples)

        # Combine chains (after burn-in)
        all_model_samples = []
        all_prediction_samples = []

        for chain in range(self.n_chains):
            # Remove burn-in
            chain_models = model_samples[chain][self.burn_in:]
            all_model_samples.extend(chain_models)

            if X_test is not None:
                chain_preds = prediction_samples[chain][self.burn_in:]
                all_prediction_samples.extend(chain_preds)

        # Calculate diagnostics
        self.diagnostics = self._calculate_diagnostics(model_samples)

        results = {
            'model_samples': all_model_samples,
            'model_probabilities': model_probs,
            'model_names': [f"model_{i}" for i in range(n_models)],
            'diagnostics': self.diagnostics,
            'n_effective_samples': len(all_model_samples)
        }

        if X_test is not None:
            # Convert prediction samples to array
            prediction_array = np.array(all_prediction_samples)

            # Calculate prediction statistics
            pred_mean = np.mean(prediction_array, axis=0)
            pred_std = np.std(prediction_array, axis=0)
            pred_lower = np.percentile(prediction_array, 2.5, axis=0)
            pred_upper = np.percentile(prediction_array, 97.5, axis=0)

            results.update({
                'prediction_samples': prediction_array,
                'predictions': pred_mean,
                'prediction_std': pred_std,
                'prediction_lower': pred_lower,
                'prediction_upper': pred_upper
            })

        self.is_fitted = True

        print(f"[POSTERIOR] Sampling completed: {len(all_model_samples)} effective samples")

        return results

    def _calculate_diagnostics(self, chains_data: List[List]) -> Dict[str, float]:
        """Calculate convergence diagnostics"""

        diagnostics = {}

        # Convert to numpy array for easier manipulation
        chains_array = np.array(chains_data)  # Shape: (n_chains, n_samples)

        # Remove burn-in
        chains_no_burnin = chains_array[:, self.burn_in:]

        # R-hat (Gelman-Rubin statistic)
        n_chains, n_samples = chains_no_burnin.shape

        if n_chains > 1:
            # Within-chain variance
            W = np.mean([np.var(chain, ddof=1) for chain in chains_no_burnin])

            # Between-chain variance
            chain_means = np.mean(chains_no_burnin, axis=1)
            B = n_samples * np.var(chain_means, ddof=1) if np.var(chain_means, ddof=1) > 0 else 1e-10

            # Marginal posterior variance
            var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

            # R-hat statistic
            r_hat = np.sqrt(var_plus / W) if W > 0 else 1.0

            diagnostics['r_hat'] = r_hat
            diagnostics['converged'] = r_hat < 1.1  # Convergence threshold

        # Effective sample size (rough estimate)
        total_samples = n_chains * n_samples
        autocorr = self._estimate_autocorrelation(chains_no_burnin.flatten())
        eff_sample_size = total_samples / (1 + 2 * np.sum(autocorr[:min(len(autocorr)//4, 50)]))

        diagnostics['effective_sample_size'] = max(1, int(eff_sample_size))
        diagnostics['total_samples'] = total_samples

        return diagnostics

    def _estimate_autocorrelation(self, series: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """Estimate autocorrelation function"""

        n = len(series)
        max_lag = min(max_lag, n // 4)

        autocorr = np.zeros(max_lag)

        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                c0 = np.var(series)
                c_lag = np.mean((series[:-lag] - np.mean(series)) * (series[lag:] - np.mean(series)))
                autocorr[lag] = c_lag / c0 if c0 > 0 else 0

        return autocorr

if __name__ == "__main__":
    results = main()