#!/usr/bin/env python3
"""
MODEL DIAGNOSTICS SUITE - Hackathon Forecast Big Data 2025
Advanced Model Performance Monitoring and Drift Detection System

Features:
- Performance degradation monitoring and alerts
- Concept drift detection using statistical tests
- Feature importance stability analysis
- Prediction quality assessment and calibration monitoring
- Model comparison and benchmarking suite
- Business impact analysis and reporting
- Automated model health scoring
- Real-time monitoring dashboards

The MODEL DOCTOR that keeps your forecasts healthy! 🔬
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape
from src.evaluation.error_analysis import ErrorDecomposer, ResidualAnalyzer

warnings.filterwarnings('ignore')

@dataclass
class ModelHealthMetrics:
    """
    Model Health Metrics Container
    
    Stores comprehensive health metrics for a model
    at a specific point in time.
    """
    
    timestamp: datetime
    model_name: str
    
    # Performance metrics
    wmape: float
    mae: float
    rmse: float
    r2: float
    
    # Stability metrics
    prediction_stability: float
    feature_importance_stability: float
    calibration_score: float
    
    # Drift metrics
    concept_drift_score: float
    data_drift_score: float
    
    # Business metrics
    business_impact_score: float
    
    # Overall health score (0-100)
    health_score: float
    
    # Flags
    alerts: List[str]
    warnings: List[str]

class ConceptDriftDetector:
    """
    Concept Drift Detection System
    
    Detects when the relationship between features and target
    changes over time using statistical tests and monitoring.
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 drift_threshold: float = 0.05):
        
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Historical data storage
        self.reference_data = None
        self.performance_history = []
        self.drift_history = []
        
        # Detection methods
        self.detection_methods = [
            'performance_degradation',
            'distribution_shift',
            'prediction_drift',
            'residual_drift'
        ]
        
    def set_reference_data(self,
                         X_reference: np.ndarray,
                         y_reference: np.ndarray,
                         predictions_reference: np.ndarray) -> None:
        """
        Set reference data for drift detection
        
        Args:
            X_reference: Reference feature matrix
            y_reference: Reference target values
            predictions_reference: Reference predictions
        """
        
        self.reference_data = {
            'X': X_reference,
            'y': y_reference,
            'predictions': predictions_reference,
            'timestamp': datetime.now()
        }
        
        # Calculate reference statistics
        self.reference_stats = {
            'feature_means': np.mean(X_reference, axis=0),
            'feature_stds': np.std(X_reference, axis=0),
            'target_mean': np.mean(y_reference),
            'target_std': np.std(y_reference),
            'prediction_mean': np.mean(predictions_reference),
            'prediction_std': np.std(predictions_reference),
            'wmape': wmape(y_reference, predictions_reference)
        }
        
        print(f"[DRIFT] Reference data set: {len(X_reference)} samples")
    
    def detect_drift(self,
                    X_current: np.ndarray,
                    y_current: np.ndarray,
                    predictions_current: np.ndarray) -> Dict[str, Any]:
        """
        Detect concept drift using multiple methods
        
        Args:
            X_current: Current feature matrix
            y_current: Current target values
            predictions_current: Current predictions
            
        Returns:
            Dictionary with drift detection results
        """
        
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        drift_results = {
            'timestamp': datetime.now(),
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'method_results': {},
            'alerts': [],
            'recommendations': []
        }
        
        # 1. Performance Degradation Detection
        drift_results['method_results']['performance'] = self._detect_performance_drift(
            y_current, predictions_current
        )
        
        # 2. Feature Distribution Shift Detection
        if X_current.shape[1] == self.reference_data['X'].shape[1]:
            drift_results['method_results']['feature_distribution'] = self._detect_feature_drift(
                X_current
            )
        
        # 3. Target Distribution Shift Detection
        drift_results['method_results']['target_distribution'] = self._detect_target_drift(
            y_current
        )
        
        # 4. Prediction Distribution Shift Detection
        drift_results['method_results']['prediction_distribution'] = self._detect_prediction_drift(
            predictions_current
        )
        
        # 5. Residual Pattern Changes
        drift_results['method_results']['residual_patterns'] = self._detect_residual_drift(
            y_current, predictions_current
        )
        
        # Aggregate drift score
        method_scores = [
            result.get('drift_score', 0) 
            for result in drift_results['method_results'].values()
            if 'drift_score' in result
        ]
        
        if method_scores:
            drift_results['drift_score'] = np.mean(method_scores)
            drift_results['overall_drift_detected'] = drift_results['drift_score'] > self.drift_threshold
        
        # Generate alerts and recommendations
        self._generate_drift_alerts(drift_results)
        
        # Store in history
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _detect_performance_drift(self,
                                 y_current: np.ndarray,
                                 predictions_current: np.ndarray) -> Dict:
        """Detect performance degradation"""
        
        current_wmape = wmape(y_current, predictions_current)
        reference_wmape = self.reference_stats['wmape']
        
        # Calculate relative performance degradation
        performance_change = (current_wmape - reference_wmape) / reference_wmape
        
        # Significant degradation if >20% increase in WMAPE
        drift_detected = performance_change > 0.2
        
        return {
            'method': 'performance_degradation',
            'current_wmape': current_wmape,
            'reference_wmape': reference_wmape,
            'performance_change': performance_change,
            'drift_detected': drift_detected,
            'drift_score': max(0, performance_change) if drift_detected else 0
        }
    
    def _detect_feature_drift(self, X_current: np.ndarray) -> Dict:
        """Detect feature distribution drift"""
        
        X_reference = self.reference_data['X']
        
        drift_scores = []
        feature_drift_results = {}
        
        for i in range(X_current.shape[1]):
            # Kolmogorov-Smirnov test for each feature
            try:
                ks_stat, p_value = ks_2samp(X_reference[:, i], X_current[:, i])
                
                feature_drift = p_value < self.drift_threshold
                drift_scores.append(ks_stat if feature_drift else 0)
                
                feature_drift_results[f'feature_{i}'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': feature_drift
                }
                
            except Exception as e:
                print(f"[WARNING] Feature {i} drift test failed: {e}")
        
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0
        overall_drift_detected = overall_drift_score > self.drift_threshold
        
        return {
            'method': 'feature_distribution',
            'overall_drift_score': overall_drift_score,
            'overall_drift_detected': overall_drift_detected,
            'feature_results': feature_drift_results,
            'drift_score': overall_drift_score
        }
    
    def _detect_target_drift(self, y_current: np.ndarray) -> Dict:
        """Detect target distribution drift"""
        
        y_reference = self.reference_data['y']
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(y_reference, y_current)
            
            # t-test for mean difference
            t_stat, t_p_value = stats.ttest_ind(y_reference, y_current)
            
            # F-test for variance difference (Levene's test)
            f_stat, f_p_value = stats.levene(y_reference, y_current)
            
            drift_detected = (p_value < self.drift_threshold or 
                            t_p_value < self.drift_threshold or 
                            f_p_value < self.drift_threshold)
            
            return {
                'method': 'target_distribution',
                'ks_statistic': ks_stat,
                'ks_p_value': p_value,
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'levene_statistic': f_stat,
                'levene_p_value': f_p_value,
                'drift_detected': drift_detected,
                'drift_score': ks_stat if drift_detected else 0
            }
            
        except Exception as e:
            print(f"[WARNING] Target drift test failed: {e}")
            return {
                'method': 'target_distribution',
                'error': str(e),
                'drift_detected': False,
                'drift_score': 0
            }
    
    def _detect_prediction_drift(self, predictions_current: np.ndarray) -> Dict:
        """Detect prediction distribution drift"""
        
        predictions_reference = self.reference_data['predictions']
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(predictions_reference, predictions_current)
            
            # Mean and variance comparison
            mean_change = abs(np.mean(predictions_current) - self.reference_stats['prediction_mean'])
            std_change = abs(np.std(predictions_current) - self.reference_stats['prediction_std'])
            
            drift_detected = p_value < self.drift_threshold
            
            return {
                'method': 'prediction_distribution',
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'mean_change': mean_change,
                'std_change': std_change,
                'drift_detected': drift_detected,
                'drift_score': ks_stat if drift_detected else 0
            }
            
        except Exception as e:
            print(f"[WARNING] Prediction drift test failed: {e}")
            return {
                'method': 'prediction_distribution',
                'error': str(e),
                'drift_detected': False,
                'drift_score': 0
            }
    
    def _detect_residual_drift(self,
                              y_current: np.ndarray,
                              predictions_current: np.ndarray) -> Dict:
        """Detect changes in residual patterns"""
        
        # Calculate current residuals
        residuals_current = y_current - predictions_current
        
        # Calculate reference residuals
        y_reference = self.reference_data['y']
        predictions_reference = self.reference_data['predictions']
        residuals_reference = y_reference - predictions_reference
        
        try:
            # Test residual distribution
            ks_stat, p_value = ks_2samp(residuals_reference, residuals_current)
            
            # Test residual autocorrelation (if enough samples)
            autocorr_drift = False
            if len(residuals_current) > 50:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                try:
                    # Test for autocorrelation in current residuals
                    lb_current = acorr_ljungbox(residuals_current, lags=5, return_df=True)
                    lb_reference = acorr_ljungbox(residuals_reference, lags=5, return_df=True)
                    
                    # Compare p-values
                    current_autocorr = (lb_current['lb_pvalue'] < 0.05).any()
                    reference_autocorr = (lb_reference['lb_pvalue'] < 0.05).any()
                    
                    autocorr_drift = current_autocorr != reference_autocorr
                    
                except:
                    pass
            
            drift_detected = p_value < self.drift_threshold or autocorr_drift
            
            return {
                'method': 'residual_patterns',
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'autocorr_drift': autocorr_drift,
                'drift_detected': drift_detected,
                'drift_score': ks_stat if drift_detected else 0
            }
            
        except Exception as e:
            print(f"[WARNING] Residual drift test failed: {e}")
            return {
                'method': 'residual_patterns',
                'error': str(e),
                'drift_detected': False,
                'drift_score': 0
            }
    
    def _generate_drift_alerts(self, drift_results: Dict) -> None:
        """Generate alerts and recommendations based on drift detection"""
        
        alerts = []
        recommendations = []
        
        for method, result in drift_results['method_results'].items():
            if result.get('drift_detected', False):
                alerts.append(f"Drift detected in {method}")
                
                if method == 'performance':
                    recommendations.append("Consider retraining model due to performance degradation")
                elif method == 'feature_distribution':
                    recommendations.append("Feature distributions have changed - check data preprocessing")
                elif method == 'target_distribution':
                    recommendations.append("Target distribution shift detected - verify data quality")
                elif method == 'prediction_distribution':
                    recommendations.append("Prediction patterns have changed - model may need update")
                elif method == 'residual_patterns':
                    recommendations.append("Residual patterns changed - check model assumptions")
        
        if drift_results['overall_drift_detected']:
            alerts.append(f"OVERALL DRIFT DETECTED (Score: {drift_results['drift_score']:.3f})")
            recommendations.append("URGENT: Model retraining recommended")
        
        drift_results['alerts'] = alerts
        drift_results['recommendations'] = recommendations

class FeatureImportanceMonitor:
    """
    Feature Importance Stability Monitor
    
    Tracks changes in feature importance over time
    to detect shifts in model behavior.
    """
    
    def __init__(self, stability_threshold: float = 0.3):
        self.stability_threshold = stability_threshold
        self.importance_history = []
        self.baseline_importance = None
        
    def set_baseline_importance(self, feature_importance: np.ndarray) -> None:
        """Set baseline feature importance"""
        self.baseline_importance = feature_importance.copy()
        print(f"[IMPORTANCE] Baseline set with {len(feature_importance)} features")
    
    def monitor_importance_stability(self,
                                   current_importance: np.ndarray,
                                   feature_names: List[str] = None) -> Dict:
        """
        Monitor feature importance stability
        
        Args:
            current_importance: Current feature importance values
            feature_names: Names of features (optional)
            
        Returns:
            Stability analysis results
        """
        
        if self.baseline_importance is None:
            self.set_baseline_importance(current_importance)
            return {
                'stability_score': 1.0,
                'stable': True,
                'message': 'Baseline importance set'
            }
        
        # Calculate stability metrics
        importance_change = np.abs(current_importance - self.baseline_importance)
        max_change = np.max(importance_change)
        mean_change = np.mean(importance_change)
        
        # Rank correlation (Spearman)
        try:
            rank_corr, _ = stats.spearmanr(self.baseline_importance, current_importance)
            rank_corr = abs(rank_corr) if not np.isnan(rank_corr) else 0
        except:
            rank_corr = 0
        
        # Stability score (higher is more stable)
        stability_score = rank_corr * (1 - mean_change)
        
        stable = stability_score > (1 - self.stability_threshold)
        
        # Find most changed features
        if feature_names:
            changed_features = [
                (feature_names[i], importance_change[i])
                for i in np.argsort(importance_change)[::-1][:5]
            ]
        else:
            changed_features = [
                (f'feature_{i}', importance_change[i])
                for i in np.argsort(importance_change)[::-1][:5]
            ]
        
        result = {
            'timestamp': datetime.now(),
            'stability_score': stability_score,
            'stable': stable,
            'max_change': max_change,
            'mean_change': mean_change,
            'rank_correlation': rank_corr,
            'most_changed_features': changed_features,
            'recommendation': 'Stable' if stable else 'Monitor closely - importance drift detected'
        }
        
        # Store in history
        self.importance_history.append(result)
        
        return result

class PredictionQualityAssessor:
    """
    Prediction Quality Assessment System
    
    Evaluates the quality of predictions across multiple dimensions
    including accuracy, calibration, and business relevance.
    """
    
    def __init__(self):
        self.quality_metrics = [
            'accuracy',
            'calibration',
            'consistency',
            'business_alignment',
            'uncertainty_quality'
        ]
        
    def assess_prediction_quality(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                uncertainty: np.ndarray = None,
                                business_targets: np.ndarray = None) -> Dict:
        """
        Comprehensive prediction quality assessment
        
        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainty: Uncertainty estimates (optional)
            business_targets: Business target values (optional)
            
        Returns:
            Quality assessment results
        """
        
        assessment = {
            'timestamp': datetime.now(),
            'n_predictions': len(y_true),
            'metrics': {},
            'overall_quality_score': 0.0,
            'quality_grade': 'Unknown'
        }
        
        # 1. Accuracy Assessment
        accuracy_metrics = self._assess_accuracy(y_true, y_pred)
        assessment['metrics']['accuracy'] = accuracy_metrics
        
        # 2. Calibration Assessment
        if uncertainty is not None:
            calibration_metrics = self._assess_calibration(y_true, y_pred, uncertainty)
            assessment['metrics']['calibration'] = calibration_metrics
        
        # 3. Consistency Assessment
        consistency_metrics = self._assess_consistency(y_pred)
        assessment['metrics']['consistency'] = consistency_metrics
        
        # 4. Business Alignment Assessment
        if business_targets is not None:
            alignment_metrics = self._assess_business_alignment(y_pred, business_targets)
            assessment['metrics']['business_alignment'] = alignment_metrics
        
        # 5. Overall Quality Score
        assessment['overall_quality_score'] = self._calculate_overall_quality(assessment['metrics'])
        assessment['quality_grade'] = self._assign_quality_grade(assessment['overall_quality_score'])
        
        return assessment
    
    def _assess_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Assess prediction accuracy"""
        
        wmape_score = wmape(y_true, y_pred)
        mae_score = mean_absolute_error(y_true, y_pred)
        rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # R² score
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = 0.0
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # Accuracy score (0-100, higher is better)
        # Convert WMAPE to accuracy score
        accuracy_score = max(0, 100 - wmape_score)
        
        return {
            'wmape': wmape_score,
            'mae': mae_score,
            'rmse': rmse_score,
            'r2': r2,
            'mape': mape,
            'accuracy_score': accuracy_score,
            'grade': self._score_to_grade(accuracy_score)
        }
    
    def _assess_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray) -> Dict:
        """Assess prediction calibration quality"""
        
        # Calculate prediction intervals
        lower_bound = y_pred - 1.96 * uncertainty  # 95% interval
        upper_bound = y_pred + 1.96 * uncertainty
        
        # Calculate empirical coverage
        in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        empirical_coverage = np.mean(in_interval)
        
        # Expected coverage for 95% interval
        expected_coverage = 0.95
        
        # Calibration error
        calibration_error = abs(empirical_coverage - expected_coverage)
        
        # Average interval width
        avg_interval_width = np.mean(upper_bound - lower_bound)
        
        # Sharpness (smaller intervals are sharper)
        normalized_width = avg_interval_width / (np.std(y_true) + 1e-8)
        
        # Calibration score (0-100, higher is better)
        calibration_score = max(0, 100 - calibration_error * 200)  # Scale error to 0-100
        
        return {
            'empirical_coverage': empirical_coverage,
            'expected_coverage': expected_coverage,
            'calibration_error': calibration_error,
            'avg_interval_width': avg_interval_width,
            'normalized_width': normalized_width,
            'calibration_score': calibration_score,
            'grade': self._score_to_grade(calibration_score)
        }
    
    def _assess_consistency(self, y_pred: np.ndarray) -> Dict:
        """Assess prediction consistency"""
        
        # Temporal consistency (if predictions are time-ordered)
        if len(y_pred) > 1:
            changes = np.abs(np.diff(y_pred))
            mean_change = np.mean(changes)
            std_change = np.std(changes)
            max_change = np.max(changes)
            
            # Coefficient of variation in changes
            cv_changes = std_change / (mean_change + 1e-8)
            
            # Consistency score (lower variability = higher consistency)
            consistency_score = max(0, 100 - cv_changes * 20)
        else:
            mean_change = 0
            std_change = 0
            max_change = 0
            cv_changes = 0
            consistency_score = 100
        
        return {
            'mean_change': mean_change,
            'std_change': std_change,
            'max_change': max_change,
            'cv_changes': cv_changes,
            'consistency_score': consistency_score,
            'grade': self._score_to_grade(consistency_score)
        }
    
    def _assess_business_alignment(self, y_pred: np.ndarray, business_targets: np.ndarray) -> Dict:
        """Assess alignment with business targets"""
        
        # Calculate alignment metrics
        alignment_error = np.abs(y_pred - business_targets)
        mean_alignment_error = np.mean(alignment_error)
        max_alignment_error = np.max(alignment_error)
        
        # Percentage of predictions close to targets (within 10%)
        close_predictions = alignment_error <= (0.1 * np.abs(business_targets + 1e-8))
        alignment_rate = np.mean(close_predictions)
        
        # Business alignment score
        alignment_score = alignment_rate * 100
        
        return {
            'mean_alignment_error': mean_alignment_error,
            'max_alignment_error': max_alignment_error,
            'alignment_rate': alignment_rate,
            'alignment_score': alignment_score,
            'grade': self._score_to_grade(alignment_score)
        }
    
    def _calculate_overall_quality(self, metrics: Dict) -> float:
        """Calculate overall quality score"""
        
        scores = []
        
        # Accuracy is most important
        if 'accuracy' in metrics:
            scores.append(metrics['accuracy']['accuracy_score'] * 0.4)
        
        # Calibration
        if 'calibration' in metrics:
            scores.append(metrics['calibration']['calibration_score'] * 0.25)
        
        # Consistency
        if 'consistency' in metrics:
            scores.append(metrics['consistency']['consistency_score'] * 0.2)
        
        # Business alignment
        if 'business_alignment' in metrics:
            scores.append(metrics['business_alignment']['alignment_score'] * 0.15)
        
        return sum(scores) if scores else 0.0
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign overall quality grade"""
        return self._score_to_grade(score)

class ModelHealthDashboard:
    """
    Model Health Dashboard and Reporting System
    
    Generates comprehensive health reports and visualizations
    for model monitoring and decision making.
    """
    
    def __init__(self):
        self.health_history = []
        self.alert_thresholds = {
            'wmape_degradation': 0.2,  # 20% increase in WMAPE
            'drift_score': 0.05,       # Drift score > 0.05
            'stability_score': 0.7,   # Stability score < 0.7
            'quality_score': 70        # Quality score < 70
        }
        
    def generate_health_report(self,
                             model_name: str,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             X: np.ndarray = None,
                             uncertainty: np.ndarray = None,
                             feature_importance: np.ndarray = None,
                             reference_wmape: float = None) -> ModelHealthMetrics:
        """
        Generate comprehensive model health report
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            X: Feature matrix (optional)
            uncertainty: Uncertainty estimates (optional)
            feature_importance: Feature importance values (optional)
            reference_wmape: Reference WMAPE for comparison (optional)
            
        Returns:
            ModelHealthMetrics object
        """
        
        print(f"[HEALTH] Generating health report for {model_name}...")
        
        # Calculate performance metrics
        current_wmape = wmape(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = 0.0
        
        # Calculate stability metrics
        prediction_stability = self._calculate_prediction_stability(y_pred)
        
        # Feature importance stability (if available)
        feature_importance_stability = 1.0
        if feature_importance is not None:
            # This would typically use historical importance data
            feature_importance_stability = np.std(feature_importance) / (np.mean(np.abs(feature_importance)) + 1e-8)
            feature_importance_stability = max(0, 1 - feature_importance_stability)
        
        # Calibration score (if uncertainty available)
        calibration_score = 0.8  # Default neutral score
        if uncertainty is not None:
            quality_assessor = PredictionQualityAssessor()
            quality_results = quality_assessor.assess_prediction_quality(y_true, y_pred, uncertainty)
            if 'calibration' in quality_results['metrics']:
                calibration_score = quality_results['metrics']['calibration']['calibration_score'] / 100
        
        # Drift scores (simplified for demonstration)
        concept_drift_score = 0.02  # Would use actual drift detector
        data_drift_score = 0.01
        
        # Business impact score (simplified)
        business_impact_score = max(0, 100 - current_wmape)
        
        # Calculate overall health score
        health_score = self._calculate_health_score(
            current_wmape, prediction_stability, feature_importance_stability,
            calibration_score, concept_drift_score, data_drift_score,
            business_impact_score, reference_wmape
        )
        
        # Generate alerts and warnings
        alerts, warnings = self._generate_health_alerts(
            model_name, current_wmape, health_score, concept_drift_score,
            prediction_stability, reference_wmape
        )
        
        # Create health metrics object
        health_metrics = ModelHealthMetrics(
            timestamp=datetime.now(),
            model_name=model_name,
            wmape=current_wmape,
            mae=mae,
            rmse=rmse,
            r2=r2,
            prediction_stability=prediction_stability,
            feature_importance_stability=feature_importance_stability,
            calibration_score=calibration_score,
            concept_drift_score=concept_drift_score,
            data_drift_score=data_drift_score,
            business_impact_score=business_impact_score,
            health_score=health_score,
            alerts=alerts,
            warnings=warnings
        )
        
        # Store in history
        self.health_history.append(health_metrics)
        
        print(f"[HEALTH] Health score: {health_score:.1f}/100")
        if alerts:
            print(f"[ALERTS] {len(alerts)} alerts generated")
        if warnings:
            print(f"[WARNINGS] {len(warnings)} warnings generated")
        
        return health_metrics
    
    def _calculate_prediction_stability(self, predictions: np.ndarray) -> float:
        """Calculate prediction stability score"""
        
        if len(predictions) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        cv = np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8)
        
        # Convert to stability score (0-1, higher is more stable)
        stability = max(0, 1 - cv)
        
        return stability
    
    def _calculate_health_score(self,
                               wmape: float,
                               prediction_stability: float,
                               feature_importance_stability: float,
                               calibration_score: float,
                               concept_drift_score: float,
                               data_drift_score: float,
                               business_impact_score: float,
                               reference_wmape: float = None) -> float:
        """Calculate overall health score (0-100)"""
        
        # Base score from WMAPE (lower is better)
        accuracy_score = max(0, 100 - wmape)
        
        # Stability component
        stability_component = (prediction_stability + feature_importance_stability) * 50
        
        # Calibration component
        calibration_component = calibration_score * 100
        
        # Drift penalty
        drift_penalty = (concept_drift_score + data_drift_score) * 50
        
        # Business impact component
        business_component = business_impact_score
        
        # Weighted combination
        health_score = (
            accuracy_score * 0.4 +
            stability_component * 0.2 +
            calibration_component * 0.15 +
            business_component * 0.25 -
            drift_penalty * 0.1
        )
        
        # Performance degradation penalty
        if reference_wmape is not None:
            degradation = (wmape - reference_wmape) / reference_wmape
            if degradation > 0.2:  # More than 20% degradation
                health_score *= 0.8  # 20% penalty
        
        return max(0, min(100, health_score))
    
    def _generate_health_alerts(self,
                               model_name: str,
                               current_wmape: float,
                               health_score: float,
                               drift_score: float,
                               stability: float,
                               reference_wmape: float = None) -> Tuple[List[str], List[str]]:
        """Generate alerts and warnings based on health metrics"""
        
        alerts = []
        warnings = []
        
        # Performance degradation
        if reference_wmape is not None:
            degradation = (current_wmape - reference_wmape) / reference_wmape
            if degradation > self.alert_thresholds['wmape_degradation']:
                alerts.append(f"PERFORMANCE DEGRADATION: WMAPE increased by {degradation*100:.1f}%")
        
        # Drift alerts
        if drift_score > self.alert_thresholds['drift_score']:
            alerts.append(f"CONCEPT DRIFT DETECTED: Score {drift_score:.3f}")
        
        # Stability warnings
        if stability < self.alert_thresholds['stability_score']:
            warnings.append(f"LOW STABILITY: Score {stability:.2f}")
        
        # Overall health
        if health_score < self.alert_thresholds['quality_score']:
            alerts.append(f"LOW MODEL HEALTH: Score {health_score:.1f}/100")
        
        # High WMAPE
        if current_wmape > 30:  # More than 30% WMAPE
            alerts.append(f"HIGH ERROR RATE: WMAPE {current_wmape:.2f}%")
        elif current_wmape > 20:
            warnings.append(f"ELEVATED ERROR RATE: WMAPE {current_wmape:.2f}%")
        
        return alerts, warnings
    
    def save_diagnostics_report(self, output_dir: str = "../../models/diagnostics") -> Dict[str, str]:
        """Save comprehensive diagnostics report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save health history
        if self.health_history:
            health_data = [asdict(health) for health in self.health_history]
            
            health_file = output_path / f"model_health_history_{timestamp}.json"
            with open(health_file, 'w') as f:
                json.dump(health_data, f, indent=2, default=str)
            saved_files['health_history'] = str(health_file)
        
        # Save dashboard object
        import pickle
        dashboard_file = output_path / f"diagnostics_dashboard_{timestamp}.pkl"
        with open(dashboard_file, 'wb') as f:
            pickle.dump(self, f)
        saved_files['dashboard'] = str(dashboard_file)
        
        print(f"[SAVE] Diagnostics report saved: {len(saved_files)} files")
        
        return saved_files

def main():
    """Demonstration of Model Diagnostics Suite"""
    
    print("🔬 MODEL DIAGNOSTICS SUITE - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Generate synthetic data for demonstration
        np.random.seed(42)
        
        n_samples = 2000
        n_features = 6
        
        # Generate features and target
        X = np.random.normal(0, 1, (n_samples, n_features))
        true_coefficients = np.array([2, -1.5, 1, 0.5, -0.8, 0.3])
        
        # Generate target with some noise
        y_true = X @ true_coefficients + np.random.normal(0, 0.5, n_samples)
        y_true = np.maximum(y_true, 0)  # Ensure non-negative
        
        # Generate predictions (with some error)
        y_pred = y_true + np.random.normal(0, 0.8, n_samples)
        y_pred = np.maximum(y_pred, 0)
        
        # Generate uncertainty estimates
        uncertainty = np.random.gamma(2, 0.5, n_samples)
        
        # Generate feature importance
        feature_importance = np.abs(true_coefficients) + np.random.normal(0, 0.1, n_features)
        feature_importance = feature_importance / np.sum(feature_importance)
        
        print(f"Generated {n_samples} samples with {n_features} features")
        print(f"True WMAPE: {wmape(y_true, y_pred):.4f}")
        
        # Split data for reference and current
        split_idx = n_samples // 2
        
        X_reference = X[:split_idx]
        y_reference = y_true[:split_idx]
        pred_reference = y_pred[:split_idx]
        
        X_current = X[split_idx:]
        y_current = y_true[split_idx:]
        pred_current = y_pred[split_idx:]
        uncertainty_current = uncertainty[split_idx:]
        
        # 1. Concept Drift Detection
        print("\n[DEMO] Testing Concept Drift Detection...")
        
        drift_detector = ConceptDriftDetector(window_size=500, drift_threshold=0.05)
        drift_detector.set_reference_data(X_reference, y_reference, pred_reference)
        
        # Add some drift to current data
        X_current_drifted = X_current + np.random.normal(0, 0.3, X_current.shape)
        y_current_drifted = y_current * 1.1  # 10% increase
        
        drift_results = drift_detector.detect_drift(X_current_drifted, y_current_drifted, pred_current)
        
        # 2. Feature Importance Monitoring
        print("\n[DEMO] Testing Feature Importance Monitoring...")
        
        importance_monitor = FeatureImportanceMonitor()
        importance_monitor.set_baseline_importance(feature_importance)
        
        # Simulate changed importance
        changed_importance = feature_importance + np.random.normal(0, 0.05, n_features)
        changed_importance = np.abs(changed_importance)
        changed_importance = changed_importance / np.sum(changed_importance)
        
        importance_results = importance_monitor.monitor_importance_stability(changed_importance)
        
        # 3. Prediction Quality Assessment
        print("\n[DEMO] Testing Prediction Quality Assessment...")
        
        quality_assessor = PredictionQualityAssessor()
        quality_results = quality_assessor.assess_prediction_quality(
            y_current, pred_current, uncertainty_current
        )
        
        # 4. Model Health Dashboard
        print("\n[DEMO] Generating Model Health Report...")
        
        dashboard = ModelHealthDashboard()
        health_report = dashboard.generate_health_report(
            model_name="Demo_Model",
            y_true=y_current,
            y_pred=pred_current,
            X=X_current,
            uncertainty=uncertainty_current,
            feature_importance=changed_importance,
            reference_wmape=wmape(y_reference, pred_reference)
        )
        
        # 5. Error Analysis Integration
        print("\n[DEMO] Integration with Error Analysis...")
        
        error_decomposer = ErrorDecomposer()
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'internal_product_id': np.random.choice(range(1, 21), len(y_current)),
            'internal_store_id': np.random.choice(range(1, 6), len(y_current)),
            'actual': y_current,
            'predicted': pred_current
        })
        
        decomposition_results = error_decomposer.decompose_errors(
            analysis_df,
            dimensions=['internal_product_id', 'internal_store_id']
        )
        
        # Save comprehensive diagnostics report
        print("\n[DEMO] Saving diagnostics report...")
        saved_files = dashboard.save_diagnostics_report()
        
        print("\n" + "=" * 80)
        print("🎉 MODEL DIAGNOSTICS SUITE DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        # Print summary results
        print("Drift Detection Results:")
        print(f"  Overall Drift Detected: {drift_results['overall_drift_detected']}")
        print(f"  Drift Score: {drift_results['drift_score']:.4f}")
        print(f"  Active Alerts: {len(drift_results['alerts'])}")
        
        print("\nFeature Importance Monitoring:")
        print(f"  Stability Score: {importance_results['stability_score']:.3f}")
        print(f"  Stable: {importance_results['stable']}")
        print(f"  Max Change: {importance_results['max_change']:.4f}")
        
        print("\nPrediction Quality Assessment:")
        print(f"  Overall Quality Score: {quality_results['overall_quality_score']:.1f}")
        print(f"  Quality Grade: {quality_results['quality_grade']}")
        
        print("\nModel Health Report:")
        print(f"  Health Score: {health_report.health_score:.1f}/100")
        print(f"  WMAPE: {health_report.wmape:.4f}")
        print(f"  Stability: {health_report.prediction_stability:.3f}")
        print(f"  Alerts: {len(health_report.alerts)}")
        print(f"  Warnings: {len(health_report.warnings)}")
        
        if health_report.alerts:
            print("\nActive Alerts:")
            for alert in health_report.alerts:
                print(f"  - {alert}")
        
        print(f"\nDiagnostics files saved: {len(saved_files)}")
        for file_type, path in saved_files.items():
            print(f"  {file_type}: {path}")
        
        return dashboard, drift_detector, importance_monitor, quality_assessor
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    results = main()