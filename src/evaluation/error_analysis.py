#!/usr/bin/env python3
"""
ERROR ANALYSIS ENGINE - Hackathon Forecast Big Data 2025
Advanced Error Decomposition and Residual Analysis System

Features:
- Multi-dimensional error decomposition (product, PDV, time, volume)
- Statistical tests for residuals (Ljung-Box, Jarque-Bera, ARCH, Runs)
- Visual diagnostics (Q-Q plots, residual plots, ACF/PACF)
- Pattern detection (bias, heteroskedasticity, autocorrelation)
- Seasonal error analysis and trend detection
- Performance degradation monitoring
- Business-aware error segmentation

The DIAGNOSTIC CENTER for model performance! 🔍
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
try:
    from statsmodels.stats.runs import runstest_1samp
except ImportError:
    # runstest is not available in newer statsmodels versions
    # We'll implement a simple alternative
    def runstest_1samp(x, cutoff='median'):
        """Simple runs test implementation"""
        import numpy as np
        x = np.array(x)
        if cutoff == 'median':
            cutoff = np.median(x)
        elif cutoff == 'mean':
            cutoff = np.mean(x)

        runs, n1, n2 = 0, 0, 0
        # Convert to binary
        binary = x > cutoff
        n1 = np.sum(binary)
        n2 = len(x) - n1

        # Count runs
        if len(binary) > 0:
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i-1]:
                    runs += 1

        # Expected runs and variance
        expected = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

        # Z-score
        if variance > 0:
            z_score = (runs - expected) / np.sqrt(variance)
        else:
            z_score = 0

        # P-value (approximate, two-tailed)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        return z_score, p_value
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our utilities
from src.evaluation.metrics import wmape
from src.utils.data_loader import load_data_efficiently

warnings.filterwarnings('ignore')

class ErrorDecomposer:
    """
    Multi-Dimensional Error Decomposition Engine
    
    Decomposes prediction errors across different business dimensions
    to identify systematic patterns and problem areas.
    """
    
    def __init__(self):
        self.decomposition_results = {}
        self.error_patterns = {}
        self.problematic_segments = {}
        
    def decompose_errors(self, 
                        df: pd.DataFrame,
                        actual_col: str = 'actual',
                        predicted_col: str = 'predicted',
                        dimensions: List[str] = None) -> Dict[str, Dict]:
        """
        Decompose errors across multiple business dimensions
        
        Args:
            df: DataFrame with actual and predicted values
            actual_col: Column name for actual values
            predicted_col: Column name for predicted values
            dimensions: List of columns to decompose by
            
        Returns:
            Dictionary with decomposition results
        """
        
        print("[INFO] Decomposing errors across business dimensions...")
        
        # Calculate basic errors
        df = df.copy()
        df['error'] = df[actual_col] - df[predicted_col]
        df['abs_error'] = np.abs(df['error'])
        df['pct_error'] = df['error'] / (np.abs(df[actual_col]) + 1e-8) * 100
        df['abs_pct_error'] = np.abs(df['pct_error'])
        
        # Default dimensions if not specified
        if dimensions is None:
            dimensions = [col for col in df.columns 
                         if col in ['internal_product_id', 'internal_store_id', 'categoria', 
                                   'transaction_date', 'volume_tier']]
        
        decomposition_results = {}
        
        # Decompose by each dimension
        for dimension in dimensions:
            if dimension not in df.columns:
                continue
                
            print(f"[DECOMPOSE] Analyzing errors by {dimension}...")
            
            # Aggregate errors by dimension
            dim_errors = df.groupby(dimension).agg({
                'error': ['mean', 'std', 'min', 'max'],
                'abs_error': ['mean', 'std', 'sum'],
                'pct_error': ['mean', 'std'],
                'abs_pct_error': ['mean', 'std'],
                actual_col: ['count', 'sum', 'mean'],
                predicted_col: ['sum', 'mean']
            }).round(4)
            
            # Flatten column names
            dim_errors.columns = ['_'.join(col).strip() for col in dim_errors.columns]
            
            # Calculate additional metrics
            dim_errors['wmape'] = df.groupby(dimension).apply(
                lambda x: wmape(x[actual_col].values, x[predicted_col].values)
            ).round(4)
            
            dim_errors['bias'] = dim_errors['error_mean']
            dim_errors['rmse'] = np.sqrt(
                df.groupby(dimension)['error'].apply(lambda x: np.mean(x**2))
            ).round(4)
            
            # Identify problematic segments
            problematic_threshold = dim_errors['wmape'].quantile(0.8)
            problematic_segments = dim_errors[dim_errors['wmape'] > problematic_threshold]
            
            decomposition_results[dimension] = {
                'summary': dim_errors,
                'problematic_segments': problematic_segments,
                'best_segments': dim_errors.nsmallest(5, 'wmape'),
                'worst_segments': dim_errors.nlargest(5, 'wmape'),
                'dimension_stats': {
                    'total_segments': len(dim_errors),
                    'problematic_count': len(problematic_segments),
                    'avg_wmape': dim_errors['wmape'].mean(),
                    'wmape_std': dim_errors['wmape'].std(),
                    'max_wmape': dim_errors['wmape'].max(),
                    'min_wmape': dim_errors['wmape'].min()
                }
            }
            
            print(f"[OK] {dimension}: {len(dim_errors)} segments, "
                  f"avg WMAPE: {dim_errors['wmape'].mean():.4f}")
        
        # Cross-dimensional analysis
        if len(dimensions) >= 2:
            decomposition_results['cross_analysis'] = self._cross_dimensional_analysis(
                df, dimensions[:2], actual_col, predicted_col
            )
        
        self.decomposition_results = decomposition_results
        
        return decomposition_results
    
    def _cross_dimensional_analysis(self, 
                                   df: pd.DataFrame, 
                                   dimensions: List[str],
                                   actual_col: str,
                                   predicted_col: str) -> Dict:
        """Analyze errors across multiple dimensions simultaneously"""
        
        if len(dimensions) < 2:
            return {}
        
        dim1, dim2 = dimensions[0], dimensions[1]
        
        # Create pivot table of WMAPE
        cross_wmape = df.groupby([dim1, dim2]).apply(
            lambda x: wmape(x[actual_col].values, x[predicted_col].values) if len(x) > 0 else np.nan
        ).unstack(fill_value=np.nan)
        
        # Find interaction effects
        interaction_analysis = {
            'cross_wmape_matrix': cross_wmape,
            'worst_combinations': [],
            'best_combinations': []
        }
        
        # Find worst and best combinations
        wmape_values = []
        for idx in cross_wmape.index:
            for col in cross_wmape.columns:
                value = cross_wmape.loc[idx, col]
                if not np.isnan(value):
                    wmape_values.append((idx, col, value))
        
        if wmape_values:
            wmape_values.sort(key=lambda x: x[2])
            interaction_analysis['best_combinations'] = wmape_values[:5]
            interaction_analysis['worst_combinations'] = wmape_values[-5:]
        
        return interaction_analysis
    
    def analyze_temporal_patterns(self, 
                                 df: pd.DataFrame,
                                 actual_col: str = 'actual',
                                 predicted_col: str = 'predicted',
                                 date_col: str = 'transaction_date') -> Dict:
        """Analyze temporal patterns in errors"""
        
        print("[INFO] Analyzing temporal error patterns...")
        
        if date_col not in df.columns:
            return {}
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['error'] = df[actual_col] - df[predicted_col]
        df['abs_error'] = np.abs(df['error'])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Extract temporal features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        temporal_analysis = {}
        
        # Monthly patterns
        monthly_errors = df.groupby('month').agg({
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'std'],
            actual_col: 'count'
        }).round(4)
        monthly_errors.columns = ['_'.join(col) for col in monthly_errors.columns]
        temporal_analysis['monthly'] = monthly_errors
        
        # Weekly patterns  
        weekly_errors = df.groupby('day_of_week').agg({
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'std'],
            actual_col: 'count'
        }).round(4)
        weekly_errors.columns = ['_'.join(col) for col in weekly_errors.columns]
        temporal_analysis['weekly'] = weekly_errors
        
        # Trend analysis
        if len(df) >= 30:
            # Rolling error metrics
            df_daily = df.groupby(date_col).agg({
                'error': 'mean',
                'abs_error': 'mean',
                actual_col: ['count', 'sum'],
                predicted_col: 'sum'
            }).reset_index()
            
            df_daily.columns = ['_'.join(col).strip('_') for col in df_daily.columns]
            
            # Calculate rolling WMAPE
            window_size = min(7, len(df_daily) // 4)
            if window_size >= 3:
                df_daily['rolling_wmape'] = df_daily.apply(
                    lambda x: wmape(
                        df_daily['actual_sum'].rolling(window_size, center=True).apply(lambda y: y.iloc[-1] if len(y) > 0 else np.nan),
                        df_daily['predicted_sum'].rolling(window_size, center=True).apply(lambda y: y.iloc[-1] if len(y) > 0 else np.nan)
                    ) if not np.isnan(x['error_mean']) else np.nan, axis=1
                )
                
                temporal_analysis['trend'] = {
                    'daily_errors': df_daily,
                    'trend_detected': self._detect_trend(df_daily['error_mean'].dropna()),
                    'seasonality_detected': self._detect_seasonality(df_daily['error_mean'].dropna())
                }
        
        return temporal_analysis

    def _detect_trend(self, series: pd.Series) -> Dict:
        """Detect trend in error series"""
        
        if len(series) < 10:
            return {'trend_detected': False}
        
        # Mann-Kendall trend test
        n = len(series)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(series.iloc[j] - series.iloc[i])
        
        var_s = n * (n-1) * (2*n+5) / 18
        z = s / np.sqrt(var_s) if var_s > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'trend_detected': p_value < 0.05,
            'trend_direction': 'increasing' if s > 0 else 'decreasing',
            'trend_strength': abs(z),
            'p_value': p_value
        }
    
    def _detect_seasonality(self, series: pd.Series, period: int = 7) -> Dict:
        """Detect seasonality in error series"""
        
        if len(series) < 2 * period:
            return {'seasonality_detected': False}
        
        # Simple seasonality test using autocorrelation
        try:
            autocorr_seasonal = acf(series.dropna(), nlags=period, fft=False)[period]
            
            return {
                'seasonality_detected': abs(autocorr_seasonal) > 0.2,
                'seasonal_strength': abs(autocorr_seasonal),
                'period': period
            }
        except:
            return {'seasonality_detected': False}

    def analyze_volume_patterns(self,
                               df: pd.DataFrame,
                               actual_col: str = 'actual',
                               predicted_col: str = 'predicted',
                               volume_thresholds: List[float] = None) -> Dict[str, Any]:
        """
        Analyze error patterns by volume (small vs large sales) - PHASE 5 REQUIREMENT

        Args:
            df: DataFrame with actual and predicted columns
            actual_col: Name of actual values column
            predicted_col: Name of predicted values column
            volume_thresholds: Volume thresholds for segmentation

        Returns:
            Volume-based error analysis results
        """

        print("[VOLUME] Analyzing error patterns by volume segments...")

        if volume_thresholds is None:
            # Define volume segments based on quantiles
            volume_thresholds = [
                df[actual_col].quantile(0.33),  # Small volume threshold
                df[actual_col].quantile(0.67)   # Large volume threshold
            ]

        # Create volume segments
        df = df.copy()
        df['volume_segment'] = pd.cut(
            df[actual_col],
            bins=[-np.inf] + volume_thresholds + [np.inf],
            labels=['Small', 'Medium', 'Large']
        )

        # Calculate errors by volume segment
        df['error'] = df[actual_col] - df[predicted_col]
        df['abs_error'] = np.abs(df['error'])
        df['pct_error'] = df['abs_error'] / (np.abs(df[actual_col]) + 1e-8) * 100

        volume_analysis = {}

        for segment in ['Small', 'Medium', 'Large']:
            segment_data = df[df['volume_segment'] == segment]

            if len(segment_data) > 0:
                volume_analysis[segment] = {
                    'count': len(segment_data),
                    'volume_range': {
                        'min': segment_data[actual_col].min(),
                        'max': segment_data[actual_col].max(),
                        'mean': segment_data[actual_col].mean()
                    },
                    'error_metrics': {
                        'wmape': wmape(segment_data[actual_col], segment_data[predicted_col]),
                        'mae': segment_data['abs_error'].mean(),
                        'mape': segment_data['pct_error'].mean(),
                        'bias': segment_data['error'].mean(),
                        'std_error': segment_data['error'].std()
                    },
                    'error_patterns': {
                        'overestimate_pct': (segment_data['error'] < 0).mean() * 100,
                        'underestimate_pct': (segment_data['error'] > 0).mean() * 100,
                        'large_errors_pct': (segment_data['pct_error'] > 50).mean() * 100
                    }
                }

        # Cross-segment analysis
        segment_wmapes = [
            volume_analysis[seg]['error_metrics']['wmape']
            for seg in volume_analysis.keys()
        ]

        results = {
            'volume_segments': volume_analysis,
            'volume_thresholds': volume_thresholds,
            'summary': {
                'segments_count': len(volume_analysis),
                'total_observations': len(df),
                'best_performing_segment': min(volume_analysis.keys(),
                                             key=lambda x: volume_analysis[x]['error_metrics']['wmape']),
                'worst_performing_segment': max(volume_analysis.keys(),
                                              key=lambda x: volume_analysis[x]['error_metrics']['wmape']),
                'wmape_range': {
                    'min': min(segment_wmapes),
                    'max': max(segment_wmapes),
                    'difference': max(segment_wmapes) - min(segment_wmapes)
                }
            },
            'insights': []
        }

        # Generate insights
        if results['summary']['wmape_range']['difference'] > 5:  # > 5% WMAPE difference
            results['insights'].append(f"Significant volume-based performance difference: "
                                     f"{results['summary']['wmape_range']['difference']:.1f}% WMAPE range")

        if 'Small' in volume_analysis and 'Large' in volume_analysis:
            small_wmape = volume_analysis['Small']['error_metrics']['wmape']
            large_wmape = volume_analysis['Large']['error_metrics']['wmape']

            if small_wmape > large_wmape * 1.2:
                results['insights'].append("Model struggles more with small volume predictions")
            elif large_wmape > small_wmape * 1.2:
                results['insights'].append("Model struggles more with large volume predictions")

        print(f"[VOLUME] Analysis completed: {len(volume_analysis)} segments analyzed")

        return results

    def detect_systematic_bias(self,
                              df: pd.DataFrame,
                              actual_col: str = 'actual',
                              predicted_col: str = 'predicted',
                              dimensions: List[str] = None) -> Dict[str, Any]:
        """
        Detect systematic bias patterns - PHASE 5 REQUIREMENT

        Args:
            df: DataFrame with predictions and actuals
            actual_col: Name of actual values column
            predicted_col: Name of predicted values column
            dimensions: List of dimensions to analyze bias across

        Returns:
            Systematic bias analysis results
        """

        print("[BIAS] Detecting systematic bias patterns...")

        df = df.copy()
        df['error'] = df[actual_col] - df[predicted_col]
        df['abs_error'] = np.abs(df['error'])

        bias_analysis = {
            'overall_bias': {},
            'dimensional_bias': {},
            'temporal_bias': {},
            'bias_tests': {}
        }

        # Overall bias analysis
        overall_error = df['error']
        bias_analysis['overall_bias'] = {
            'mean_bias': overall_error.mean(),
            'median_bias': overall_error.median(),
            'bias_percentage': (overall_error.mean() / df[actual_col].mean()) * 100 if df[actual_col].mean() != 0 else 0,
            'systematic_overestimate': (overall_error < 0).mean() * 100,
            'systematic_underestimate': (overall_error > 0).mean() * 100,
            'bias_magnitude': np.abs(overall_error.mean())
        }

        # Statistical tests for bias
        if len(overall_error) > 10:
            # One-sample t-test for bias
            t_stat, t_p_value = stats.ttest_1samp(overall_error, 0)

            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, w_p_value = stats.wilcoxon(overall_error)
            except:
                w_stat, w_p_value = np.nan, np.nan

            bias_analysis['bias_tests'] = {
                'ttest_statistic': t_stat,
                'ttest_p_value': t_p_value,
                'ttest_significant': t_p_value < 0.05,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_p_value': w_p_value,
                'wilcoxon_significant': w_p_value < 0.05 if not np.isnan(w_p_value) else False
            }

        # Dimensional bias analysis
        if dimensions:
            for dimension in dimensions:
                if dimension in df.columns:
                    dim_bias = df.groupby(dimension)['error'].agg([
                        'mean', 'median', 'std', 'count'
                    ]).reset_index()

                    dim_bias['abs_mean_bias'] = np.abs(dim_bias['mean'])
                    dim_bias['bias_significance'] = dim_bias.apply(
                        lambda row: abs(row['mean']) > 2 * (row['std'] / np.sqrt(row['count']))
                        if row['std'] > 0 and row['count'] > 1 else False, axis=1
                    )

                    bias_analysis['dimensional_bias'][dimension] = {
                        'segment_analysis': dim_bias.to_dict('records'),
                        'max_bias_segment': dim_bias.loc[dim_bias['abs_mean_bias'].idxmax(), dimension],
                        'max_bias_value': dim_bias['abs_mean_bias'].max(),
                        'significant_bias_segments': dim_bias[dim_bias['bias_significance']]['count'].sum()
                    }

        # Temporal bias analysis (if date column exists)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            if pd.api.types.is_datetime64_any_dtype(df[date_col]) or df[date_col].dtype == 'object':
                try:
                    df[date_col] = pd.to_datetime(df[date_col])

                    # Monthly bias trend
                    df['month'] = df[date_col].dt.to_period('M')
                    monthly_bias = df.groupby('month')['error'].mean()

                    # Trend in bias over time
                    if len(monthly_bias) > 2:
                        x = np.arange(len(monthly_bias))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, monthly_bias.values)

                        bias_analysis['temporal_bias'] = {
                            'monthly_bias': monthly_bias.to_dict(),
                            'bias_trend': {
                                'slope': slope,
                                'r_squared': r_value ** 2,
                                'p_value': p_value,
                                'trend_significant': p_value < 0.05,
                                'trend_direction': 'increasing' if slope > 0 else 'decreasing'
                            }
                        }
                except Exception as e:
                    print(f"[WARNING] Temporal bias analysis failed: {e}")

        # Bias severity assessment
        bias_severity = 'Low'
        if abs(bias_analysis['overall_bias']['bias_percentage']) > 10:
            bias_severity = 'High'
        elif abs(bias_analysis['overall_bias']['bias_percentage']) > 5:
            bias_severity = 'Medium'

        bias_analysis['assessment'] = {
            'bias_severity': bias_severity,
            'bias_direction': 'Overestimate' if bias_analysis['overall_bias']['mean_bias'] < 0 else 'Underestimate',
            'systematic_bias_detected': (
                bias_analysis.get('bias_tests', {}).get('ttest_significant', False) or
                abs(bias_analysis['overall_bias']['bias_percentage']) > 5
            ),
            'recommendations': []
        }

        # Generate recommendations
        if bias_analysis['assessment']['systematic_bias_detected']:
            if bias_analysis['assessment']['bias_direction'] == 'Overestimate':
                bias_analysis['assessment']['recommendations'].append("Model consistently overestimates - consider bias correction")
            else:
                bias_analysis['assessment']['recommendations'].append("Model consistently underestimates - consider bias correction")

        if bias_severity == 'High':
            bias_analysis['assessment']['recommendations'].append("High systematic bias detected - model recalibration recommended")

        print(f"[BIAS] Analysis completed: {bias_severity} systematic bias detected")

        return bias_analysis

class ResidualAnalyzer:
    """
    Statistical Residual Analysis Engine
    
    Performs comprehensive statistical tests on model residuals
    to validate model assumptions and detect patterns.
    """
    
    def __init__(self):
        self.test_results = {}
        self.diagnostic_plots = {}
        
    def comprehensive_residual_analysis(self,
                                      residuals: np.ndarray,
                                      fitted_values: np.ndarray = None,
                                      significance_level: float = 0.05) -> Dict:
        """
        Perform comprehensive residual analysis
        
        Args:
            residuals: Model residuals
            fitted_values: Fitted/predicted values
            significance_level: Significance level for tests
            
        Returns:
            Dictionary with all test results
        """
        
        print("[INFO] Performing comprehensive residual analysis...")
        
        residuals = np.array(residuals).flatten()
        residuals_clean = residuals[~np.isnan(residuals)]
        
        if len(residuals_clean) < 10:
            return {'error': 'Insufficient data for residual analysis'}
        
        analysis_results = {}
        
        # 1. Normality Tests
        analysis_results['normality'] = self._test_normality(residuals_clean, significance_level)
        
        # 2. Autocorrelation Tests
        analysis_results['autocorrelation'] = self._test_autocorrelation(residuals_clean, significance_level)
        
        # 3. Heteroskedasticity Tests
        if fitted_values is not None:
            fitted_values_clean = np.array(fitted_values).flatten()
            if len(fitted_values_clean) == len(residuals_clean):
                analysis_results['heteroskedasticity'] = self._test_heteroskedasticity(
                    residuals_clean, fitted_values_clean, significance_level
                )
        
        # 4. Randomness Tests
        analysis_results['randomness'] = self._test_randomness(residuals_clean, significance_level)
        
        # 5. Distributional Properties
        analysis_results['distribution'] = self._analyze_distribution(residuals_clean)
        
        # 6. Outlier Detection
        analysis_results['outliers'] = self._detect_outliers(residuals_clean)
        
        # Overall assessment
        analysis_results['overall_assessment'] = self._assess_residuals(analysis_results)
        
        self.test_results = analysis_results
        
        print("[OK] Residual analysis completed")
        
        return analysis_results
    
    def _test_normality(self, residuals: np.ndarray, alpha: float = 0.05) -> Dict:
        """Test normality of residuals"""
        
        results = {}
        
        try:
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(residuals)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > alpha,
                'interpretation': 'Normal' if jb_p > alpha else 'Non-normal'
            }
        except:
            results['jarque_bera'] = {'error': 'Test failed'}
        
        try:
            # D'Agostino-Pearson test
            dp_stat, dp_p = normaltest(residuals)
            results['dagostino_pearson'] = {
                'statistic': dp_stat,
                'p_value': dp_p,
                'is_normal': dp_p > alpha,
                'interpretation': 'Normal' if dp_p > alpha else 'Non-normal'
            }
        except:
            results['dagostino_pearson'] = {'error': 'Test failed'}
        
        try:
            # Shapiro-Wilk test (if sample size allows)
            if len(residuals) <= 5000:  # Shapiro-Wilk limitation
                sw_stat, sw_p = stats.shapiro(residuals)
                results['shapiro_wilk'] = {
                    'statistic': sw_stat,
                    'p_value': sw_p,
                    'is_normal': sw_p > alpha,
                    'interpretation': 'Normal' if sw_p > alpha else 'Non-normal'
                }
        except:
            pass
        
        return results
    
    def _test_autocorrelation(self, residuals: np.ndarray, alpha: float = 0.05) -> Dict:
        """Test autocorrelation in residuals"""
        
        results = {}
        
        try:
            # Ljung-Box test for multiple lags
            max_lags = min(10, len(residuals) // 4)
            if max_lags >= 1:
                lb_result = acorr_ljungbox(residuals, lags=max_lags, return_df=True)
                
                results['ljung_box'] = {
                    'statistics': lb_result['lb_stat'].to_dict(),
                    'p_values': lb_result['lb_pvalue'].to_dict(),
                    'significant_lags': lb_result[lb_result['lb_pvalue'] < alpha].index.tolist(),
                    'has_autocorrelation': any(lb_result['lb_pvalue'] < alpha),
                    'interpretation': 'Autocorrelated' if any(lb_result['lb_pvalue'] < alpha) else 'Independent'
                }
        except:
            results['ljung_box'] = {'error': 'Test failed'}
        
        try:
            # ACF and PACF analysis
            max_lags = min(20, len(residuals) // 3)
            if max_lags >= 1:
                acf_values = acf(residuals, nlags=max_lags, fft=False)
                pacf_values = pacf(residuals, nlags=max_lags)
                
                results['acf_analysis'] = {
                    'acf_values': acf_values.tolist(),
                    'pacf_values': pacf_values.tolist(),
                    'significant_acf_lags': [i for i, val in enumerate(acf_values[1:], 1) if abs(val) > 1.96/np.sqrt(len(residuals))],
                    'significant_pacf_lags': [i for i, val in enumerate(pacf_values[1:], 1) if abs(val) > 1.96/np.sqrt(len(residuals))]
                }
        except:
            results['acf_analysis'] = {'error': 'Analysis failed'}
        
        return results
    
    def _test_heteroskedasticity(self, residuals: np.ndarray, fitted_values: np.ndarray, alpha: float = 0.05) -> Dict:
        """Test heteroskedasticity in residuals"""
        
        results = {}
        
        try:
            # ARCH test
            arch_stat, arch_p, _, _ = het_arch(residuals)
            results['arch_test'] = {
                'statistic': arch_stat,
                'p_value': arch_p,
                'is_homoskedastic': arch_p > alpha,
                'interpretation': 'Homoskedastic' if arch_p > alpha else 'Heteroskedastic'
            }
        except:
            results['arch_test'] = {'error': 'Test failed'}
        
        try:
            # Breusch-Pagan test (manual implementation)
            squared_residuals = residuals ** 2
            correlation = np.corrcoef(fitted_values, squared_residuals)[0, 1]
            
            # Simple correlation test
            n = len(residuals)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation ** 2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            results['breusch_pagan'] = {
                'correlation': correlation,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_homoskedastic': p_value > alpha,
                'interpretation': 'Homoskedastic' if p_value > alpha else 'Heteroskedastic'
            }
        except:
            results['breusch_pagan'] = {'error': 'Test failed'}
        
        return results
    
    def _test_randomness(self, residuals: np.ndarray, alpha: float = 0.05) -> Dict:
        """Test randomness of residuals"""
        
        results = {}
        
        try:
            # Runs test
            # Convert to binary (above/below median)
            median_val = np.median(residuals)
            binary_series = (residuals > median_val).astype(int)
            
            z_stat, p_value = runstest_1samp(binary_series)
            results['runs_test'] = {
                'z_statistic': z_stat,
                'p_value': p_value,
                'is_random': p_value > alpha,
                'interpretation': 'Random' if p_value > alpha else 'Non-random pattern'
            }
        except:
            results['runs_test'] = {'error': 'Test failed'}
        
        return results
    
    def _analyze_distribution(self, residuals: np.ndarray) -> Dict:
        """Analyze distributional properties"""
        
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'q50': np.percentile(residuals, 50),
            'q75': np.percentile(residuals, 75),
            'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25)
        }
    
    def _detect_outliers(self, residuals: np.ndarray) -> Dict:
        """Detect outliers in residuals"""
        
        # IQR method
        q25, q75 = np.percentile(residuals, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        iqr_outliers = (residuals < lower_bound) | (residuals > upper_bound)
        
        # Z-score method
        z_scores = np.abs(stats.zscore(residuals))
        z_outliers = z_scores > 3
        
        return {
            'iqr_method': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': np.sum(iqr_outliers),
                'outlier_percentage': np.sum(iqr_outliers) / len(residuals) * 100,
                'outlier_indices': np.where(iqr_outliers)[0].tolist()
            },
            'zscore_method': {
                'n_outliers': np.sum(z_outliers),
                'outlier_percentage': np.sum(z_outliers) / len(residuals) * 100,
                'outlier_indices': np.where(z_outliers)[0].tolist(),
                'max_zscore': np.max(z_scores)
            }
        }
    
    def _assess_residuals(self, analysis_results: Dict) -> Dict:
        """Provide overall assessment of residuals"""
        
        issues = []
        quality_score = 100
        
        # Check normality
        if 'normality' in analysis_results:
            normality_tests = analysis_results['normality']
            normal_count = sum(1 for test in normality_tests.values() 
                             if isinstance(test, dict) and test.get('is_normal', False))
            if normal_count == 0:
                issues.append("Residuals are not normally distributed")
                quality_score -= 20
        
        # Check autocorrelation
        if 'autocorrelation' in analysis_results:
            autocorr_tests = analysis_results['autocorrelation']
            if autocorr_tests.get('ljung_box', {}).get('has_autocorrelation', False):
                issues.append("Residuals show autocorrelation")
                quality_score -= 25
        
        # Check heteroskedasticity
        if 'heteroskedasticity' in analysis_results:
            hetero_tests = analysis_results['heteroskedasticity']
            hetero_count = sum(1 for test in hetero_tests.values() 
                             if isinstance(test, dict) and not test.get('is_homoskedastic', True))
            if hetero_count > 0:
                issues.append("Residuals show heteroskedasticity")
                quality_score -= 20
        
        # Check randomness
        if 'randomness' in analysis_results:
            random_tests = analysis_results['randomness']
            if not random_tests.get('runs_test', {}).get('is_random', True):
                issues.append("Residuals show non-random patterns")
                quality_score -= 15
        
        # Check outliers
        if 'outliers' in analysis_results:
            outlier_pct = analysis_results['outliers']['iqr_method']['outlier_percentage']
            if outlier_pct > 5:
                issues.append(f"High percentage of outliers ({outlier_pct:.1f}%)")
                quality_score -= 10
        
        quality_score = max(0, quality_score)
        
        if quality_score >= 90:
            assessment = "Excellent"
        elif quality_score >= 70:
            assessment = "Good"
        elif quality_score >= 50:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        return {
            'quality_score': quality_score,
            'assessment': assessment,
            'issues_detected': issues,
            'recommendations': self._get_recommendations(issues)
        }
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations based on detected issues"""
        
        recommendations = []
        
        for issue in issues:
            if "not normally distributed" in issue:
                recommendations.append("Consider robust regression methods or data transformations")
            elif "autocorrelation" in issue:
                recommendations.append("Add temporal features or use time series models")
            elif "heteroskedasticity" in issue:
                recommendations.append("Use weighted regression or transform target variable")
            elif "non-random patterns" in issue:
                recommendations.append("Investigate missing features or model specification")
            elif "outliers" in issue:
                recommendations.append("Investigate and potentially remove or transform outliers")
        
        return recommendations

class ErrorVisualizationEngine:
    """
    Advanced Error Visualization Engine
    
    Creates comprehensive diagnostic plots for error analysis
    """
    
    def __init__(self):
        self.figures = {}
        
    def create_comprehensive_diagnostics(self,
                                       residuals: np.ndarray,
                                       fitted_values: np.ndarray = None,
                                       actual_values: np.ndarray = None,
                                       save_dir: str = None) -> Dict[str, str]:
        """Create comprehensive diagnostic plots"""
        
        print("[INFO] Creating comprehensive diagnostic plots...")
        
        residuals = np.array(residuals).flatten()
        residuals_clean = residuals[~np.isnan(residuals)]
        
        if len(residuals_clean) < 10:
            return {}
        
        # Set up plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        saved_plots = {}
        
        # 1. Residual Distribution Plot
        saved_plots['residual_distribution'] = self._plot_residual_distribution(
            residuals_clean, save_dir
        )
        
        # 2. Q-Q Plot
        saved_plots['qq_plot'] = self._plot_qq(residuals_clean, save_dir)
        
        # 3. Residual vs Fitted Plot (if fitted values available)
        if fitted_values is not None:
            fitted_clean = np.array(fitted_values).flatten()
            if len(fitted_clean) == len(residuals_clean):
                saved_plots['residual_vs_fitted'] = self._plot_residual_vs_fitted(
                    residuals_clean, fitted_clean, save_dir
                )
        
        # 4. ACF/PACF Plots
        saved_plots['acf_pacf'] = self._plot_acf_pacf(residuals_clean, save_dir)
        
        # 5. Time Series Plot (if actual values available)
        if actual_values is not None:
            actual_clean = np.array(actual_values).flatten()[:len(residuals_clean)]
            saved_plots['time_series'] = self._plot_time_series(
                actual_clean, residuals_clean, save_dir
            )
        
        return saved_plots
    
    def _plot_residual_distribution(self, residuals: np.ndarray, save_dir: str = None) -> str:
        """Plot residual distribution with normality assessment"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Residual Distribution Analysis', fontsize=16)
        
        # Histogram with normal overlay
        axes[0, 0].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        axes[0, 0].set_title('Histogram vs Normal Distribution')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(residuals, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue'))
        axes[0, 1].set_title('Residual Box Plot')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Kernel density estimate
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        axes[1, 0].plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        axes[1, 0].plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r--', linewidth=2, label='Normal')
        axes[1, 0].set_title('Kernel Density vs Normal')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Mean: {np.mean(residuals):.4f}
        Std: {np.std(residuals):.4f}
        Skewness: {stats.skew(residuals):.4f}
        Kurtosis: {stats.kurtosis(residuals):.4f}
        Min: {np.min(residuals):.4f}
        Max: {np.max(residuals):.4f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'residual_distribution.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return "displayed"
    
    def _plot_qq(self, residuals: np.ndarray, save_dir: str = None) -> str:
        """Create Q-Q plot for normality assessment"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add R² for the fit
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        observed_quantiles = np.sort(residuals)
        r_squared = np.corrcoef(theoretical_quantiles, observed_quantiles)[0, 1] ** 2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        if save_dir:
            save_path = Path(save_dir) / 'qq_plot.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return "displayed"
    
    def _plot_residual_vs_fitted(self, residuals: np.ndarray, fitted: np.ndarray, save_dir: str = None) -> str:
        """Create residual vs fitted values plot"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residual vs Fitted
        axes[0].scatter(fitted, residuals, alpha=0.6, s=20)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted Values')
        axes[0].grid(True, alpha=0.3)
        
        # Add LOWESS smooth line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted, frac=0.3)
            axes[0].plot(smoothed[:, 0], smoothed[:, 1], 'blue', linewidth=2, label='LOWESS')
            axes[0].legend()
        except:
            pass
        
        # Scale-Location plot (sqrt of absolute residuals vs fitted)
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1].scatter(fitted, sqrt_abs_residuals, alpha=0.6, s=20)
        axes[1].set_xlabel('Fitted Values')
        axes[1].set_ylabel('√|Residuals|')
        axes[1].set_title('Scale-Location Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Add LOWESS smooth line
        try:
            smoothed = lowess(sqrt_abs_residuals, fitted, frac=0.3)
            axes[1].plot(smoothed[:, 0], smoothed[:, 1], 'blue', linewidth=2, label='LOWESS')
            axes[1].legend()
        except:
            pass
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'residual_vs_fitted.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return "displayed"
    
    def _plot_acf_pacf(self, residuals: np.ndarray, save_dir: str = None) -> str:
        """Create ACF and PACF plots"""
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        max_lags = min(30, len(residuals) // 4)
        
        # ACF Plot
        plot_acf(residuals, lags=max_lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].grid(True, alpha=0.3)
        
        # PACF Plot
        plot_pacf(residuals, lags=max_lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'acf_pacf_plots.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return "displayed"
    
    def _plot_time_series(self, actual: np.ndarray, residuals: np.ndarray, save_dir: str = None) -> str:
        """Create time series plots of actual vs residuals"""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        time_index = np.arange(len(actual))
        
        # Actual values
        axes[0].plot(time_index, actual, 'b-', linewidth=1, alpha=0.7)
        axes[0].set_title('Actual Values Over Time')
        axes[0].set_ylabel('Actual')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1].plot(time_index, residuals, 'r-', linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_title('Residuals Over Time')
        axes[1].set_ylabel('Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # Rolling statistics of residuals
        window_size = max(10, len(residuals) // 20)
        if window_size < len(residuals):
            rolling_mean = pd.Series(residuals).rolling(window=window_size).mean()
            rolling_std = pd.Series(residuals).rolling(window=window_size).std()
            
            axes[2].plot(time_index, rolling_mean, 'g-', linewidth=2, label=f'Rolling Mean ({window_size})')
            axes[2].plot(time_index, rolling_std, 'm-', linewidth=2, label=f'Rolling Std ({window_size})')
            axes[2].set_title('Rolling Statistics of Residuals')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'time_series_plots.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return "displayed"

    def plot_seasonal_residuals(self,
                               residuals: np.ndarray,
                               dates: pd.DatetimeIndex = None,
                               periods: List[int] = None,
                               save_dir: str = None) -> str:
        """
        Create seasonal plots of residuals - PHASE 5 REQUIREMENT

        Args:
            residuals: Residual values
            dates: Datetime index for seasonal decomposition
            periods: List of seasonal periods to analyze (e.g., [7, 30, 365])
            save_dir: Directory to save plots

        Returns:
            Path to saved plot or "displayed"
        """

        if periods is None:
            periods = [7, 30]  # Weekly and monthly seasonality

        if dates is None:
            dates = pd.date_range(start='2023-01-01', periods=len(residuals), freq='D')
        elif len(dates) != len(residuals):
            dates = pd.date_range(start='2023-01-01', periods=len(residuals), freq='D')

        residuals_series = pd.Series(residuals, index=dates)

        # Number of subplots based on periods
        n_periods = len(periods)
        fig, axes = plt.subplots(n_periods + 1, 2, figsize=(15, 6 * (n_periods + 1)))

        if n_periods == 1:
            axes = axes.reshape(2, 2)

        # Overall residuals over time
        axes[0, 0].plot(residuals_series.index, residuals_series.values, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Monthly aggregated residuals
        monthly_residuals = residuals_series.groupby(residuals_series.index.to_period('M')).mean()
        axes[0, 1].bar(range(len(monthly_residuals)), monthly_residuals.values, alpha=0.7)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_title('Average Monthly Residuals')
        axes[0, 1].set_ylabel('Average Residuals')
        axes[0, 1].set_xticks(range(len(monthly_residuals)))
        axes[0, 1].set_xticklabels([str(p) for p in monthly_residuals.index], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Seasonal decomposition for each period
        for i, period in enumerate(periods):
            row_idx = i + 1

            try:
                if len(residuals_series) >= 2 * period:
                    # Seasonal pattern by period
                    if period == 7:  # Weekly
                        seasonal_groups = residuals_series.groupby(residuals_series.index.dayofweek)
                        seasonal_data = seasonal_groups.mean()
                        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        title_suffix = '(Day of Week)'
                    elif period == 30:  # Monthly
                        seasonal_groups = residuals_series.groupby(residuals_series.index.day)
                        seasonal_data = seasonal_groups.mean()
                        labels = [str(i) for i in seasonal_data.index]
                        title_suffix = '(Day of Month)'
                    else:  # Generic period
                        seasonal_groups = residuals_series.groupby(residuals_series.index.dayofyear % period)
                        seasonal_data = seasonal_groups.mean()
                        labels = [str(i) for i in seasonal_data.index]
                        title_suffix = f'(Period {period})'

                    # Bar plot of seasonal patterns
                    axes[row_idx, 0].bar(range(len(seasonal_data)), seasonal_data.values, alpha=0.7)
                    axes[row_idx, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
                    axes[row_idx, 0].set_title(f'Average Residuals by Season {title_suffix}')
                    axes[row_idx, 0].set_ylabel('Average Residuals')
                    axes[row_idx, 0].set_xticks(range(len(seasonal_data)))
                    axes[row_idx, 0].set_xticklabels(labels[:len(seasonal_data)], rotation=45)
                    axes[row_idx, 0].grid(True, alpha=0.3)

                    # Box plot of seasonal residual distributions
                    seasonal_boxplot_data = []
                    seasonal_boxplot_labels = []

                    for group_key in seasonal_groups.groups.keys():
                        group_data = seasonal_groups.get_group(group_key)
                        if len(group_data) > 0:
                            seasonal_boxplot_data.append(group_data.values)
                            if period == 7:
                                seasonal_boxplot_labels.append(labels[group_key])
                            else:
                                seasonal_boxplot_labels.append(str(group_key))

                    if seasonal_boxplot_data:
                        box_plot = axes[row_idx, 1].boxplot(seasonal_boxplot_data, labels=seasonal_boxplot_labels)
                        axes[row_idx, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
                        axes[row_idx, 1].set_title(f'Residual Distribution by Season {title_suffix}')
                        axes[row_idx, 1].set_ylabel('Residuals')
                        axes[row_idx, 1].tick_params(axis='x', rotation=45)
                        axes[row_idx, 1].grid(True, alpha=0.3)
                else:
                    # Not enough data for this period
                    axes[row_idx, 0].text(0.5, 0.5, f'Insufficient data\nfor period {period}',
                                         ha='center', va='center', transform=axes[row_idx, 0].transAxes)
                    axes[row_idx, 1].text(0.5, 0.5, f'Insufficient data\nfor period {period}',
                                         ha='center', va='center', transform=axes[row_idx, 1].transAxes)
                    axes[row_idx, 0].set_title(f'Seasonal Analysis (Period {period})')
                    axes[row_idx, 1].set_title(f'Seasonal Analysis (Period {period})')

            except Exception as e:
                print(f"[WARNING] Seasonal analysis failed for period {period}: {e}")
                axes[row_idx, 0].text(0.5, 0.5, f'Analysis failed\nfor period {period}',
                                     ha='center', va='center', transform=axes[row_idx, 0].transAxes)
                axes[row_idx, 1].text(0.5, 0.5, f'Analysis failed\nfor period {period}',
                                     ha='center', va='center', transform=axes[row_idx, 1].transAxes)

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'seasonal_residual_plots.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return "displayed"

def main():
    """Demonstration of Error Analysis Engine"""
    
    print("🔍 ERROR ANALYSIS ENGINE - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load sample data
        print("Loading data for error analysis demonstration...")
        trans_df, prod_df, pdv_df = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=15000,
            sample_products=300,
            enable_joins=True,
            validate_loss=True
        )
        
        print(f"Data loaded: {trans_df.shape}")
        
        # Create synthetic predictions for demonstration
        np.random.seed(42)
        actual_values = trans_df['quantity'].values
        
        # Create realistic predictions with some systematic errors
        predicted_values = actual_values * (1 + np.random.normal(0, 0.2, len(actual_values)))
        
        # Add systematic bias for some products
        if 'internal_product_id' in trans_df.columns:
            product_ids = trans_df['internal_product_id'].unique()[:50]
            for pid in product_ids:
                mask = trans_df['internal_product_id'] == pid
                predicted_values[mask] *= 1.1  # 10% systematic overestimation
        
        # Create analysis DataFrame
        analysis_df = trans_df.copy()
        analysis_df['actual'] = actual_values
        analysis_df['predicted'] = predicted_values
        
        print(f"Analysis data prepared: {analysis_df.shape}")
        
        # 1. Error Decomposition Analysis
        print("\n[DEMO] Performing error decomposition analysis...")
        
        error_decomposer = ErrorDecomposer()
        
        decomposition_results = error_decomposer.decompose_errors(
            analysis_df,
            actual_col='actual',
            predicted_col='predicted',
            dimensions=['internal_product_id', 'internal_store_id']
        )
        
        # 2. Temporal Pattern Analysis
        print("\n[DEMO] Analyzing temporal error patterns...")
        
        temporal_results = error_decomposer.analyze_temporal_patterns(
            analysis_df,
            actual_col='actual',
            predicted_col='predicted',
            date_col='transaction_date'
        )
        
        # 3. Residual Analysis
        print("\n[DEMO] Performing comprehensive residual analysis...")
        
        residual_analyzer = ResidualAnalyzer()
        residuals = actual_values - predicted_values
        
        residual_results = residual_analyzer.comprehensive_residual_analysis(
            residuals,
            fitted_values=predicted_values
        )
        
        # 4. Visualization
        print("\n[DEMO] Creating diagnostic visualizations...")
        
        viz_engine = ErrorVisualizationEngine()
        viz_results = viz_engine.create_comprehensive_diagnostics(
            residuals,
            fitted_values=predicted_values,
            actual_values=actual_values,
            save_dir="../../models/diagnostics"
        )
        
        print("\n" + "=" * 80)
        print("🎉 ERROR ANALYSIS ENGINE DEMONSTRATION COMPLETED!")
        print("=" * 80)
        
        # Print summary results
        print("Error Decomposition Summary:")
        for dimension, results in decomposition_results.items():
            if dimension != 'cross_analysis':
                stats = results['dimension_stats']
                print(f"  {dimension}:")
                print(f"    Total segments: {stats['total_segments']}")
                print(f"    Avg WMAPE: {stats['avg_wmape']:.4f}")
                print(f"    Problematic segments: {stats['problematic_count']}")
        
        print(f"\nResidual Analysis Summary:")
        if 'overall_assessment' in residual_results:
            assessment = residual_results['overall_assessment']
            print(f"  Quality Score: {assessment['quality_score']}/100")
            print(f"  Assessment: {assessment['assessment']}")
            if assessment['issues_detected']:
                print(f"  Issues: {', '.join(assessment['issues_detected'])}")
        
        print(f"\nVisualization Files Created: {len(viz_results)}")
        for plot_name, path in viz_results.items():
            print(f"  {plot_name}: {path}")
        
        return error_decomposer, residual_analyzer, viz_engine
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    results = main()