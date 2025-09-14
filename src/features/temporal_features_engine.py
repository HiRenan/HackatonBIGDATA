#!/usr/bin/env python3
"""
TEMPORAL FEATURES ENGINE - Hackathon Forecast Big Data 2025
Advanced Temporal Feature Engineering for Time Series Forecasting

Features baseadas nos insights da EDA:
- Sunday seasonality (76x maior que outros dias)
- September seasonality (mês de pico)
- Growing trend detected
- Autocorrelação significativa detectada

Otimizado para WMAPE: Volume-weighted features para diferentes tiers ABC
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
from scipy import signal
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

warnings.filterwarnings('ignore')

class TemporalFeaturesEngine:
    """
    Advanced Temporal Features Engine
    
    Capabilities:
    - Strategic lag features based on EDA insights
    - Rolling statistics with optimized windows
    - Exponential smoothing features  
    - Fourier decomposition for seasonality
    - Trend analysis and momentum features
    - WMAPE-optimized volume-weighted features
    """
    
    def __init__(self, date_col: str = 'transaction_date', value_col: str = 'quantity'):
        self.date_col = date_col
        self.value_col = value_col
        self.features_created = []
        self.feature_metadata = {}
        
        # Strategic lag periods based on EDA insights
        self.strategic_lags = [1, 2, 3, 4, 7, 8, 12, 26, 52]  # weeks
        
        # Rolling window sizes (optimized for different patterns)
        self.rolling_windows = [4, 8, 12, 26, 52]  # weeks
        
        # Exponential smoothing alphas
        self.ema_alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Seasonal periods detected in EDA
        self.seasonal_periods = {
            'weekly': 7,      # Strong Sunday pattern
            'monthly': 30,    # Monthly patterns
            'quarterly': 91,  # Quarterly cycles
            'yearly': 365     # Annual cycles
        }
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic time-based features"""
        
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        print("[INFO] Creating basic time features...")
        
        # Basic time components
        df['year'] = df[self.date_col].dt.year
        df['month'] = df[self.date_col].dt.month
        df['quarter'] = df[self.date_col].dt.quarter
        df['day_of_year'] = df[self.date_col].dt.dayofyear
        df['day_of_month'] = df[self.date_col].dt.day
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['week_of_year'] = df[self.date_col].dt.isocalendar().week
        
        # Special patterns detected in EDA
        df['is_sunday'] = (df['day_of_week'] == 6).astype(int)  # Sunday effect
        df['is_september'] = (df['month'] == 9).astype(int)     # September peak
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
        df['is_quarter_start'] = df['month'].isin([1, 4, 7, 10]).astype(int)
        
        # Week of month
        df['week_of_month'] = ((df['day_of_month'] - 1) // 7) + 1
        df['week_of_month'] = df['week_of_month'].clip(upper=4)
        
        # Days since epoch (for trend analysis)
        epoch = pd.Timestamp('2022-01-01')
        df['days_since_epoch'] = (df[self.date_col] - epoch).dt.days
        
        # Cyclical encoding for important temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        basic_features = ['year', 'month', 'quarter', 'day_of_year', 'day_of_month', 
                         'day_of_week', 'week_of_year', 'is_sunday', 'is_september',
                         'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start',
                         'week_of_month', 'days_since_epoch', 'month_sin', 'month_cos',
                         'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos']
        
        self.features_created.extend(basic_features)
        print(f"[OK] Created {len(basic_features)} basic time features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create strategic lag features based on EDA insights"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print(f"[INFO] Creating lag features with periods {self.strategic_lags}...")
        
        # Sort by group and date
        df = df.sort_values(groupby_cols + [self.date_col])
        
        lag_features = []
        
        for lag in self.strategic_lags:
            lag_col = f'{self.value_col}_lag_{lag}'
            
            # Create lag feature
            df[lag_col] = df.groupby(groupby_cols)[self.value_col].shift(lag)
            
            # Lag difference (momentum)
            df[f'{lag_col}_diff'] = df[self.value_col] - df[lag_col]
            df[f'{lag_col}_pct_change'] = df[self.value_col] / (df[lag_col] + 1e-8) - 1
            
            # Lag ratio
            df[f'{lag_col}_ratio'] = df[self.value_col] / (df[lag_col] + 1e-8)
            
            lag_features.extend([lag_col, f'{lag_col}_diff', f'{lag_col}_pct_change', f'{lag_col}_ratio'])
        
        # Fill NaN values with group-specific medians
        for feature in lag_features:
            if feature in df.columns:
                df[feature] = df.groupby(groupby_cols)[feature].transform(
                    lambda x: x.fillna(x.median())
                )
        
        self.features_created.extend(lag_features)
        print(f"[OK] Created {len(lag_features)} lag features")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                              groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create rolling statistics features"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print(f"[INFO] Creating rolling features with windows {self.rolling_windows}...")
        
        # Sort by group and date
        df = df.sort_values(groupby_cols + [self.date_col])
        
        rolling_features = []
        
        for window in self.rolling_windows:
            # Basic rolling statistics
            for stat in ['mean', 'median', 'std', 'min', 'max']:
                feature_name = f'{self.value_col}_rolling_{window}_{stat}'
                df[feature_name] = df.groupby(groupby_cols)[self.value_col].rolling(
                    window, min_periods=1
                ).agg(stat).reset_index(level=groupby_cols, drop=True)
                rolling_features.append(feature_name)
            
            # Advanced rolling statistics
            # Coefficient of variation (volatility)
            mean_col = f'{self.value_col}_rolling_{window}_mean'
            std_col = f'{self.value_col}_rolling_{window}_std'
            cv_col = f'{self.value_col}_rolling_{window}_cv'
            df[cv_col] = df[std_col] / (df[mean_col] + 1e-8)
            rolling_features.append(cv_col)
            
            # Skewness (approximation)
            skew_col = f'{self.value_col}_rolling_{window}_skew'
            df[skew_col] = df.groupby(groupby_cols)[self.value_col].rolling(
                window, min_periods=3
            ).skew().reset_index(level=groupby_cols, drop=True)
            rolling_features.append(skew_col)
            
            # Rolling trend (slope)
            trend_col = f'{self.value_col}_rolling_{window}_trend'
            df[trend_col] = df.groupby(groupby_cols)[self.value_col].rolling(
                window, min_periods=2
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0).reset_index(level=groupby_cols, drop=True)
            rolling_features.append(trend_col)
            
            # Rolling autocorrelation (lag-1)
            autocorr_col = f'{self.value_col}_rolling_{window}_autocorr'
            df[autocorr_col] = df.groupby(groupby_cols)[self.value_col].rolling(
                window, min_periods=2
            ).apply(lambda x: x.autocorr(lag=1) if len(x) >= 3 else 0).reset_index(level=groupby_cols, drop=True)
            rolling_features.append(autocorr_col)
        
        # Fill NaN values
        for feature in rolling_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)
        
        self.features_created.extend(rolling_features)
        print(f"[OK] Created {len(rolling_features)} rolling features")
        
        return df
    
    def create_exponential_features(self, df: pd.DataFrame,
                                   groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create exponential moving average features"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print(f"[INFO] Creating exponential features with alphas {self.ema_alphas}...")
        
        # Sort by group and date
        df = df.sort_values(groupby_cols + [self.date_col])
        
        ema_features = []
        
        for alpha in self.ema_alphas:
            # Exponential moving average
            ema_col = f'{self.value_col}_ema_{int(alpha*10)}'
            df[ema_col] = df.groupby(groupby_cols)[self.value_col].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
            ema_features.append(ema_col)
            
            # EMA momentum (current value vs EMA)
            ema_momentum_col = f'{self.value_col}_ema_{int(alpha*10)}_momentum'
            df[ema_momentum_col] = df[self.value_col] - df[ema_col]
            ema_features.append(ema_momentum_col)
            
            # EMA ratio
            ema_ratio_col = f'{self.value_col}_ema_{int(alpha*10)}_ratio'
            df[ema_ratio_col] = df[self.value_col] / (df[ema_col] + 1e-8)
            ema_features.append(ema_ratio_col)
        
        # Double exponential smoothing (Holt's method approximation)
        for alpha in [0.3, 0.5, 0.7]:
            level_col = f'{self.value_col}_holt_level_{int(alpha*10)}'
            trend_col = f'{self.value_col}_holt_trend_{int(alpha*10)}'
            
            # Simple implementation of Holt's method
            df[level_col] = df.groupby(groupby_cols)[self.value_col].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
            
            # Trend approximation using EMA of differences
            df[trend_col] = df.groupby(groupby_cols)[self.value_col].transform(
                lambda x: x.diff().ewm(alpha=alpha/2, adjust=False).mean()
            ).fillna(0)
            
            ema_features.extend([level_col, trend_col])
        
        self.features_created.extend(ema_features)
        print(f"[OK] Created {len(ema_features)} exponential features")
        
        return df
    
    def create_fourier_features(self, df: pd.DataFrame,
                               groupby_cols: List[str] = None,
                               max_harmonics: int = 3) -> pd.DataFrame:
        """Create Fourier decomposition features for seasonality"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print(f"[INFO] Creating Fourier features for seasonal decomposition...")
        
        fourier_features = []
        
        # Create Fourier features for each seasonal period
        for period_name, period in self.seasonal_periods.items():
            for harmonic in range(1, max_harmonics + 1):
                # Sine component
                sin_col = f'fourier_{period_name}_sin_{harmonic}'
                df[sin_col] = np.sin(2 * np.pi * harmonic * df['days_since_epoch'] / period)
                fourier_features.append(sin_col)
                
                # Cosine component  
                cos_col = f'fourier_{period_name}_cos_{harmonic}'
                df[cos_col] = np.cos(2 * np.pi * harmonic * df['days_since_epoch'] / period)
                fourier_features.append(cos_col)
        
        self.features_created.extend(fourier_features)
        print(f"[OK] Created {len(fourier_features)} Fourier features")
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame,
                             groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create trend analysis features"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print("[INFO] Creating trend analysis features...")
        
        # Sort by group and date
        df = df.sort_values(groupby_cols + [self.date_col])
        
        trend_features = []
        
        # Linear trend over different windows
        for window in [4, 8, 12, 26]:
            trend_col = f'{self.value_col}_trend_{window}'
            
            # Linear regression slope
            df[trend_col] = df.groupby(groupby_cols)[self.value_col].rolling(
                window, min_periods=3
            ).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
            ).reset_index(level=groupby_cols, drop=True)
            
            trend_features.append(trend_col)
            
            # Trend acceleration (second derivative approximation)
            accel_col = f'{self.value_col}_accel_{window}'
            df[accel_col] = df.groupby(groupby_cols)[trend_col].diff().fillna(0)
            trend_features.append(accel_col)
        
        # Momentum features
        for period in [1, 7, 30]:
            momentum_col = f'{self.value_col}_momentum_{period}'
            df[momentum_col] = df.groupby(groupby_cols)[self.value_col].pct_change(period).fillna(0)
            trend_features.append(momentum_col)
        
        # Volatility features
        for window in [7, 14, 30]:
            vol_col = f'{self.value_col}_volatility_{window}'
            df[vol_col] = df.groupby(groupby_cols)[self.value_col].rolling(
                window, min_periods=2
            ).std().fillna(0).reset_index(level=groupby_cols, drop=True)
            trend_features.append(vol_col)
        
        # Stability indicator (inverse of coefficient of variation)
        stability_col = f'{self.value_col}_stability'
        df[stability_col] = df.groupby(groupby_cols)[self.value_col].transform(
            lambda x: x.mean() / (x.std() + 1e-8) if len(x) > 1 else 1
        )
        trend_features.append(stability_col)
        
        self.features_created.extend(trend_features)
        print(f"[OK] Created {len(trend_features)} trend features")
        
        return df
    
    def create_wmape_optimized_features(self, df: pd.DataFrame,
                                       groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create features optimized specifically for WMAPE metric"""
        
        df = df.copy()
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
            
        print("[INFO] Creating WMAPE-optimized features...")
        
        wmape_features = []
        
        # Volume-weighted features (critical for WMAPE)
        total_volume = df.groupby(groupby_cols)[self.value_col].sum()
        df = df.merge(
            total_volume.rename('total_volume_product_store').reset_index(),
            on=groupby_cols,
            how='left'
        )
        wmape_features.append('total_volume_product_store')
        
        # Relative performance vs category/region
        for group_col in ['internal_product_id', 'internal_store_id']:
            if group_col in df.columns:
                relative_col = f'{self.value_col}_relative_to_{group_col.split("_")[1]}'
                group_mean = df.groupby(group_col)[self.value_col].transform('mean')
                df[relative_col] = df[self.value_col] / (group_mean + 1e-8)
                wmape_features.append(relative_col)
        
        # Error difficulty indicators
        # High volume = lower percentage error tolerance
        df['volume_weight'] = df['total_volume_product_store'] / df['total_volume_product_store'].max()
        df['forecast_difficulty'] = 1 / (df['volume_weight'] + 0.1)  # High volume = easier to forecast accurately
        wmape_features.extend(['volume_weight', 'forecast_difficulty'])
        
        # Percentage error features (historical simulation)
        for lag in [1, 4, 12]:
            lag_col = f'{self.value_col}_lag_{lag}'
            if lag_col in df.columns:
                pct_error_col = f'historical_pct_error_{lag}'
                df[pct_error_col] = np.abs(df[self.value_col] - df[lag_col]) / (df[self.value_col] + 1e-8)
                wmape_features.append(pct_error_col)
        
        # Intermittency indicators (critical for WMAPE optimization)
        df['is_zero'] = (df[self.value_col] == 0).astype(int)
        df['zero_ratio'] = df.groupby(groupby_cols)['is_zero'].transform('mean')
        wmape_features.extend(['is_zero', 'zero_ratio'])
        
        self.features_created.extend(wmape_features)
        print(f"[OK] Created {len(wmape_features)} WMAPE-optimized features")
        
        return df
    
    def create_all_temporal_features(self, df: pd.DataFrame,
                                    groupby_cols: List[str] = None) -> pd.DataFrame:
        """Create all temporal features in optimized sequence"""
        
        print("\n" + "="*80)
        print("TEMPORAL FEATURES ENGINE - CREATING ALL FEATURES")
        print("="*80)
        
        if groupby_cols is None:
            groupby_cols = ['internal_product_id', 'internal_store_id']
        
        initial_shape = df.shape
        print(f"[START] Initial dataset: {initial_shape}")
        
        # Sequence of feature creation (optimized order)
        df = self.create_time_features(df)
        df = self.create_lag_features(df, groupby_cols)
        df = self.create_rolling_features(df, groupby_cols)
        df = self.create_exponential_features(df, groupby_cols)
        df = self.create_fourier_features(df, groupby_cols)
        df = self.create_trend_features(df, groupby_cols)
        df = self.create_wmape_optimized_features(df, groupby_cols)
        
        final_shape = df.shape
        features_added = final_shape[1] - initial_shape[1]
        
        print(f"\n[SUMMARY] Temporal Features Engine completed:")
        print(f"  Features added: {features_added}")
        print(f"  Total features created: {len(self.features_created)}")
        print(f"  Final dataset shape: {final_shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
    
    def get_feature_importance_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights about created features for documentation"""
        
        insights = {
            'total_features': len(self.features_created),
            'feature_categories': {
                'time_features': len([f for f in self.features_created if any(x in f for x in ['year', 'month', 'day', 'week', 'sin', 'cos'])]),
                'lag_features': len([f for f in self.features_created if 'lag' in f]),
                'rolling_features': len([f for f in self.features_created if 'rolling' in f]),
                'ema_features': len([f for f in self.features_created if 'ema' in f or 'holt' in f]),
                'fourier_features': len([f for f in self.features_created if 'fourier' in f]),
                'trend_features': len([f for f in self.features_created if any(x in f for x in ['trend', 'momentum', 'volatility', 'accel'])]),
                'wmape_features': len([f for f in self.features_created if any(x in f for x in ['volume', 'relative', 'difficulty', 'error', 'zero'])])
            },
            'key_insights': [
                f"Strategic lags based on EDA autocorrelation: {self.strategic_lags}",
                f"Optimized rolling windows: {self.rolling_windows}",
                f"Sunday seasonality captured in Fourier components",
                f"September seasonality incorporated",
                f"Volume-weighted features for WMAPE optimization",
                f"Intermittency features for sparse data handling"
            ],
            'features_list': self.features_created
        }
        
        return insights

def main():
    """Demonstration and testing of Temporal Features Engine"""
    
    print("HACKATHON FORECAST BIG DATA 2025")
    print("TEMPORAL FEATURES ENGINE - DEMONSTRATION")
    print("="*80)
    
    # Load sample data for testing
    try:
        from src.utils.data_loader import load_data_efficiently
        
        print("Loading sample data...")
        trans_df, _, _ = load_data_efficiently(
            data_path="../../data/raw",
            sample_transactions=50000,
            sample_products=1000
        )
        
        print(f"Sample data loaded: {trans_df.shape}")
        
        # Initialize engine
        engine = TemporalFeaturesEngine()
        
        # Create all temporal features
        features_df = engine.create_all_temporal_features(trans_df)
        
        # Get insights
        insights = engine.get_feature_importance_insights(features_df)
        
        print("\n" + "="*80)
        print("FEATURE ENGINEERING INSIGHTS")
        print("="*80)
        
        print(f"Total features created: {insights['total_features']}")
        print("\nFeature categories:")
        for category, count in insights['feature_categories'].items():
            print(f"  {category}: {count}")
        
        print("\nKey insights:")
        for insight in insights['key_insights']:
            print(f"  • {insight}")
        
        # Save results
        output_dir = Path("../../data/features")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        features_file = output_dir / "temporal_features_demo.parquet"
        features_df.to_parquet(features_file, index=False)
        
        print(f"\n[SAVED] Sample features saved to: {features_file}")
        
        return features_df, insights
        
    except Exception as e:
        print(f"[ERROR] Demonstration failed: {e}")
        print("This is expected if running outside the project structure")
        return None, None

if __name__ == "__main__":
    results = main()