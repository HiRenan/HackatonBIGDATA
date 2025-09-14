#!/usr/bin/env python3
"""
Phase 6: Temporal Feature Engineering
Advanced time-based feature extraction for forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class BaseTemporalFeatureEngine(ABC):
    """Abstract base class for temporal feature engines"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.date_column = self.config.get('date_column', 'date')
        self.target_column = self.config.get('target_column', 'total_sales')
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseTemporalFeatureEngine':
        """Fit the feature engine on training data"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with temporal features"""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)

class DateTimeFeatures(BaseTemporalFeatureEngine):
    """Basic datetime feature extraction"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.include_cyclical = self.config.get('include_cyclical', True)
        self.include_holidays = self.config.get('include_holidays', True)

    def fit(self, data: pd.DataFrame) -> 'DateTimeFeatures':
        """Fit datetime features (mostly stateless)"""
        logger.info("Fitting datetime features")
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with datetime features"""
        if not self.is_fitted:
            raise ValueError("TemporalFeatures must be fitted before transform")

        logger.info("Generating datetime features")
        df = data.copy()

        if self.date_column not in df.columns:
            logger.warning(f"Date column '{self.date_column}' not found")
            return df

        # Ensure datetime type
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Basic date components
        df['year'] = df[self.date_column].dt.year
        df['month'] = df[self.date_column].dt.month
        df['day'] = df[self.date_column].dt.day
        df['dayofweek'] = df[self.date_column].dt.dayofweek
        df['dayofyear'] = df[self.date_column].dt.dayofyear
        df['quarter'] = df[self.date_column].dt.quarter
        df['weekofyear'] = df[self.date_column].dt.isocalendar().week

        # Boolean features
        df['is_weekend'] = df['dayofweek'].isin([5, 6])
        df['is_month_start'] = df[self.date_column].dt.is_month_start
        df['is_month_end'] = df[self.date_column].dt.is_month_end
        df['is_quarter_start'] = df[self.date_column].dt.is_quarter_start
        df['is_quarter_end'] = df[self.date_column].dt.is_quarter_end
        df['is_year_start'] = df[self.date_column].dt.is_year_start
        df['is_year_end'] = df[self.date_column].dt.is_year_end

        # Cyclical features (sine/cosine encoding)
        if self.include_cyclical:
            # Month cyclical (1-12)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            # Day of week cyclical (0-6)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

            # Day of year cyclical (1-366)
            df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 366)
            df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 366)

        # Holiday features
        if self.include_holidays:
            df = self._add_holiday_features(df)

        logger.info(f"Added datetime features: {df.shape[1] - data.shape[1]} new columns")
        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday features"""
        # Simple holiday detection (can be expanded)
        df['is_new_year'] = (df['month'] == 1) & (df['day'] == 1)
        df['is_christmas'] = (df['month'] == 12) & (df['day'] == 25)

        # Month-based seasonality flags
        df['is_back_to_school'] = df['month'].isin([8, 9])  # August-September
        df['is_holiday_season'] = df['month'].isin([11, 12])  # November-December
        df['is_summer'] = df['month'].isin([6, 7, 8])  # June-August
        df['is_spring'] = df['month'].isin([3, 4, 5])  # March-May

        return df

class LagFeatures(BaseTemporalFeatureEngine):
    """Lag-based feature engineering"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lag_periods = self.config.get('lag_periods', [1, 7, 14, 30])
        self.group_columns = self.config.get('group_columns', ['store_id', 'product_id'])
        self.fill_method = self.config.get('fill_method', 'forward')

    def fit(self, data: pd.DataFrame) -> 'LagFeatures':
        """Fit lag features (stateless for basic lags)"""
        logger.info("Fitting lag features")
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with lag features"""
        if not self.is_fitted:
            raise ValueError("LagFeatures must be fitted before transform")

        logger.info("Generating lag features")
        df = data.copy()

        if self.target_column not in df.columns:
            logger.warning(f"Target column '{self.target_column}' not found")
            return df

        # Sort by group and date
        sort_columns = self.group_columns + [self.date_column]
        df = df.sort_values(sort_columns)

        # Generate lag features
        for lag in self.lag_periods:
            lag_col = f'{self.target_column}_lag_{lag}'

            if self.group_columns:
                df[lag_col] = df.groupby(self.group_columns)[self.target_column].shift(lag)
            else:
                df[lag_col] = df[self.target_column].shift(lag)

        # Fill missing values
        if self.fill_method == 'forward':
            lag_columns = [f'{self.target_column}_lag_{lag}' for lag in self.lag_periods]
            df[lag_columns] = df[lag_columns].fillna(method='ffill')
        elif self.fill_method == 'zero':
            lag_columns = [f'{self.target_column}_lag_{lag}' for lag in self.lag_periods]
            df[lag_columns] = df[lag_columns].fillna(0)

        logger.info(f"Added lag features: {len(self.lag_periods)} lag periods")
        return df

class RollingFeatures(BaseTemporalFeatureEngine):
    """Rolling window feature engineering"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.windows = self.config.get('windows', [7, 14, 30])
        self.group_columns = self.config.get('group_columns', ['store_id', 'product_id'])
        self.statistics = self.config.get('statistics', ['mean', 'std', 'min', 'max'])

    def fit(self, data: pd.DataFrame) -> 'RollingFeatures':
        """Fit rolling features"""
        logger.info("Fitting rolling features")
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with rolling features"""
        if not self.is_fitted:
            raise ValueError("RollingFeatures must be fitted before transform")

        logger.info("Generating rolling features")
        df = data.copy()

        if self.target_column not in df.columns:
            logger.warning(f"Target column '{self.target_column}' not found")
            return df

        # Sort by group and date
        sort_columns = self.group_columns + [self.date_column]
        df = df.sort_values(sort_columns)

        # Generate rolling features
        for window in self.windows:
            for stat in self.statistics:
                col_name = f'{self.target_column}_rolling_{window}d_{stat}'

                if self.group_columns:
                    rolling_values = df.groupby(self.group_columns)[self.target_column].transform(
                        lambda x: getattr(x.rolling(window=window, min_periods=1), stat)()
                    )
                else:
                    rolling_values = getattr(
                        df[self.target_column].rolling(window=window, min_periods=1), stat
                    )()

                df[col_name] = rolling_values

        logger.info(f"Added rolling features: {len(self.windows) * len(self.statistics)} features")
        return df

class SeasonalFeatures(BaseTemporalFeatureEngine):
    """Seasonal decomposition and features"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.seasonal_periods = self.config.get('seasonal_periods', [7, 30, 365])
        self.group_columns = self.config.get('group_columns', ['store_id', 'product_id'])
        self.seasonal_stats = {}

    def fit(self, data: pd.DataFrame) -> 'SeasonalFeatures':
        """Fit seasonal features by computing seasonal averages"""
        logger.info("Fitting seasonal features")

        df = data.copy()
        if self.target_column not in df.columns:
            return self

        # Ensure datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Compute seasonal statistics
        for period in self.seasonal_periods:
            if period == 7:  # Weekly seasonality
                df['seasonal_group'] = df[self.date_column].dt.dayofweek
            elif period == 30:  # Monthly seasonality
                df['seasonal_group'] = df[self.date_column].dt.day
            elif period == 365:  # Yearly seasonality
                df['seasonal_group'] = df[self.date_column].dt.dayofyear

            # Compute group averages
            group_columns = self.group_columns + ['seasonal_group']
            seasonal_avg = df.groupby(group_columns)[self.target_column].mean()

            self.seasonal_stats[period] = seasonal_avg

        self.is_fitted = True
        logger.info(f"Fitted seasonal features for {len(self.seasonal_periods)} periods")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with seasonal features"""
        if not self.is_fitted:
            raise ValueError("SeasonalFeatures must be fitted before transform")

        logger.info("Generating seasonal features")
        df = data.copy()

        if self.target_column not in df.columns:
            return df

        # Ensure datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Apply seasonal features
        for period in self.seasonal_periods:
            if period == 7:
                df['seasonal_group'] = df[self.date_column].dt.dayofweek
                feature_name = f'seasonal_dow_avg'
            elif period == 30:
                df['seasonal_group'] = df[self.date_column].dt.day
                feature_name = f'seasonal_dom_avg'
            elif period == 365:
                df['seasonal_group'] = df[self.date_column].dt.dayofyear
                feature_name = f'seasonal_doy_avg'

            # Map seasonal averages
            group_columns = self.group_columns + ['seasonal_group']
            df[feature_name] = df.set_index(group_columns).index.map(
                self.seasonal_stats[period]
            )

            # Fill missing with overall average
            df[feature_name] = df[feature_name].fillna(
                df[self.target_column].mean()
            )

        # Drop temporary column
        df = df.drop('seasonal_group', axis=1, errors='ignore')

        logger.info(f"Added seasonal features: {len(self.seasonal_periods)} features")
        return df

class TrendFeatures(BaseTemporalFeatureEngine):
    """Trend-based feature engineering"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.trend_windows = self.config.get('trend_windows', [7, 30])
        self.group_columns = self.config.get('group_columns', ['store_id', 'product_id'])

    def fit(self, data: pd.DataFrame) -> 'TrendFeatures':
        """Fit trend features"""
        logger.info("Fitting trend features")
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with trend features"""
        if not self.is_fitted:
            raise ValueError("TrendFeatures must be fitted before transform")

        logger.info("Generating trend features")
        df = data.copy()

        if self.target_column not in df.columns:
            return df

        # Sort by group and date
        sort_columns = self.group_columns + [self.date_column]
        df = df.sort_values(sort_columns)

        # Generate trend features
        for window in self.trend_windows:
            # Rolling mean
            if self.group_columns:
                rolling_mean = df.groupby(self.group_columns)[self.target_column].transform(
                    lambda x: x.rolling(window=window, min_periods=2).mean()
                )
            else:
                rolling_mean = df[self.target_column].rolling(window=window, min_periods=2).mean()

            # Trend (current vs rolling mean)
            df[f'trend_{window}d'] = df[self.target_column] / rolling_mean - 1

            # Momentum (rate of change)
            if self.group_columns:
                momentum = df.groupby(self.group_columns)[self.target_column].pct_change(periods=window)
            else:
                momentum = df[self.target_column].pct_change(periods=window)

            df[f'momentum_{window}d'] = momentum

        # Fill infinite and missing values
        trend_columns = [col for col in df.columns if 'trend_' in col or 'momentum_' in col]
        df[trend_columns] = df[trend_columns].replace([np.inf, -np.inf], np.nan)
        df[trend_columns] = df[trend_columns].fillna(0)

        logger.info(f"Added trend features: {len(self.trend_windows) * 2} features")
        return df

class TemporalFeaturePipeline:
    """Pipeline for combining temporal features"""

    def __init__(self, feature_engines: List[BaseTemporalFeatureEngine]):
        self.feature_engines = feature_engines
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> 'TemporalFeaturePipeline':
        """Fit all feature engines"""
        logger.info("Fitting temporal feature pipeline")

        for engine in self.feature_engines:
            engine.fit(data)

        self.is_fitted = True
        logger.info(f"Fitted {len(self.feature_engines)} feature engines")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data with all feature engines"""
        if not self.is_fitted:
            raise ValueError("TemporalFeaturePipeline must be fitted before transform")

        logger.info("Applying temporal feature pipeline")
        df = data.copy()

        for engine in self.feature_engines:
            df = engine.transform(df)

        logger.info(f"Pipeline complete: {df.shape[1] - data.shape[1]} new features")
        return df

def create_temporal_pipeline(config: Optional[Dict[str, Any]] = None) -> TemporalFeaturePipeline:
    """Create a comprehensive temporal feature pipeline"""
    config = config or {}

    # Create feature engines
    engines = []

    # Basic datetime features
    engines.append(DateTimeFeatures(config.get('datetime', {})))

    # Lag features
    engines.append(LagFeatures(config.get('lag', {})))

    # Rolling features
    engines.append(RollingFeatures(config.get('rolling', {})))

    # Seasonal features
    engines.append(SeasonalFeatures(config.get('seasonal', {})))

    # Trend features
    engines.append(TrendFeatures(config.get('trend', {})))

    return TemporalFeaturePipeline(engines)

if __name__ == "__main__":
    # Demo usage
    print("⏰ Temporal Feature Engineering Demo")
    print("=" * 50)

    # Test configuration
    config = {
        'datetime': {'include_cyclical': True, 'include_holidays': True},
        'lag': {'lag_periods': [1, 7, 14, 30]},
        'rolling': {'windows': [7, 14, 30], 'statistics': ['mean', 'std']},
        'seasonal': {'seasonal_periods': [7, 30]},
        'trend': {'trend_windows': [7, 30]}
    }

    # Create pipeline
    pipeline = create_temporal_pipeline(config)
    print("✅ Created temporal feature pipeline")

    print("\n⚙️ Pipeline components:")
    for i, engine in enumerate(pipeline.feature_engines):
        print(f"  {i+1}. {engine.__class__.__name__}")

    print("\n🏭 Temporal feature pipeline ready!")
    print("Ready to generate comprehensive time-based features for forecasting.")