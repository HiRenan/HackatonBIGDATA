#!/usr/bin/env python3
"""
Phase 7: Post-Processing System
Advanced post-processing strategies for submission optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class PostProcessingResult:
    """Result of post-processing operation"""
    original_predictions: pd.DataFrame
    processed_predictions: pd.DataFrame
    transformation_applied: str
    improvement_estimate: float
    metadata: Dict[str, Any]
    processing_time: float

class BasePostProcessor(ABC):
    """Abstract base class for post-processors"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.weight = self.config.get('weight', 1.0)
        self.is_fitted = False

    @abstractmethod
    def fit(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'BasePostProcessor':
        """Fit the post-processor on predictions"""
        pass

    @abstractmethod
    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Transform predictions"""
        pass

    def fit_transform(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(predictions, context).transform(predictions)

    def estimate_improvement(self, original: pd.DataFrame, processed: pd.DataFrame) -> float:
        """Estimate improvement from post-processing"""
        # Simple heuristic: assume small improvements
        return 0.01  # 1% improvement estimate

class BusinessRulesPostProcessor(BasePostProcessor):
    """Apply business rules to predictions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.min_value = self.config.get('min_value', 0.0)
        self.max_value = self.config.get('max_value', None)
        self.growth_rate_limit = self.config.get('max_growth_rate', 2.0)
        self.seasonality_constraints = self.config.get('enable_seasonality_constraints', True)
        self.non_negativity = self.config.get('enable_non_negativity', True)

    def fit(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'BusinessRulesPostProcessor':
        """Fit business rules (mostly rule-based, minimal fitting)"""
        logger.info("Fitting business rules post-processor")

        # Calculate historical statistics if training data is available
        if context and 'historical_data' in context:
            historical_data = context['historical_data']
            if 'total_sales' in historical_data.columns:
                self.historical_mean = historical_data['total_sales'].mean()
                self.historical_std = historical_data['total_sales'].std()
                self.historical_max = historical_data['total_sales'].quantile(0.99)
            else:
                self.historical_mean = predictions['prediction'].mean()
                self.historical_std = predictions['prediction'].std()
                self.historical_max = predictions['prediction'].quantile(0.99)
        else:
            self.historical_mean = predictions['prediction'].mean()
            self.historical_std = predictions['prediction'].std()
            self.historical_max = predictions['prediction'].quantile(0.99)

        self.is_fitted = True
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules to predictions"""
        if not self.is_fitted:
            raise ValueError("PostProcessor must be fitted before transform")

        logger.info("Applying business rules to predictions")
        processed = predictions.copy()

        # Non-negativity constraint
        if self.non_negativity:
            negative_count = (processed['prediction'] < 0).sum()
            if negative_count > 0:
                logger.info(f"Clipping {negative_count} negative predictions to 0")
                processed['prediction'] = np.maximum(processed['prediction'], self.min_value)

        # Maximum value constraint
        if self.max_value is not None:
            exceeded_count = (processed['prediction'] > self.max_value).sum()
            if exceeded_count > 0:
                logger.info(f"Clipping {exceeded_count} predictions exceeding max value {self.max_value}")
                processed['prediction'] = np.minimum(processed['prediction'], self.max_value)

        # Extreme value constraint based on historical data
        extreme_threshold = self.historical_mean + 5 * self.historical_std
        extreme_count = (processed['prediction'] > extreme_threshold).sum()
        if extreme_count > 0:
            logger.info(f"Capping {extreme_count} extreme predictions at {extreme_threshold:.2f}")
            processed['prediction'] = np.minimum(processed['prediction'], extreme_threshold)

        # Growth rate constraints (if temporal data available)
        if 'date' in processed.columns and 'store_id' in processed.columns and 'product_id' in processed.columns:
            processed = self._apply_growth_constraints(processed)

        return processed

    def _apply_growth_constraints(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply growth rate constraints to predictions"""
        if self.growth_rate_limit <= 0:
            return predictions

        # Sort by date for temporal analysis
        predictions_sorted = predictions.sort_values(['store_id', 'product_id', 'date'])

        # Calculate period-over-period growth
        predictions_sorted['prev_prediction'] = predictions_sorted.groupby(['store_id', 'product_id'])['prediction'].shift(1)

        # Calculate growth rates
        predictions_sorted['growth_rate'] = (
            predictions_sorted['prediction'] / predictions_sorted['prev_prediction']
        ).fillna(1.0)

        # Cap excessive growth
        excessive_growth = predictions_sorted['growth_rate'] > self.growth_rate_limit
        if excessive_growth.any():
            capped_count = excessive_growth.sum()
            logger.info(f"Capping {capped_count} predictions with excessive growth")

            predictions_sorted.loc[excessive_growth, 'prediction'] = (
                predictions_sorted.loc[excessive_growth, 'prev_prediction'] * self.growth_rate_limit
            )

        # Remove temporary columns and restore original order
        predictions_sorted = predictions_sorted.drop(['prev_prediction', 'growth_rate'], axis=1)
        return predictions_sorted.sort_index()

class OutlierCappingPostProcessor(BasePostProcessor):
    """Cap outliers in predictions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.method = self.config.get('method', 'quantile')  # 'quantile', 'iqr', 'zscore'
        self.lower_quantile = self.config.get('lower_quantile', 0.01)
        self.upper_quantile = self.config.get('upper_quantile', 0.99)
        self.iqr_factor = self.config.get('iqr_factor', 1.5)
        self.zscore_threshold = self.config.get('zscore_threshold', 3.0)
        self.cap_values = {}

    def fit(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'OutlierCappingPostProcessor':
        """Fit outlier capping parameters"""
        logger.info(f"Fitting outlier capping using {self.method} method")

        prediction_values = predictions['prediction']

        if self.method == 'quantile':
            self.cap_values = {
                'lower': prediction_values.quantile(self.lower_quantile),
                'upper': prediction_values.quantile(self.upper_quantile)
            }

        elif self.method == 'iqr':
            q1 = prediction_values.quantile(0.25)
            q3 = prediction_values.quantile(0.75)
            iqr = q3 - q1
            self.cap_values = {
                'lower': q1 - self.iqr_factor * iqr,
                'upper': q3 + self.iqr_factor * iqr
            }

        elif self.method == 'zscore':
            mean_val = prediction_values.mean()
            std_val = prediction_values.std()
            self.cap_values = {
                'lower': mean_val - self.zscore_threshold * std_val,
                'upper': mean_val + self.zscore_threshold * std_val
            }

        logger.info(f"Outlier caps: lower={self.cap_values['lower']:.2f}, upper={self.cap_values['upper']:.2f}")
        self.is_fitted = True
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier capping"""
        if not self.is_fitted:
            raise ValueError("PostProcessor must be fitted before transform")

        logger.info("Applying outlier capping")
        processed = predictions.copy()

        original_count = len(processed)
        lower_outliers = (processed['prediction'] < self.cap_values['lower']).sum()
        upper_outliers = (processed['prediction'] > self.cap_values['upper']).sum()

        if lower_outliers > 0 or upper_outliers > 0:
            logger.info(f"Capping {lower_outliers} lower and {upper_outliers} upper outliers")

            processed['prediction'] = np.clip(
                processed['prediction'],
                self.cap_values['lower'],
                self.cap_values['upper']
            )

        return processed

class SeasonalAdjustmentPostProcessor(BasePostProcessor):
    """Apply seasonal adjustments to predictions"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.seasonal_factors = {}
        self.groupby_columns = self.config.get('groupby_columns', ['store_id', 'product_id'])
        self.seasonal_periods = self.config.get('seasonal_periods', [7, 30, 365])  # daily, monthly, yearly

    def fit(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'SeasonalAdjustmentPostProcessor':
        """Fit seasonal adjustment factors"""
        logger.info("Fitting seasonal adjustment factors")

        if 'date' not in predictions.columns:
            logger.warning("Date column not found, skipping seasonal adjustment")
            self.is_fitted = True
            return self

        # Ensure date is datetime
        predictions = predictions.copy()
        predictions['date'] = pd.to_datetime(predictions['date'])

        # Extract time components
        predictions['dayofweek'] = predictions['date'].dt.dayofweek
        predictions['day'] = predictions['date'].dt.day
        predictions['dayofyear'] = predictions['date'].dt.dayofyear

        # Calculate seasonal factors for different periods
        for period in self.seasonal_periods:
            if period == 7:  # Weekly seasonality
                time_col = 'dayofweek'
            elif period == 30:  # Monthly seasonality (using day of month)
                time_col = 'day'
            elif period == 365:  # Yearly seasonality
                time_col = 'dayofyear'
            else:
                continue

            # Calculate average predictions by time period
            if self.groupby_columns:
                seasonal_avgs = predictions.groupby(self.groupby_columns + [time_col])['prediction'].mean()
            else:
                seasonal_avgs = predictions.groupby(time_col)['prediction'].mean()

            self.seasonal_factors[period] = seasonal_avgs

        self.is_fitted = True
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply seasonal adjustments"""
        if not self.is_fitted or not self.seasonal_factors:
            logger.info("No seasonal factors fitted, returning original predictions")
            return predictions

        logger.info("Applying seasonal adjustments")
        processed = predictions.copy()

        if 'date' not in processed.columns:
            return processed

        # Ensure date is datetime
        processed['date'] = pd.to_datetime(processed['date'])

        # Extract time components
        processed['dayofweek'] = processed['date'].dt.dayofweek
        processed['day'] = processed['date'].dt.day
        processed['dayofyear'] = processed['date'].dt.dayofyear

        # Apply seasonal adjustments
        for period, factors in self.seasonal_factors.items():
            if period == 7:
                time_col = 'dayofweek'
            elif period == 30:
                time_col = 'day'
            elif period == 365:
                time_col = 'dayofyear'
            else:
                continue

            # Create adjustment factor
            adjustment_weight = 0.05  # 5% maximum adjustment

            if self.groupby_columns:
                # Map seasonal factors
                group_cols = self.groupby_columns + [time_col]
                seasonal_map = factors.to_dict()

                # Create index for mapping
                processed['group_key'] = processed[group_cols].apply(
                    lambda row: tuple(row), axis=1
                )

                # Apply adjustments where factors are available
                for idx, row in processed.iterrows():
                    if row['group_key'] in seasonal_map:
                        seasonal_factor = seasonal_map[row['group_key']]
                        overall_mean = processed['prediction'].mean()
                        if overall_mean > 0:
                            adjustment = (seasonal_factor / overall_mean - 1) * adjustment_weight
                            processed.loc[idx, 'prediction'] *= (1 + adjustment)

                processed = processed.drop('group_key', axis=1)
            else:
                # Global seasonal adjustment
                overall_mean = processed['prediction'].mean()
                for time_val in factors.index:
                    mask = processed[time_col] == time_val
                    if mask.any() and overall_mean > 0:
                        seasonal_factor = factors[time_val]
                        adjustment = (seasonal_factor / overall_mean - 1) * adjustment_weight
                        processed.loc[mask, 'prediction'] *= (1 + adjustment)

        # Clean up temporary columns
        processed = processed.drop(['dayofweek', 'day', 'dayofyear'], axis=1, errors='ignore')

        return processed

class EnsembleBlendingPostProcessor(BasePostProcessor):
    """Blend multiple predictions or smooth ensemble outputs"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.smoothing_factor = self.config.get('smoothing_factor', 0.1)
        self.blending_method = self.config.get('blending_method', 'moving_average')
        self.window_size = self.config.get('window_size', 7)

    def fit(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'EnsembleBlendingPostProcessor':
        """Fit blending parameters"""
        logger.info(f"Fitting ensemble blending with {self.blending_method} method")
        self.is_fitted = True
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply ensemble blending"""
        if not self.is_fitted:
            raise ValueError("PostProcessor must be fitted before transform")

        logger.info("Applying ensemble blending")
        processed = predictions.copy()

        if self.blending_method == 'moving_average':
            processed = self._apply_moving_average(processed)
        elif self.blending_method == 'exponential_smoothing':
            processed = self._apply_exponential_smoothing(processed)
        elif self.blending_method == 'median_filter':
            processed = self._apply_median_filter(processed)

        return processed

    def _apply_moving_average(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply moving average smoothing"""
        if 'date' in predictions.columns and len(predictions) > self.window_size:
            # Sort by date
            predictions_sorted = predictions.sort_values('date')

            # Apply moving average
            predictions_sorted['smoothed'] = predictions_sorted['prediction'].rolling(
                window=self.window_size, center=True, min_periods=1
            ).mean()

            # Blend original and smoothed
            predictions_sorted['prediction'] = (
                (1 - self.smoothing_factor) * predictions_sorted['prediction'] +
                self.smoothing_factor * predictions_sorted['smoothed']
            )

            predictions_sorted = predictions_sorted.drop('smoothed', axis=1)
            return predictions_sorted.sort_index()

        return predictions

    def _apply_exponential_smoothing(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply exponential smoothing"""
        if len(predictions) > 1:
            alpha = self.smoothing_factor
            smoothed_values = predictions['prediction'].ewm(alpha=alpha).mean()

            predictions = predictions.copy()
            predictions['prediction'] = (
                (1 - self.smoothing_factor) * predictions['prediction'] +
                self.smoothing_factor * smoothed_values
            )

        return predictions

    def _apply_median_filter(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply median filter smoothing"""
        if len(predictions) > self.window_size:
            predictions = predictions.copy()
            median_values = predictions['prediction'].rolling(
                window=self.window_size, center=True, min_periods=1
            ).median()

            predictions['prediction'] = (
                (1 - self.smoothing_factor) * predictions['prediction'] +
                self.smoothing_factor * median_values
            )

        return predictions

class PostProcessorPipeline:
    """Pipeline for chaining multiple post-processors"""

    def __init__(self, processors: List[BasePostProcessor]):
        self.processors = processors
        self.is_fitted = False

    def fit(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> 'PostProcessorPipeline':
        """Fit all processors in pipeline"""
        logger.info(f"Fitting post-processing pipeline with {len(self.processors)} processors")

        for processor in self.processors:
            if processor.enabled:
                processor.fit(predictions, context)

        self.is_fitted = True
        return self

    def transform(self, predictions: pd.DataFrame) -> PostProcessingResult:
        """Transform with all processors"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        start_time = datetime.now()
        original_predictions = predictions.copy()
        processed = predictions.copy()

        transformations_applied = []
        total_improvement = 0.0

        for processor in self.processors:
            if processor.enabled:
                try:
                    before_transform = processed.copy()
                    processed = processor.transform(processed)

                    # Estimate improvement
                    improvement = processor.estimate_improvement(before_transform, processed)
                    total_improvement += improvement * processor.weight

                    transformations_applied.append(processor.__class__.__name__)

                except Exception as e:
                    logger.error(f"Error in {processor.__class__.__name__}: {str(e)}")
                    continue

        processing_time = (datetime.now() - start_time).total_seconds()

        return PostProcessingResult(
            original_predictions=original_predictions,
            processed_predictions=processed,
            transformation_applied=", ".join(transformations_applied),
            improvement_estimate=total_improvement,
            metadata={
                'processors_used': len(transformations_applied),
                'processors_failed': len(self.processors) - len(transformations_applied)
            },
            processing_time=processing_time
        )

    def fit_transform(self, predictions: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> PostProcessingResult:
        """Fit and transform in one step"""
        return self.fit(predictions, context).transform(predictions)

def create_standard_pipeline(config: Optional[Dict[str, Any]] = None) -> PostProcessorPipeline:
    """Create standard post-processing pipeline"""
    config = config or {}

    processors = [
        BusinessRulesPostProcessor(config.get('business_rules', {})),
        OutlierCappingPostProcessor(config.get('outlier_capping', {})),
        SeasonalAdjustmentPostProcessor(config.get('seasonal_adjustment', {})),
        EnsembleBlendingPostProcessor(config.get('ensemble_blending', {}))
    ]

    return PostProcessorPipeline(processors)

def create_competitive_pipeline(competitive_intelligence: Dict[str, Any],
                               config: Optional[Dict[str, Any]] = None) -> PostProcessorPipeline:
    """Create competitive-informed post-processing pipeline"""
    config = config or {}

    # Adjust configuration based on competitive position
    position = competitive_intelligence.get('position_analysis', {})
    competitive_zone = position.get('competitive_zone', 'middle_pack')

    if competitive_zone == 'leader':
        # Conservative post-processing for leaders
        config.setdefault('business_rules', {})['enable_non_negativity'] = True
        config.setdefault('outlier_capping', {})['method'] = 'quantile'
        config.setdefault('ensemble_blending', {})['smoothing_factor'] = 0.05

    elif competitive_zone == 'contender':
        # Aggressive post-processing for contenders
        config.setdefault('outlier_capping', {})['upper_quantile'] = 0.98
        config.setdefault('ensemble_blending', {})['smoothing_factor'] = 0.15

    return create_standard_pipeline(config)

if __name__ == "__main__":
    # Demo usage
    print("🔄 Post-Processing System Demo")
    print("=" * 50)

    # Create sample predictions
    sample_predictions = pd.DataFrame({
        'store_id': [1, 1, 1, 2, 2, 2],
        'product_id': [100, 100, 100, 200, 200, 200],
        'date': pd.date_range('2023-01-01', periods=6),
        'prediction': [150.5, 200.3, -10.0, 300.8, 450.2, 1000.0]  # Including outliers
    })

    # Create pipeline
    pipeline = create_standard_pipeline()
    print(f"✅ Created pipeline with {len(pipeline.processors)} processors")

    # Fit and transform
    result = pipeline.fit_transform(sample_predictions)

    print(f"\nPost-processing results:")
    print(f"  Transformations: {result.transformation_applied}")
    print(f"  Estimated improvement: {result.improvement_estimate:.1%}")
    print(f"  Processing time: {result.processing_time:.3f}s")

    print(f"\nBefore/After comparison:")
    print(f"  Original mean: {result.original_predictions['prediction'].mean():.2f}")
    print(f"  Processed mean: {result.processed_predictions['prediction'].mean():.2f}")
    print(f"  Negative values removed: {(result.original_predictions['prediction'] < 0).sum()}")

    print("\n🔄 Post-processing system ready!")
    print("Ready to optimize predictions with advanced post-processing techniques.")


def create_post_processor(config: Optional[Union[str, Dict[str, Any]]] = None) -> PostProcessorPipeline:
    """Factory function to create a configured PostProcessorPipeline

    Args:
        config: Configuration for post-processor. Can be:
            - None: Creates standard pipeline with default config
            - str: Processor type ('business_rules', 'outlier_capping', etc.)
            - dict: Full configuration dictionary
    """
    if config is None:
        config = {}
    elif isinstance(config, str):
        # Convert string processor type to config dict
        processor_type = config
        config = {processor_type: {'enabled': True}}

    return create_standard_pipeline(config)