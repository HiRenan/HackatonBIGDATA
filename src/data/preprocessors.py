#!/usr/bin/env python3
"""
Phase 6: Data Preprocessing Utilities
Advanced data preprocessing with memory optimization and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_memory_gb = self.config.get('max_memory_usage_gb', 8.0)
        self.enable_validation = self.config.get('enable_validation', True)
        self._statistics = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BasePreprocessor':
        """Fit the preprocessor on training data"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)

    def get_memory_usage_mb(self, data: pd.DataFrame) -> float:
        """Calculate memory usage in MB"""
        return data.memory_usage(deep=True).sum() / (1024 ** 2)

    def check_memory_limit(self, data: pd.DataFrame):
        """Check if data exceeds memory limit"""
        memory_gb = self.get_memory_usage_mb(data) / 1024
        if memory_gb > self.max_memory_gb:
            logger.warning(f"Data memory usage ({memory_gb:.1f}GB) exceeds limit ({self.max_memory_gb}GB)")

class TransactionPreprocessor(BasePreprocessor):
    """Specialized preprocessor for transaction data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.date_column = self.config.get('date_column', 'date')
        self.sales_column = self.config.get('sales_column', 'total_sales')
        self.handle_outliers = self.config.get('handle_outliers', True)
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.min_date = None
        self.max_date = None
        self.sales_stats = {}

    def fit(self, data: pd.DataFrame) -> 'TransactionPreprocessor':
        """Fit preprocessor on transaction data"""
        logger.info("Fitting transaction preprocessor")

        # Store date range
        if self.date_column in data.columns:
            self.min_date = data[self.date_column].min()
            self.max_date = data[self.date_column].max()
            logger.info(f"Date range: {self.min_date} to {self.max_date}")

        # Calculate sales statistics for outlier detection
        if self.sales_column in data.columns:
            self.sales_stats = {
                'mean': data[self.sales_column].mean(),
                'std': data[self.sales_column].std(),
                'median': data[self.sales_column].median(),
                'q25': data[self.sales_column].quantile(0.25),
                'q75': data[self.sales_column].quantile(0.75)
            }
            logger.info(f"Sales statistics: {self.sales_stats}")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform transaction data"""
        logger.info(f"Transforming {len(data)} transaction records")
        data_transformed = data.copy()

        # Date processing
        if self.date_column in data_transformed.columns:
            data_transformed[self.date_column] = pd.to_datetime(
                data_transformed[self.date_column], errors='coerce'
            )

            # Add temporal features
            data_transformed['year'] = data_transformed[self.date_column].dt.year
            data_transformed['month'] = data_transformed[self.date_column].dt.month
            data_transformed['day'] = data_transformed[self.date_column].dt.day
            data_transformed['dayofweek'] = data_transformed[self.date_column].dt.dayofweek
            data_transformed['quarter'] = data_transformed[self.date_column].dt.quarter
            data_transformed['is_weekend'] = data_transformed['dayofweek'].isin([5, 6])

        # Numeric columns processing
        numeric_columns = ['quantity', 'unit_price', 'total_sales']
        for col in numeric_columns:
            if col in data_transformed.columns:
                # Convert to numeric
                data_transformed[col] = pd.to_numeric(
                    data_transformed[col], errors='coerce'
                )

                # Handle outliers if enabled
                if self.handle_outliers and col == self.sales_column:
                    data_transformed = self._handle_outliers(data_transformed, col)

        # Handle missing values
        data_transformed = self._handle_missing_values(data_transformed)

        # Optimize data types
        data_transformed = self._optimize_dtypes(data_transformed)

        self.check_memory_limit(data_transformed)
        logger.info(f"Transformation complete: {data_transformed.shape}")

        return data_transformed

    def _handle_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Handle outliers in sales data"""
        if column not in self.sales_stats:
            return data

        mean = self.sales_stats['mean']
        std = self.sales_stats['std']
        threshold = self.outlier_threshold

        # Z-score method
        z_scores = np.abs((data[column] - mean) / std)
        outliers = z_scores > threshold

        outlier_count = outliers.sum()
        if outlier_count > 0:
            logger.warning(f"Found {outlier_count} outliers in {column}")

            # Cap outliers at threshold
            upper_limit = mean + threshold * std
            lower_limit = max(0, mean - threshold * std)  # Sales can't be negative

            data[column] = data[column].clip(lower=lower_limit, upper=upper_limit)

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in transaction data"""
        # Critical columns that shouldn't have missing values
        critical_columns = ['store_id', 'product_id']

        for col in critical_columns:
            if col in data.columns:
                missing_count = data[col].isnull().sum()
                if missing_count > 0:
                    logger.error(f"Found {missing_count} missing values in critical column {col}")
                    # Drop rows with missing critical values
                    data = data.dropna(subset=[col])

        # Fill missing sales with 0 (represents no sale)
        if 'total_sales' in data.columns:
            data['total_sales'] = data['total_sales'].fillna(0)

        if 'quantity' in data.columns:
            data['quantity'] = data['quantity'].fillna(0)

        return data

    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        # Categorical columns
        categorical_columns = ['store_id', 'product_id']
        for col in categorical_columns:
            if col in data.columns:
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.5:
                    data[col] = data[col].astype('category')

        # Integer columns
        int_columns = ['year', 'month', 'day', 'dayofweek', 'quarter', 'quantity']
        for col in int_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], downcast='integer', errors='ignore')

        # Boolean columns
        bool_columns = ['is_weekend']
        for col in bool_columns:
            if col in data.columns:
                data[col] = data[col].astype('bool')

        return data

class ProductPreprocessor(BasePreprocessor):
    """Specialized preprocessor for product data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.encoding_strategy = self.config.get('encoding_strategy', 'frequency')
        self.min_frequency = self.config.get('min_frequency', 10)
        self.category_encodings = {}

    def fit(self, data: pd.DataFrame) -> 'ProductPreprocessor':
        """Fit preprocessor on product data"""
        logger.info("Fitting product preprocessor")

        # Calculate category frequencies for encoding
        categorical_columns = ['category', 'subcategory', 'brand']
        for col in categorical_columns:
            if col in data.columns:
                value_counts = data[col].value_counts()
                if self.encoding_strategy == 'frequency':
                    # Use frequency encoding
                    self.category_encodings[col] = value_counts.to_dict()
                else:
                    # Use label encoding for high cardinality
                    frequent_values = value_counts[value_counts >= self.min_frequency].index
                    encoding = {val: i for i, val in enumerate(frequent_values)}
                    encoding['__other__'] = len(encoding)
                    self.category_encodings[col] = encoding

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform product data"""
        logger.info(f"Transforming {len(data)} product records")
        data_transformed = data.copy()

        # Handle categorical encoding
        for col, encoding in self.category_encodings.items():
            if col in data_transformed.columns:
                if self.encoding_strategy == 'frequency':
                    # Frequency encoding
                    data_transformed[f'{col}_frequency'] = data_transformed[col].map(
                        encoding
                    ).fillna(0)
                else:
                    # Label encoding with unknown handling
                    data_transformed[f'{col}_encoded'] = data_transformed[col].map(
                        encoding
                    ).fillna(encoding.get('__other__', -1))

        # Handle numeric product attributes
        numeric_columns = ['unit_cost', 'unit_price', 'weight']
        for col in numeric_columns:
            if col in data_transformed.columns:
                data_transformed[col] = pd.to_numeric(
                    data_transformed[col], errors='coerce'
                )
                # Fill missing with median
                median_value = data_transformed[col].median()
                data_transformed[col] = data_transformed[col].fillna(median_value)

        # Create derived features
        data_transformed = self._create_product_features(data_transformed)

        # Optimize data types
        data_transformed = self._optimize_dtypes(data_transformed)

        logger.info(f"Product transformation complete: {data_transformed.shape}")
        return data_transformed

    def _create_product_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for products"""
        # Price-related features
        if 'unit_price' in data.columns and 'unit_cost' in data.columns:
            data['profit_margin'] = (
                data['unit_price'] - data['unit_cost']
            ) / data['unit_price'].replace(0, np.nan)
            data['profit_margin'] = data['profit_margin'].fillna(0)

        # Create price tiers
        if 'unit_price' in data.columns:
            data['price_tier'] = pd.cut(
                data['unit_price'],
                bins=[0, 10, 50, 200, np.inf],
                labels=['low', 'medium', 'high', 'premium'],
                include_lowest=True
            )

        return data

    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for product data"""
        # Categorical columns
        categorical_columns = ['product_id', 'category', 'subcategory', 'brand', 'price_tier']
        for col in categorical_columns:
            if col in data.columns and data[col].dtype == 'object':
                data[col] = data[col].astype('category')

        # Float columns
        float_columns = ['unit_cost', 'unit_price', 'weight', 'profit_margin']
        for col in float_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], downcast='float', errors='ignore')

        return data

class StorePreprocessor(BasePreprocessor):
    """Specialized preprocessor for store data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.region_encodings = {}
        self.size_categories = self.config.get('size_categories', ['small', 'medium', 'large'])

    def fit(self, data: pd.DataFrame) -> 'StorePreprocessor':
        """Fit preprocessor on store data"""
        logger.info("Fitting store preprocessor")

        # Create region encodings
        if 'region' in data.columns:
            regions = data['region'].unique()
            self.region_encodings = {region: i for i, region in enumerate(regions)}

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform store data"""
        logger.info(f"Transforming {len(data)} store records")
        data_transformed = data.copy()

        # Encode regions
        if 'region' in data_transformed.columns:
            data_transformed['region_encoded'] = data_transformed['region'].map(
                self.region_encodings
            )

        # Handle store size
        if 'store_size' in data_transformed.columns:
            data_transformed['store_size'] = pd.to_numeric(
                data_transformed['store_size'], errors='coerce'
            )

            # Create size categories
            data_transformed['size_category'] = pd.cut(
                data_transformed['store_size'],
                bins=3,
                labels=self.size_categories[:3]
            )

        # Handle coordinates
        coordinate_columns = ['latitude', 'longitude']
        for col in coordinate_columns:
            if col in data_transformed.columns:
                data_transformed[col] = pd.to_numeric(
                    data_transformed[col], errors='coerce'
                )

        # Optimize data types
        data_transformed = self._optimize_dtypes(data_transformed)

        logger.info(f"Store transformation complete: {data_transformed.shape}")
        return data_transformed

    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for store data"""
        # Categorical columns
        categorical_columns = ['store_id', 'region', 'size_category']
        for col in categorical_columns:
            if col in data.columns and data[col].dtype == 'object':
                data[col] = data[col].astype('category')

        # Numeric columns
        numeric_columns = ['store_size', 'latitude', 'longitude']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], downcast='float', errors='ignore')

        return data

class PreprocessorPipeline:
    """Pipeline for chaining multiple preprocessors"""

    def __init__(self, preprocessors: List[BasePreprocessor]):
        self.preprocessors = preprocessors
        self.is_fitted = False

    def fit(self, data: Dict[str, pd.DataFrame]) -> 'PreprocessorPipeline':
        """Fit all preprocessors"""
        logger.info("Fitting preprocessor pipeline")

        for i, preprocessor in enumerate(self.preprocessors):
            if hasattr(preprocessor, 'data_type'):
                data_key = preprocessor.data_type
                if data_key in data:
                    preprocessor.fit(data[data_key])
                    logger.info(f"Fitted preprocessor {i+1}/{len(self.preprocessors)}")

        self.is_fitted = True
        return self

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform all data"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        logger.info("Transforming data with pipeline")
        transformed_data = {}

        for key, df in data.items():
            transformed_data[key] = df.copy()

        # Apply relevant preprocessors
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'data_type'):
                data_key = preprocessor.data_type
                if data_key in transformed_data:
                    transformed_data[data_key] = preprocessor.transform(
                        transformed_data[data_key]
                    )

        return transformed_data

def create_preprocessing_pipeline(config: Optional[Dict[str, Any]] = None) -> PreprocessorPipeline:
    """Create a complete preprocessing pipeline"""
    config = config or {}

    # Create specialized preprocessors
    transaction_preprocessor = TransactionPreprocessor(config.get('transaction', {}))
    transaction_preprocessor.data_type = 'transactions'

    product_preprocessor = ProductPreprocessor(config.get('product', {}))
    product_preprocessor.data_type = 'products'

    store_preprocessor = StorePreprocessor(config.get('store', {}))
    store_preprocessor.data_type = 'stores'

    preprocessors = [
        transaction_preprocessor,
        product_preprocessor,
        store_preprocessor
    ]

    return PreprocessorPipeline(preprocessors)

if __name__ == "__main__":
    # Demo usage
    print("🔄 Data Preprocessors Demo")
    print("=" * 50)

    # Test configuration
    config = {
        'transaction': {
            'handle_outliers': True,
            'outlier_threshold': 3.0
        },
        'product': {
            'encoding_strategy': 'frequency',
            'min_frequency': 10
        },
        'store': {
            'size_categories': ['small', 'medium', 'large']
        }
    }

    # Create pipeline
    pipeline = create_preprocessing_pipeline(config)
    print("✅ Created preprocessing pipeline")

    print("\n🔄 Pipeline components:")
    for i, preprocessor in enumerate(pipeline.preprocessors):
        print(f"  {i+1}. {preprocessor.__class__.__name__}")

    print("\n🏭 Preprocessing pipeline ready!")
    print("Ready to handle transaction, product, and store data preprocessing.")