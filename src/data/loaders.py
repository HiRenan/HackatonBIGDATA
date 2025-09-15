#!/usr/bin/env python3
"""
Phase 6: Data Loading Utilities
Specialized data loaders with memory optimization and validation
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 100000)
        self.max_memory_gb = self.config.get('max_memory_usage_gb', 8.0)

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from source"""
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data"""
        pass

    def get_memory_usage_mb(self, data: pd.DataFrame) -> float:
        """Calculate memory usage in MB"""
        return data.memory_usage(deep=True).sum() / (1024 ** 2)

class ParquetLoader(BaseDataLoader):
    """Optimized Parquet file loader"""

    def load(self, path: Union[str, Path],
             sample_size: Optional[int] = None,
             columns: Optional[List[str]] = None,
             filters: Optional[List[Tuple]] = None) -> pd.DataFrame:
        """
        Load parquet file with optimizations

        Args:
            path: Path to parquet file
            sample_size: Number of rows to sample
            columns: Specific columns to load
            filters: PyArrow filters to apply

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading parquet file: {path}")

        try:
            parquet_file = pq.ParquetFile(path)

            # Get file metadata
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Total rows in file: {total_rows}")

            if sample_size and sample_size < total_rows:
                # Sample random row groups
                num_row_groups = parquet_file.num_row_groups
                sample_row_groups = np.random.choice(
                    num_row_groups,
                    size=min(10, num_row_groups),
                    replace=False
                )

                chunks = []
                current_rows = 0

                for rg_idx in sample_row_groups:
                    if current_rows >= sample_size:
                        break

                    chunk = parquet_file.read_row_group(
                        rg_idx,
                        columns=columns,
                        use_pandas_metadata=True
                    ).to_pandas()

                    remaining = sample_size - current_rows
                    if len(chunk) > remaining:
                        chunk = chunk.sample(n=remaining, random_state=42)

                    chunks.append(chunk)
                    current_rows += len(chunk)

                data = pd.concat(chunks, ignore_index=True)
                logger.info(f"Sampled {len(data)} rows from {total_rows}")

            else:
                # Load full file or with filters
                data = parquet_file.read(
                    columns=columns,
                    filters=filters,
                    use_pandas_metadata=True
                ).to_pandas()

            # Memory optimization
            data = self._optimize_memory(data)

            # Validation
            if not self.validate_data(data):
                raise ValueError("Data validation failed")

            memory_mb = self.get_memory_usage_mb(data)
            logger.info(f"Loaded data: {data.shape}, Memory: {memory_mb:.1f}MB")

            return data

        except Exception as e:
            logger.error(f"Failed to load parquet file {path}: {e}")
            raise

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate parquet data"""
        if data.empty:
            logger.error("Loaded data is empty")
            return False

        # Check memory usage
        memory_gb = self.get_memory_usage_mb(data) / 1024
        if memory_gb > self.max_memory_gb:
            logger.warning(f"Data memory usage ({memory_gb:.1f}GB) exceeds limit ({self.max_memory_gb}GB)")

        # Check for basic data integrity
        if data.isnull().all().any():
            logger.warning("Some columns are entirely null")

        return True

    def _optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        logger.debug("Optimizing memory usage")

        for col in data.select_dtypes(include=['object']):
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.5:  # Convert to category if < 50% unique values
                data[col] = data[col].astype('category')

        # Downcast numeric types
        for col in data.select_dtypes(include=['int64']):
            data[col] = pd.to_numeric(data[col], downcast='integer')

        for col in data.select_dtypes(include=['float64']):
            data[col] = pd.to_numeric(data[col], downcast='float')

        return data

class TransactionLoader(ParquetLoader):
    """Specialized loader for transaction data"""

    def load(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load transaction data with specific optimizations"""
        data = super().load(path, **kwargs)

        # Transaction-specific processing
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])

        # Ensure numeric columns
        numeric_columns = ['quantity', 'unit_price', 'total_sales']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate transaction data specifically"""
        if not super().validate_data(data):
            return False

        required_columns = ['date', 'store_id', 'product_id']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for negative values in sales
        if 'total_sales' in data.columns:
            negative_sales = (data['total_sales'] < 0).sum()
            if negative_sales > 0:
                logger.warning(f"Found {negative_sales} negative sales values")

        return True

class ProductLoader(ParquetLoader):
    """Specialized loader for product data"""

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate product data specifically"""
        if not super().validate_data(data):
            return False

        required_columns = ['product_id', 'category']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for duplicate product IDs
        if 'product_id' in data.columns:
            duplicates = data['product_id'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate product IDs")

        return True

class StoreLoader(ParquetLoader):
    """Specialized loader for store data"""

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate store data specifically"""
        if not super().validate_data(data):
            return False

        required_columns = ['store_id', 'region']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for duplicate store IDs
        if 'store_id' in data.columns:
            duplicates = data['store_id'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate store IDs")

        return True

class DataLoaderFactory:
    """Factory for creating appropriate data loaders"""

    @staticmethod
    def create_loader(data_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDataLoader:
        """Create appropriate loader based on data type"""
        loaders = {
            'transaction': TransactionLoader,
            'product': ProductLoader,
            'store': StoreLoader,
            'parquet': ParquetLoader
        }

        if data_type not in loaders:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(loaders.keys())}")

        return loaders[data_type](config)

def load_competition_data(data_path: Union[str, Path],
                         config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all competition data files

    Args:
        data_path: Path to data directory
        config: Configuration dictionary

    Returns:
        Tuple of (transactions, products, stores) DataFrames
    """
    data_path = Path(data_path)
    config = config or {}

    logger.info(f"Loading competition data from: {data_path}")

    # Load transactions
    trans_loader = DataLoaderFactory.create_loader('transaction', config)
    transactions = trans_loader.load(data_path / 'transactions.parquet')

    # Load products
    prod_loader = DataLoaderFactory.create_loader('product', config)
    products = prod_loader.load(data_path / 'products.parquet')

    # Load stores
    store_loader = DataLoaderFactory.create_loader('store', config)
    stores = store_loader.load(data_path / 'stores.parquet')

    logger.info("All data files loaded successfully")

    return transactions, products, stores


if __name__ == "__main__":
    # Demo usage
    print("📁 Data Loaders Demo")
    print("=" * 50)

    # Test with sample configuration
    config = {
        'chunk_size': 50000,
        'max_memory_usage_gb': 4.0
    }

    # Create loaders
    trans_loader = DataLoaderFactory.create_loader('transaction', config)
    print(f"✅ Created transaction loader: {trans_loader.__class__.__name__}")

    prod_loader = DataLoaderFactory.create_loader('product', config)
    print(f"✅ Created product loader: {prod_loader.__class__.__name__}")

    store_loader = DataLoaderFactory.create_loader('store', config)
    print(f"✅ Created store loader: {store_loader.__class__.__name__}")

    print("\n🏭 Factory pattern working correctly!")
    print("Ready to load competition data with optimized memory usage.")