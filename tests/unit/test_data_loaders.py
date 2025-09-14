#!/usr/bin/env python3
"""
Phase 6: Unit Tests for Data Loaders
Comprehensive test suite for data loading utilities
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import modules to test
from src.data.loaders import (
    BaseDataLoader, ParquetLoader, TransactionLoader,
    ProductLoader, StoreLoader, DataLoaderFactory,
    load_competition_data
)

class TestBaseDataLoader:
    """Test the abstract base class"""

    def test_init_default_config(self):
        """Test initialization with default config"""
        # Create a concrete implementation for testing
        class ConcreteLoader(BaseDataLoader):
            def load(self, path, **kwargs):
                return pd.DataFrame()
            def validate_data(self, data):
                return True

        loader = ConcreteLoader()
        assert loader.config == {}
        assert loader.chunk_size == 100000
        assert loader.max_memory_gb == 8.0

    def test_init_custom_config(self):
        """Test initialization with custom config"""
        class ConcreteLoader(BaseDataLoader):
            def load(self, path, **kwargs):
                return pd.DataFrame()
            def validate_data(self, data):
                return True

        config = {
            'chunk_size': 50000,
            'max_memory_usage_gb': 16.0
        }
        loader = ConcreteLoader(config)
        assert loader.chunk_size == 50000
        assert loader.max_memory_gb == 16.0

    def test_get_memory_usage_mb(self):
        """Test memory usage calculation"""
        class ConcreteLoader(BaseDataLoader):
            def load(self, path, **kwargs):
                return pd.DataFrame()
            def validate_data(self, data):
                return True

        loader = ConcreteLoader()

        # Create test dataframe
        data = pd.DataFrame({
            'col1': range(1000),
            'col2': ['test'] * 1000
        })

        memory_mb = loader.get_memory_usage_mb(data)
        assert isinstance(memory_mb, float)
        assert memory_mb > 0

class TestParquetLoader:
    """Test ParquetLoader functionality"""

    @pytest.fixture
    def sample_parquet_file(self):
        """Create a temporary parquet file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            # Create sample data
            data = pd.DataFrame({
                'id': range(1000),
                'value': np.random.randn(1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000),
                'date': pd.date_range('2023-01-01', periods=1000, freq='H')
            })

            # Write to parquet
            data.to_parquet(tmp.name, index=False)

            yield tmp.name

            # Cleanup
            Path(tmp.name).unlink(missing_ok=True)

    def test_init_default(self):
        """Test ParquetLoader initialization"""
        loader = ParquetLoader()
        assert isinstance(loader, BaseDataLoader)
        assert loader.chunk_size == 100000

    def test_load_full_file(self, sample_parquet_file):
        """Test loading complete parquet file"""
        loader = ParquetLoader()
        data = loader.load(sample_parquet_file)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert list(data.columns) == ['id', 'value', 'category', 'date']

    def test_load_with_columns(self, sample_parquet_file):
        """Test loading specific columns"""
        loader = ParquetLoader()
        data = loader.load(sample_parquet_file, columns=['id', 'value'])

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000
        assert list(data.columns) == ['id', 'value']

    def test_load_with_sample_size(self, sample_parquet_file):
        """Test loading with sample size"""
        loader = ParquetLoader()
        data = loader.load(sample_parquet_file, sample_size=500)

        assert isinstance(data, pd.DataFrame)
        assert len(data) <= 500  # Should be <= because of sampling

    def test_validate_data_empty(self):
        """Test validation with empty data"""
        loader = ParquetLoader()
        empty_data = pd.DataFrame()

        assert not loader.validate_data(empty_data)

    def test_validate_data_normal(self):
        """Test validation with normal data"""
        loader = ParquetLoader()
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

        assert loader.validate_data(data)

    def test_optimize_memory(self):
        """Test memory optimization"""
        loader = ParquetLoader()

        # Create data with optimizable types
        data = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'] * 250,  # Low cardinality
            'large_int': [1000000] * 1000,
            'large_float': [3.14159] * 1000
        })

        original_memory = loader.get_memory_usage_mb(data)
        optimized_data = loader._optimize_memory(data)
        optimized_memory = loader.get_memory_usage_mb(optimized_data)

        # Memory should be reduced or same
        assert optimized_memory <= original_memory
        assert optimized_data['category'].dtype.name == 'category'

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        loader = ParquetLoader()

        with pytest.raises(Exception):  # Should raise some kind of file error
            loader.load("nonexistent_file.parquet")

class TestTransactionLoader:
    """Test TransactionLoader functionality"""

    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction parquet file"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=1000, freq='D'),
                'store_id': np.random.randint(1, 100, 1000),
                'product_id': np.random.randint(1, 500, 1000),
                'quantity': np.random.randint(1, 10, 1000),
                'unit_price': np.random.uniform(1, 100, 1000),
                'total_sales': np.random.uniform(10, 1000, 1000)
            })

            data.to_parquet(tmp.name, index=False)
            yield tmp.name

            Path(tmp.name).unlink(missing_ok=True)

    def test_load_transaction_data(self, sample_transaction_data):
        """Test loading transaction data with processing"""
        loader = TransactionLoader()
        data = loader.load(sample_transaction_data)

        assert isinstance(data, pd.DataFrame)
        assert 'date' in data.columns
        assert pd.api.types.is_datetime64_any_dtype(data['date'])

    def test_validate_transaction_data_valid(self, sample_transaction_data):
        """Test validation with valid transaction data"""
        loader = TransactionLoader()
        data = loader.load(sample_transaction_data)

        assert loader.validate_data(data)

    def test_validate_transaction_data_missing_columns(self):
        """Test validation with missing required columns"""
        loader = TransactionLoader()
        data = pd.DataFrame({
            'some_column': [1, 2, 3]
        })

        assert not loader.validate_data(data)

    def test_validate_transaction_data_negative_sales(self):
        """Test validation with negative sales (should warn but pass)"""
        loader = TransactionLoader()
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'store_id': [1, 2, 3],
            'product_id': [1, 2, 3],
            'total_sales': [100, -50, 200]  # Negative value
        })

        # Should still pass validation (just warning)
        assert loader.validate_data(data)

class TestProductLoader:
    """Test ProductLoader functionality"""

    @pytest.fixture
    def sample_product_data(self):
        """Create sample product parquet file"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data = pd.DataFrame({
                'product_id': range(1, 101),
                'category': np.random.choice(['Electronics', 'Clothing', 'Food'], 100),
                'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC'], 100),
                'price': np.random.uniform(10, 500, 100)
            })

            data.to_parquet(tmp.name, index=False)
            yield tmp.name

            Path(tmp.name).unlink(missing_ok=True)

    def test_validate_product_data_valid(self, sample_product_data):
        """Test validation with valid product data"""
        loader = ProductLoader()
        data = pd.read_parquet(sample_product_data)

        assert loader.validate_data(data)

    def test_validate_product_data_missing_columns(self):
        """Test validation with missing required columns"""
        loader = ProductLoader()
        data = pd.DataFrame({
            'some_column': [1, 2, 3]
        })

        assert not loader.validate_data(data)

    def test_validate_product_data_duplicates(self):
        """Test validation with duplicate product IDs"""
        loader = ProductLoader()
        data = pd.DataFrame({
            'product_id': [1, 1, 2],  # Duplicate ID
            'category': ['A', 'B', 'C']
        })

        # Should still pass validation (just warning)
        assert loader.validate_data(data)

class TestStoreLoader:
    """Test StoreLoader functionality"""

    @pytest.fixture
    def sample_store_data(self):
        """Create sample store parquet file"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data = pd.DataFrame({
                'store_id': range(1, 51),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
                'size': np.random.uniform(100, 1000, 50),
                'city': [f'City_{i}' for i in range(50)]
            })

            data.to_parquet(tmp.name, index=False)
            yield tmp.name

            Path(tmp.name).unlink(missing_ok=True)

    def test_validate_store_data_valid(self, sample_store_data):
        """Test validation with valid store data"""
        loader = StoreLoader()
        data = pd.read_parquet(sample_store_data)

        assert loader.validate_data(data)

    def test_validate_store_data_missing_columns(self):
        """Test validation with missing required columns"""
        loader = StoreLoader()
        data = pd.DataFrame({
            'some_column': [1, 2, 3]
        })

        assert not loader.validate_data(data)

class TestDataLoaderFactory:
    """Test DataLoaderFactory functionality"""

    def test_create_transaction_loader(self):
        """Test creating transaction loader"""
        loader = DataLoaderFactory.create_loader('transaction')
        assert isinstance(loader, TransactionLoader)

    def test_create_product_loader(self):
        """Test creating product loader"""
        loader = DataLoaderFactory.create_loader('product')
        assert isinstance(loader, ProductLoader)

    def test_create_store_loader(self):
        """Test creating store loader"""
        loader = DataLoaderFactory.create_loader('store')
        assert isinstance(loader, StoreLoader)

    def test_create_parquet_loader(self):
        """Test creating generic parquet loader"""
        loader = DataLoaderFactory.create_loader('parquet')
        assert isinstance(loader, ParquetLoader)

    def test_create_unknown_loader(self):
        """Test creating unknown loader type"""
        with pytest.raises(ValueError, match="Unknown data type"):
            DataLoaderFactory.create_loader('unknown')

    def test_create_loader_with_config(self):
        """Test creating loader with custom config"""
        config = {'chunk_size': 50000}
        loader = DataLoaderFactory.create_loader('parquet', config)
        assert loader.chunk_size == 50000

class TestLoadCompetitionData:
    """Test the load_competition_data function"""

    @pytest.fixture
    def sample_data_directory(self):
        """Create a temporary directory with sample data files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create sample transaction data
            trans_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'store_id': np.random.randint(1, 10, 100),
                'product_id': np.random.randint(1, 20, 100),
                'total_sales': np.random.uniform(10, 500, 100)
            })
            trans_data.to_parquet(data_dir / 'transactions.parquet', index=False)

            # Create sample product data
            prod_data = pd.DataFrame({
                'product_id': range(1, 21),
                'category': np.random.choice(['A', 'B', 'C'], 20)
            })
            prod_data.to_parquet(data_dir / 'products.parquet', index=False)

            # Create sample store data
            store_data = pd.DataFrame({
                'store_id': range(1, 11),
                'region': np.random.choice(['North', 'South'], 10)
            })
            store_data.to_parquet(data_dir / 'stores.parquet', index=False)

            yield data_dir

    def test_load_competition_data_success(self, sample_data_directory):
        """Test successful loading of all competition data"""
        transactions, products, stores = load_competition_data(sample_data_directory)

        assert isinstance(transactions, pd.DataFrame)
        assert isinstance(products, pd.DataFrame)
        assert isinstance(stores, pd.DataFrame)

        assert len(transactions) == 100
        assert len(products) == 20
        assert len(stores) == 10

    def test_load_competition_data_with_config(self, sample_data_directory):
        """Test loading competition data with custom config"""
        config = {'chunk_size': 50}
        transactions, products, stores = load_competition_data(sample_data_directory, config)

        assert isinstance(transactions, pd.DataFrame)
        assert len(transactions) == 100  # Should still load all data

    def test_load_competition_data_missing_directory(self):
        """Test loading from non-existent directory"""
        with pytest.raises(Exception):
            load_competition_data("nonexistent_directory")

# Integration tests
class TestDataLoadersIntegration:
    """Integration tests for data loaders"""

    def test_loader_pipeline_flow(self):
        """Test the complete flow from factory to loading"""
        # Create in-memory test data
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'store_id': range(1, 11),
                'product_id': range(1, 11),
                'total_sales': range(100, 1100, 100)
            })
            data.to_parquet(tmp.name, index=False)

            try:
                # Use factory to create loader
                loader = DataLoaderFactory.create_loader('transaction')

                # Load data
                loaded_data = loader.load(tmp.name)

                # Validate
                assert loader.validate_data(loaded_data)
                assert len(loaded_data) == 10
                assert 'date' in loaded_data.columns

            finally:
                Path(tmp.name).unlink(missing_ok=True)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])