#!/usr/bin/env python3
"""
Phase 6: PyTest Configuration and Fixtures
Central configuration for all tests with comprehensive fixtures
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, Generator, Optional
import logging
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.phase6_config import get_config, ConfigManager
from architecture.factories import model_factory, feature_factory, evaluation_factory
from architecture.observers import event_publisher, TrainingObserver, ValidationObserver
from utils.data_loader import load_data_efficiently

# Disable logging during tests
logging.disable(logging.CRITICAL)

# =====================================================
# Session-scoped fixtures (run once per test session)
# =====================================================

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Load test configuration"""
    config_manager = ConfigManager()
    config = config_manager.load_config("testing")
    return config

@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture(scope="session")
def test_data_dir(temp_dir: Path) -> Path:
    """Create test data directory"""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    (data_dir / "features").mkdir(exist_ok=True)
    return data_dir

# =====================================================
# Function-scoped fixtures (run for each test)
# =====================================================

@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Generate sample transaction data for testing"""
    np.random.seed(42)

    n_samples = 1000
    start_date = datetime(2023, 1, 1)

    data = {
        'transaction_id': range(1, n_samples + 1),
        'date': [start_date + timedelta(days=i % 365) for i in range(n_samples)],
        'store_id': np.random.randint(1, 101, n_samples),
        'product_id': np.random.randint(1, 501, n_samples),
        'quantity': np.random.poisson(5, n_samples) + 1,
        'unit_price': np.random.uniform(10, 100, n_samples),
        'total_sales': np.random.uniform(50, 500, n_samples),
        'customer_id': np.random.randint(1, 10001, n_samples),
        'promotion_id': np.random.choice([None, 1, 2, 3, 4], n_samples, p=[0.7, 0.075, 0.075, 0.075, 0.075])
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_products() -> pd.DataFrame:
    """Generate sample product data for testing"""
    np.random.seed(42)

    n_products = 500
    categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
    subcategories = {
        'Electronics': ['Phones', 'Laptops', 'Cameras'],
        'Clothing': ['Shirts', 'Pants', 'Shoes'],
        'Food': ['Snacks', 'Beverages', 'Frozen'],
        'Home': ['Furniture', 'Decor', 'Kitchen'],
        'Sports': ['Equipment', 'Apparel', 'Accessories']
    }

    data = {
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product_{i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'subcategory': [
            np.random.choice(subcategories[cat])
            for cat in np.random.choice(categories, n_products)
        ],
        'brand': [f'Brand_{np.random.randint(1, 51)}' for _ in range(n_products)],
        'unit_cost': np.random.uniform(5, 80, n_products),
        'launch_date': [
            datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095))
            for _ in range(n_products)
        ],
        'is_seasonal': np.random.choice([True, False], n_products, p=[0.3, 0.7]),
        'lifecycle_stage': np.random.choice(['Launch', 'Growth', 'Mature', 'Decline'], n_products)
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_stores() -> pd.DataFrame:
    """Generate sample store data for testing"""
    np.random.seed(42)

    n_stores = 100
    regions = ['North', 'South', 'East', 'West', 'Central']
    store_types = ['Mall', 'Standalone', 'Strip', 'Department']

    data = {
        'store_id': range(1, n_stores + 1),
        'store_name': [f'Store_{i}' for i in range(1, n_stores + 1)],
        'region': np.random.choice(regions, n_stores),
        'city': [f'City_{np.random.randint(1, 21)}' for _ in range(n_stores)],
        'store_type': np.random.choice(store_types, n_stores),
        'size_sqft': np.random.randint(1000, 10000, n_stores),
        'opening_date': [
            datetime(2015, 1, 1) + timedelta(days=np.random.randint(0, 2555))
            for _ in range(n_stores)
        ],
        'is_flagship': np.random.choice([True, False], n_stores, p=[0.1, 0.9])
    }

    return pd.DataFrame(data)

@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """Generate time series data for testing"""
    np.random.seed(42)

    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_days = len(dates)

    # Generate realistic time series with trend and seasonality
    trend = np.linspace(100, 150, n_days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)  # Yearly
    weekly = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly
    noise = np.random.normal(0, 5, n_days)

    values = trend + seasonal + weekly + noise
    values = np.maximum(values, 0)  # Ensure non-negative

    return pd.DataFrame({
        'date': dates,
        'value': values,
        'store_id': 1,
        'product_id': 1
    })

@pytest.fixture
def ml_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Generate ML-ready dataset for testing"""
    np.random.seed(42)

    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create realistic target variable
    weights = np.random.randn(n_features)
    y = pd.Series(X @ weights + np.random.normal(0, 0.1, n_samples))

    return X, y

# =====================================================
# Mock fixtures
# =====================================================

@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing"""
    with patch('mlflow.start_run'), \
         patch('mlflow.log_param'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'), \
         patch('mlflow.sklearn.log_model'):
        yield

@pytest.fixture
def mock_database():
    """Mock database connections"""
    with patch('psycopg2.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        yield mock_conn, mock_cursor

@pytest.fixture
def mock_model_training():
    """Mock model training for faster tests"""
    with patch('sklearn.ensemble.RandomForestRegressor.fit'), \
         patch('lightgbm.LGBMRegressor.fit'), \
         patch('prophet.Prophet.fit'):
        yield

# =====================================================
# Factory fixtures
# =====================================================

@pytest.fixture
def clean_factories():
    """Clean factory registries before/after tests"""
    # Store original registries
    original_model_registry = model_factory._registry.copy()
    original_feature_registry = feature_factory._registry.copy()
    original_evaluation_registry = evaluation_factory._registry.copy()

    yield

    # Restore original registries
    model_factory._registry = original_model_registry
    feature_factory._registry = original_feature_registry
    evaluation_factory._registry = original_evaluation_registry

@pytest.fixture
def clean_observers():
    """Clean event publisher observers before/after tests"""
    # Store original observers
    original_observers = event_publisher._observers.copy()

    yield

    # Restore original observers
    event_publisher._observers = original_observers
    # Clear event history
    event_publisher.event_history.clear()

# =====================================================
# Test data persistence fixtures
# =====================================================

@pytest.fixture
def save_test_data(test_data_dir: Path, sample_transactions: pd.DataFrame,
                   sample_products: pd.DataFrame, sample_stores: pd.DataFrame):
    """Save test data to files for integration tests"""
    # Save as parquet files
    sample_transactions.to_parquet(test_data_dir / "raw" / "transactions.parquet")
    sample_products.to_parquet(test_data_dir / "raw" / "products.parquet")
    sample_stores.to_parquet(test_data_dir / "raw" / "stores.parquet")

    return test_data_dir

# =====================================================
# Performance testing fixtures
# =====================================================

@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    np.random.seed(42)

    n_samples = 100000
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add categorical columns
    X['category_1'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    X['category_2'] = np.random.choice(range(1, 101), n_samples)

    # Create target
    y = pd.Series(np.random.randn(n_samples))

    return X, y

# =====================================================
# Test utilities
# =====================================================

@pytest.fixture
def assert_helpers():
    """Helper functions for assertions"""
    class AssertHelpers:
        @staticmethod
        def assert_dataframe_not_empty(df: pd.DataFrame, message: str = "DataFrame is empty"):
            assert not df.empty, message

        @staticmethod
        def assert_no_nulls(df: pd.DataFrame, columns: list = None):
            if columns is None:
                columns = df.columns
            for col in columns:
                assert not df[col].isnull().any(), f"Column {col} contains null values"

        @staticmethod
        def assert_positive_values(series: pd.Series, message: str = "Contains non-positive values"):
            assert (series > 0).all(), message

        @staticmethod
        def assert_wmape_reasonable(wmape_value: float, max_acceptable: float = 1.0):
            assert 0 <= wmape_value <= max_acceptable, f"WMAPE {wmape_value} is not reasonable"

        @staticmethod
        def assert_model_exists(model):
            assert model is not None, "Model is None"
            assert hasattr(model, 'fit'), "Model does not have fit method"
            assert hasattr(model, 'predict'), "Model does not have predict method"

    return AssertHelpers()

# =====================================================
# Test environment setup
# =====================================================

def pytest_configure(config):
    """Configure pytest environment"""
    # Set environment variable for testing
    os.environ['FORECAST_ENV'] = 'testing'

    # Create results directory if it doesn't exist
    results_dir = Path("tests/results")
    results_dir.mkdir(exist_ok=True)

def pytest_unconfigure(config):
    """Clean up after all tests"""
    # Reset logging
    logging.disable(logging.NOTSET)

def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    for item in items:
        # Add markers based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

# =====================================================
# Parametrize helpers
# =====================================================

# Common model configurations for parameterized tests
MODEL_CONFIGS = [
    ("lightgbm", {"n_estimators": 10, "learning_rate": 0.3}),
    ("prophet", {"yearly_seasonality": False, "weekly_seasonality": False}),
]

FEATURE_TYPES = ["temporal", "aggregation", "behavioral", "business"]

VALIDATION_STRATEGIES = ["timeseries_cv", "walk_forward"]