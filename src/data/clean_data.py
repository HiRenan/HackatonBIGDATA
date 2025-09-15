#!/usr/bin/env python3
"""
Phase 6: Data Cleaning Pipeline Script
Advanced data cleaning with business rules and quality assurance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import sys
from typing import Dict, Any, Optional, Tuple, List
import time
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessors import create_preprocessing_pipeline
from src.data.validators import create_transaction_validator
from src.config.phase6_config import get_config
from src.utils.logging import setup_logger

def clean_data_pipeline(transactions: pd.DataFrame,
                       products: pd.DataFrame,
                       stores: pd.DataFrame,
                       config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data cleaning pipeline

    Args:
        transactions: Raw transaction data
        products: Raw product data
        stores: Raw store data
        config: Configuration dictionary

    Returns:
        Tuple of cleaned (transactions, products, stores) DataFrames
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    logger.info("🧹 Starting data cleaning pipeline")
    logger.info(f"Input shapes - Transactions: {transactions.shape}, Products: {products.shape}, Stores: {stores.shape}")

    try:
        # Create preprocessing pipeline
        config = config or {}
        preprocessing_pipeline = create_preprocessing_pipeline(config.get('preprocessing', {}))

        # Prepare data dictionary
        data_dict = {
            'transactions': transactions.copy(),
            'products': products.copy(),
            'stores': stores.copy()
        }

        # Fit and transform data
        logger.info("🔄 Fitting preprocessing pipeline...")
        preprocessing_pipeline.fit(data_dict)

        logger.info("🔄 Transforming data...")
        cleaned_data = preprocessing_pipeline.transform(data_dict)

        # Extract cleaned datasets
        transactions_clean = cleaned_data['transactions']
        products_clean = cleaned_data['products']
        stores_clean = cleaned_data['stores']

        # Additional cleaning steps
        transactions_clean = _advanced_transaction_cleaning(transactions_clean, config)
        products_clean = _advanced_product_cleaning(products_clean, config)
        stores_clean = _advanced_store_cleaning(stores_clean, config)

        # Post-cleaning validation
        if config.get('validate_after_cleaning', True):
            logger.info("✅ Running post-cleaning validation...")
            _validate_cleaned_data(transactions_clean, products_clean, stores_clean)

        # Log results
        end_time = time.time()
        duration = end_time - start_time

        logger.info("📊 Cleaning results:")
        logger.info(f"  Transactions: {transactions.shape} → {transactions_clean.shape}")
        logger.info(f"  Products: {products.shape} → {products_clean.shape}")
        logger.info(f"  Stores: {stores.shape} → {stores_clean.shape}")
        logger.info(f"⏱️ Cleaning completed in {duration:.1f} seconds")

        return transactions_clean, products_clean, stores_clean

    except Exception as e:
        logger.error(f"❌ Data cleaning pipeline failed: {str(e)}")
        raise

def _advanced_transaction_cleaning(transactions: pd.DataFrame,
                                 config: Dict[str, Any]) -> pd.DataFrame:
    """Advanced transaction-specific cleaning"""
    logger = logging.getLogger(__name__)
    logger.info("🧹 Advanced transaction cleaning...")

    df = transactions.copy()
    initial_count = len(df)

    # Remove transactions with zero or negative sales (if configured)
    if config.get('remove_zero_sales', True):
        zero_sales = (df['total_sales'] == 0).sum()
        df = df[df['total_sales'] > 0]
        if zero_sales > 0:
            logger.info(f"  Removed {zero_sales:,} zero-sales transactions")

    # Remove extreme outliers in sales
    if config.get('remove_sales_outliers', True):
        q99 = df['total_sales'].quantile(0.99)
        q01 = df['total_sales'].quantile(0.01)
        outliers = ((df['total_sales'] > q99) | (df['total_sales'] < q01)).sum()
        df = df[(df['total_sales'] <= q99) & (df['total_sales'] >= q01)]
        if outliers > 0:
            logger.info(f"  Removed {outliers:,} sales outliers")

    # Fix quantity-price inconsistencies
    if 'quantity' in df.columns and 'unit_price' in df.columns:
        # Calculate expected total_sales
        expected_sales = df['quantity'] * df['unit_price']
        inconsistent = np.abs(df['total_sales'] - expected_sales) > 0.01
        inconsistent_count = inconsistent.sum()

        if inconsistent_count > 0:
            logger.warning(f"  Found {inconsistent_count:,} quantity-price inconsistencies")
            if config.get('fix_price_inconsistencies', True):
                # Use total_sales as truth, recalculate unit_price
                df.loc[inconsistent & (df['quantity'] > 0), 'unit_price'] = (
                    df.loc[inconsistent & (df['quantity'] > 0), 'total_sales'] /
                    df.loc[inconsistent & (df['quantity'] > 0), 'quantity']
                )
                logger.info(f"  Fixed {inconsistent_count:,} price inconsistencies")

    # Remove duplicate transactions
    if config.get('remove_duplicates', True):
        duplicate_cols = ['date', 'store_id', 'product_id']
        if all(col in df.columns for col in duplicate_cols):
            duplicates = df.duplicated(subset=duplicate_cols, keep='first').sum()
            df = df.drop_duplicates(subset=duplicate_cols, keep='first')
            if duplicates > 0:
                logger.info(f"  Removed {duplicates:,} duplicate transactions")

    # Date range filtering
    if config.get('date_range'):
        date_range = config['date_range']
        if 'start' in date_range and 'end' in date_range:
            start_date = pd.to_datetime(date_range['start'])
            end_date = pd.to_datetime(date_range['end'])

            before_filter = len(df)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            filtered_count = before_filter - len(df)

            if filtered_count > 0:
                logger.info(f"  Filtered {filtered_count:,} transactions outside date range")

    final_count = len(df)
    removed_count = initial_count - final_count
    removal_rate = removed_count / initial_count * 100

    logger.info(f"  Transaction cleaning: {initial_count:,} → {final_count:,} (-{removal_rate:.1f}%)")

    return df

def _advanced_product_cleaning(products: pd.DataFrame,
                             config: Dict[str, Any]) -> pd.DataFrame:
    """Advanced product-specific cleaning"""
    logger = logging.getLogger(__name__)
    logger.info("🧹 Advanced product cleaning...")

    df = products.copy()
    initial_count = len(df)

    # Remove products with missing critical information
    critical_columns = config.get('product_critical_columns', ['product_id', 'category'])
    for col in critical_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            df = df.dropna(subset=[col])
            if missing > 0:
                logger.info(f"  Removed {missing:,} products with missing {col}")

    # Clean category names
    if 'category' in df.columns:
        # Standardize category names
        df['category'] = df['category'].str.strip().str.title()

        # Group rare categories
        if config.get('group_rare_categories', True):
            min_category_count = config.get('min_category_count', 100)
            category_counts = df['category'].value_counts()
            rare_categories = category_counts[category_counts < min_category_count].index

            if len(rare_categories) > 0:
                df.loc[df['category'].isin(rare_categories), 'category'] = 'Other'
                logger.info(f"  Grouped {len(rare_categories)} rare categories as 'Other'")

    # Handle price anomalies
    price_columns = ['unit_price', 'unit_cost']
    for col in price_columns:
        if col in df.columns:
            # Remove negative prices
            negative_prices = (df[col] < 0).sum()
            df = df[df[col] >= 0]
            if negative_prices > 0:
                logger.info(f"  Removed {negative_prices:,} products with negative {col}")

            # Cap extreme prices
            q99 = df[col].quantile(0.99)
            extreme_prices = (df[col] > q99 * 10).sum()  # 10x the 99th percentile
            df.loc[df[col] > q99 * 10, col] = q99
            if extreme_prices > 0:
                logger.info(f"  Capped {extreme_prices:,} extreme {col} values")

    # Remove duplicate products
    if config.get('remove_duplicate_products', True):
        duplicates = df.duplicated(subset=['product_id'], keep='first').sum()
        df = df.drop_duplicates(subset=['product_id'], keep='first')
        if duplicates > 0:
            logger.info(f"  Removed {duplicates:,} duplicate products")

    final_count = len(df)
    removed_count = initial_count - final_count
    removal_rate = removed_count / initial_count * 100 if initial_count > 0 else 0

    logger.info(f"  Product cleaning: {initial_count:,} → {final_count:,} (-{removal_rate:.1f}%)")

    return df

def _advanced_store_cleaning(stores: pd.DataFrame,
                           config: Dict[str, Any]) -> pd.DataFrame:
    """Advanced store-specific cleaning"""
    logger = logging.getLogger(__name__)
    logger.info("🧹 Advanced store cleaning...")

    df = stores.copy()
    initial_count = len(df)

    # Remove stores with missing critical information
    critical_columns = config.get('store_critical_columns', ['store_id', 'region'])
    for col in critical_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            df = df.dropna(subset=[col])
            if missing > 0:
                logger.info(f"  Removed {missing:,} stores with missing {col}")

    # Clean region names
    if 'region' in df.columns:
        df['region'] = df['region'].str.strip().str.title()

    # Handle coordinate anomalies
    coordinate_columns = ['latitude', 'longitude']
    for col in coordinate_columns:
        if col in df.columns:
            # Remove invalid coordinates (e.g., 0,0 or extreme values)
            if col == 'latitude':
                invalid = ((df[col] < -90) | (df[col] > 90) | (df[col] == 0)).sum()
                df = df[(df[col] >= -90) & (df[col] <= 90) & (df[col] != 0)]
            elif col == 'longitude':
                invalid = ((df[col] < -180) | (df[col] > 180) | (df[col] == 0)).sum()
                df = df[(df[col] >= -180) & (df[col] <= 180) & (df[col] != 0)]

            if invalid > 0:
                logger.info(f"  Removed {invalid:,} stores with invalid {col}")

    # Handle store size anomalies
    if 'store_size' in df.columns:
        # Remove stores with zero or negative size
        invalid_size = (df['store_size'] <= 0).sum()
        df = df[df['store_size'] > 0]
        if invalid_size > 0:
            logger.info(f"  Removed {invalid_size:,} stores with invalid size")

        # Cap extreme sizes
        q99 = df['store_size'].quantile(0.99)
        extreme_sizes = (df['store_size'] > q99 * 5).sum()
        df.loc[df['store_size'] > q99 * 5, 'store_size'] = q99
        if extreme_sizes > 0:
            logger.info(f"  Capped {extreme_sizes:,} extreme store sizes")

    # Remove duplicate stores
    if config.get('remove_duplicate_stores', True):
        duplicates = df.duplicated(subset=['store_id'], keep='first').sum()
        df = df.drop_duplicates(subset=['store_id'], keep='first')
        if duplicates > 0:
            logger.info(f"  Removed {duplicates:,} duplicate stores")

    final_count = len(df)
    removed_count = initial_count - final_count
    removal_rate = removed_count / initial_count * 100 if initial_count > 0 else 0

    logger.info(f"  Store cleaning: {initial_count:,} → {final_count:,} (-{removal_rate:.1f}%)")

    return df

def _validate_cleaned_data(transactions: pd.DataFrame,
                         products: pd.DataFrame,
                         stores: pd.DataFrame):
    """Validate cleaned data integrity"""
    logger = logging.getLogger(__name__)

    # Check referential integrity
    transaction_stores = set(transactions['store_id'].unique())
    available_stores = set(stores['store_id'].unique())
    orphaned_stores = transaction_stores - available_stores

    if orphaned_stores:
        logger.warning(f"Found {len(orphaned_stores)} orphaned store references")

    transaction_products = set(transactions['product_id'].unique())
    available_products = set(products['product_id'].unique())
    orphaned_products = transaction_products - available_products

    if orphaned_products:
        logger.warning(f"Found {len(orphaned_products)} orphaned product references")

    # Check data consistency
    if len(transactions) == 0:
        raise ValueError("No transactions remaining after cleaning")

    if len(products) == 0:
        raise ValueError("No products remaining after cleaning")

    if len(stores) == 0:
        raise ValueError("No stores remaining after cleaning")

    logger.info("✅ Post-cleaning validation passed")

def main():
    """Main entry point for data cleaning script"""
    parser = argparse.ArgumentParser(description="Clean competition data with business rules")

    parser.add_argument("--input-path", "-i", type=str, required=True,
                       help="Path to input data (raw or processed)")
    parser.add_argument("--output-path", "-o", type=str, required=True,
                       help="Path to save cleaned data")
    parser.add_argument("--config", "-c", type=str,
                       help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("data_cleaning", level=args.log_level)

    # Load configuration
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config("development")

    try:
        # Load input data
        input_path = Path(args.input_path)
        logger.info(f"📁 Loading data from {input_path}")

        if input_path.is_dir():
            # Load from directory
            transactions = pd.read_parquet(input_path / "transactions.parquet")
            products = pd.read_parquet(input_path / "products.parquet")
            stores = pd.read_parquet(input_path / "stores.parquet")
        else:
            logger.error("Input path must be a directory containing parquet files")
            return 1

        # Run cleaning pipeline
        logger.info("🚀 Starting data cleaning pipeline")

        transactions_clean, products_clean, stores_clean = clean_data_pipeline(
            transactions, products, stores, config.get('data_cleaning', {})
        )

        # Save cleaned data
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving cleaned data to {output_path}")

        transactions_clean.to_parquet(output_path / "transactions_clean.parquet",
                                    compression='snappy', index=False)
        products_clean.to_parquet(output_path / "products_clean.parquet",
                                 compression='snappy', index=False)
        stores_clean.to_parquet(output_path / "stores_clean.parquet",
                               compression='snappy', index=False)

        logger.info("✅ Data cleaning completed successfully!")

        # Print summary
        print("\n" + "="*60)
        print("🧹 DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Transactions: {len(transactions):,} → {len(transactions_clean):,}")
        print(f"Products:     {len(products):,} → {len(products_clean):,}")
        print(f"Stores:       {len(stores):,} → {len(stores_clean):,}")
        print("="*60)

        return 0

    except KeyboardInterrupt:
        logger.info("❌ Data cleaning interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {str(e)}")
        logger.debug("Full error traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())